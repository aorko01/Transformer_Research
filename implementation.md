# Implementation Notes: `train.py` and `data_loader.py`

This document explains, at the statement/mechanism level, what the two files
actually do at runtime — not a high-level summary. It's meant to be read
alongside the source.

---

## 1. `data_loader.py`

### 1.1 `DiskTokenDataset(IterableDataset)`

Constructed with `(hf_source, tokenizer, max_seq_length)`, where `hf_source`
is a 🤗 `datasets.Dataset` slice (memory-mapped Arrow table on disk — no
in-RAM copy of the corpus).

**`__iter__` — per-worker sharding (lines 33-43)**

```python
worker_info = get_worker_info()
source = self.hf_source
if worker_info is not None:
    source = source.shard(num_shards=worker_info.num_workers, index=worker_info.id)
```

`get_worker_info()` returns `None` in the main process and a populated object
inside each `DataLoader` worker subprocess. Because `IterableDataset.__iter__`
runs independently *inside each worker process* (they don't share Python
state), without this shard call every worker would iterate the **entire**
dataset from the start, so with `num_workers=4` you'd get each example
yielded 4 times per epoch. `.shard(num_shards=N, index=i)` gives worker `i`
every `N`-th article (a disjoint, deterministic partition), so the union
across workers reconstructs the full dataset exactly once.

**Per-article tokenization + chunking (lines 44-75)**

For each article:
```python
enc = self.tokenizer(
    text, truncation=True, max_length=self.max_seq_length,
    return_overflowing_tokens=True,      # <- key line
    return_special_tokens_mask=True,
    padding=False, return_tensors=None,
)
```
`return_overflowing_tokens=True` tells the fast tokenizer: instead of
discarding everything past `max_seq_length`, split the article into
consecutive windows of at most `max_seq_length` tokens each. The result is a
dict where every value (`input_ids`, `attention_mask`,
`special_tokens_mask`, possibly `token_type_ids`) is a **list of chunks**
(list-of-lists), one entry per window, rather than a single flat sequence.
`enc.pop("overflow_to_sample_mapping", None)` discards the auxiliary index
the tokenizer adds to say "which original example did chunk *i* come from"
— irrelevant here since every chunk in this call came from the same single
article.

```python
n_chunks = len(enc["input_ids"])
for i in range(n_chunks):
    chunk = {k: v[i] for k, v in enc.items()}
    seq_len = len(chunk["input_ids"])
```
Each chunk is unpacked into its own per-example dict.

**Last-chunk handling (lines 65-74)** — only applies to `i == n_chunks - 1`
(the final, possibly short, chunk of the article):
- If the last chunk's length is `< max_seq_length / 2` tokens, it's
  `continue`'d (dropped entirely) — too little content to be a useful
  training example.
- Otherwise it's right-padded up to `max_seq_length`:
  - `input_ids` padded with `tokenizer.pad_token_id`
  - `attention_mask` padded with `0` (so the model's self-attention ignores
    those positions)
  - `special_tokens_mask` padded with `1` — this is what later tells
    `DataCollatorForLanguageModeling` "never MLM-mask these positions,"
    which correctly excludes pad tokens from being selected as masked
    prediction targets.
  - `token_type_ids`, if present, padded with `0`.

  Non-final chunks (`i < n_chunks - 1`) are always exactly
  `max_seq_length` long already (that's how the tokenizer's overflow
  splitting works), so they skip the padding branch.

```python
yield {k: torch.tensor(v) for k, v in chunk.items()}
```
Each yielded item is a dict of 1-D `torch.LongTensor`s, all of identical
length `max_seq_length`. This is what the plain `DataLoader` default
collate function will later stack into a batch (see §2.6).

### 1.2 `load_wikipedia_dataset(...)`

```python
dataset = load_dataset("wikimedia/wikipedia", "20231101.en",
                        split="train", cache_dir=data_dir,
                        trust_remote_code=True)
```
Downloads (or reuses, if `cache_dir` already has it) the full English
Wikipedia 2023-11-01 dump as Arrow files on disk. `datasets.Dataset` objects
are memory-mapped — `len(dataset)` and `dataset[i]` don't require the whole
corpus to be resident in RAM.

**Train/val split (lines 105-110)** — index-based, not random:
```python
val_hf = dataset.select(range(max_val))                       # first max_val articles
train_hf = dataset.select(range(max_val, min(max_val+max_train, len(dataset))))  # or to end if max_train is None
train_hf = train_hf.shuffle(seed=42)
```
So validation is always the *first* `max_val` articles in the underlying
dataset ordering, and training is the remainder (optionally capped at
`max_train` articles starting right after the validation slice).
`.shuffle(seed=42)` on a `Dataset` only permutes an internal index array —
the underlying Arrow file/mmap is untouched, so this is cheap regardless of
corpus size. Note: this shuffles at the **article** level, not the chunk or
token level — chunks derived from the same article are still emitted
consecutively within `DiskTokenDataset.__iter__`, just in shuffled article
order.

Both `train_hf` and `val_hf` are wrapped in `DiskTokenDataset` and returned.

---

## 2. `train.py`

### 2.1 Config dictionaries (lines 43-73)

`BERT_SMALL_CONFIG` is passed directly as `BertConfig(**BERT_SMALL_CONFIG)`.
Current values: `hidden_size=768, num_hidden_layers=12,
num_attention_heads=1, intermediate_size=3072` — i.e. BERT-Base's
dimensions, but with attention heads forced to 1 (single-head vanilla
scaled dot-product attention instead of BERT-Base's 12 heads). Since
`hidden_size % num_attention_heads == 0` is satisfied trivially (`768 % 1
== 0`), the standard HF `BertSelfAttention` handles this without any code
changes: it just never splits the 768-dim projection into multiple heads.

`TRAINING_DEFAULTS` sets the optimizer/schedule/data hyperparameters
(these are all overridable via CLI, see §2.10). Notably `batch_size=32,
gradient_accumulation_steps=8` → effective batch size 256.

`CHECKPOINT_EVERY = 5000` — controls both validation frequency *and*
checkpoint frequency (they're coupled: validation only ever runs right
before a checkpoint is written).

`PROFILE_WARMUP_ITERS = 30`, `PROFILE_ITERS = 5` — see §2.3.

### 2.2 `LayerProfileHook`

Registers a forward **pre**-hook and forward **post**-hook on every leaf
module (`if list(module.children()): continue` skips container modules —
only modules with no children, e.g. `nn.Linear`, `nn.LayerNorm`, `nn.
Dropout`, get instrumented).

- Pre-hook: if CUDA is available, `torch.cuda.reset_peak_memory_stats()` —
  this resets the *global* peak-memory counter, not a per-module one, so
  the "peak" recorded is whatever the single highest allocation was
  anywhere in the process between this pre-hook and the matching
  post-hook. It also stamps `self._pending[tag] = time.perf_counter()`.
- Post-hook: computes `elapsed_ms` from the stamp, reads
  `torch.cuda.max_memory_allocated()` as the peak VRAM for that module's
  forward call, and accumulates both into a running sum + count in
  `self.stats[tag]`.
- `averages()` divides each accumulated sum by its count → per-layer mean
  VRAM/time across however many forward passes were profiled.

Because `reset_peak_memory_stats()` is called inside *every* leaf module's
pre-hook, and these hooks fire in call order during a single forward pass,
nested/sequential leaf modules effectively each get their own "local"
window between reset points — this only works correctly because PyTorch
calls pre/post hooks for leaf modules serially (no leaf module calls
another leaf module inside its own forward).

**Driving logic in `train()` (lines 419-422, 477-492)**: `fwd_count` counts
*training* forward passes only (validation forward passes are not counted
here — `run_epoch` never touches the hook or `fwd_count`). On the 31st
training forward pass (`fwd_count == PROFILE_WARMUP_ITERS + 1`), the hook
is attached. On the 36th (`fwd_count == PROFILE_WARMUP_ITERS +
PROFILE_ITERS + 1`), it's detached, `hook.averages()` is written into
`run_meta["layer_profile"]`, and `_save_metrics` flushes it to disk
immediately — this happens exactly once per run (`profiled` flag guards
re-entry), so profiling numbers reflect forward passes 31–35 specifically
(after 30 warmup iterations let CUDA's allocator/cuDNN autotuner settle).

### 2.3 `swap_attention_layers(model, cls, config)`

```python
for _, module in model.named_modules():
    if isinstance(module, BertAttention):
        module.self = cls(config)
```
`transformers`' `BertAttention` wraps a `.self` (the actual
`BertSelfAttention` Q/K/V + softmax logic) and a `.output` (dense +
dropout + residual + LayerNorm). This loop finds every `BertAttention`
wrapper in the model (one per transformer layer) and replaces just the
`.self` submodule with an instance of the user-supplied class — `.output`,
residual connections, and everything else in `BertLayer` stay untouched.
The replacement class must accept `(config)` in its constructor and return
`(context_layer,)` or `(context_layer, attn_weights)` from `forward`, matching
`BertSelfAttention`'s contract (see `custom_attention_template.py` for the
reference implementation `LinearAttention`, which is mathematically
identical standard multi/single-head attention rewritten from scratch).
This path is **not exercised** unless `--custom_attention module.Class` is
passed; by default the model just uses the library's standard
`BertSelfAttention`.

### 2.4 Checkpoint helpers

- `find_latest_checkpoint(out_dir)`: globs `checkpoint_step_*.pt` and picks
  the one with the largest numeric step, parsed via
  `int(p.stem.rsplit("_", 1)[1])` — e.g. `checkpoint_step_20000.pt` →
  `20000`. (Sorting by numeric value rather than filename string is
  required because filename string order does not match numeric order
  once step counts have different digit counts.)
- `load_checkpoint(path, model, optimizer, scheduler, scaler, device)`:
  `torch.load`s the checkpoint dict and calls `.load_state_dict` on model,
  optimizer, and scheduler. Scaler state is restored only if
  `"scaler_state_dict"` is present in the checkpoint (defensive — older
  checkpoints saved before AMP was added wouldn't have it). Returns
  `(epoch, global_step, best_val_loss)` so the training loop can resume
  counters exactly where they left off.
- `load_existing_metrics(metrics_path)`: if `metrics.json` already exists
  in the output dir, loads it (so a resumed run appends to the same
  history instead of overwriting it); otherwise returns `{}`.

### 2.5 `run_epoch(...)` — validation only

Despite the generic name and signature (it accepts `optimizer`,
`scheduler`, `is_train`), the docstring is explicit: *"is_train=True path
is no longer called"* — the training loop was inlined directly into
`train()` and this function is now only invoked with `is_train=False` for
validation (lines 540-541). `model.eval()` is called unconditionally at
the top; the unused `optimizer`/`scheduler`/`cfg`/`is_train`/`max_steps`
parameters are dead parameters left over from that refactor (this is also
what the Pylance "not accessed" diagnostics on those names are flagging —
harmless, but the signature could be trimmed).

For each batch from `val_loader`:
```python
keys  = list(batch.keys())
items = [{k: batch[k][j] for k in keys} for j in range(len(batch[keys[0]]))]
masked = collator(items)
```
This is a re-batching step. `val_loader`'s default collate already stacked
`DiskTokenDataset`'s per-example tensors into batch tensors, e.g.
`batch["input_ids"]` has shape `(B, T)`. But `DataCollatorForLanguageModeling`
expects a **list of per-example dicts** (its `__call__` signature mirrors
what a `Dataset.__getitem__` would return, not a pre-batched tensor dict),
so this list comprehension un-batches back to `[{"input_ids": tensor(T),
...}, ...] × B` before handing it to `collator`. The collator then:
1. Pads/stacks them back into batch tensors (a no-op here since every
   example is already exactly `max_seq_length` long).
2. Applies MLM masking: for each token not flagged by
   `special_tokens_mask`, with probability `mlm_probability` (0.15) it's
   selected as a *prediction target* (its true id goes into `labels`,
   itself overwritten in `input_ids` per the standard 80/10/10 BERT
   masking scheme — 80% `[MASK]`, 10% random token, 10% unchanged), while
   `labels` for all non-selected positions is set to `-100` (the ignore
   index for `CrossEntropyLoss`).

Then:
```python
with torch.amp.autocast('cuda'):
    outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
```
Passing `labels` directly to `BertForMaskedLM` makes it compute
cross-entropy loss internally (ignoring `-100` positions) and return it as
`outputs.loss`, alongside `outputs.logits` of shape `(B, T, vocab_size)`.

Accuracy is computed only over actually-masked positions:
```python
mask = labels != -100
total_correct += (logits.argmax(-1)[mask] == labels[mask]).sum().item()
total_masked  += mask.sum().item()
```
Final returned dict includes loss, masked-token accuracy, perplexity
(`exp(loss)`, clamped to `min(loss, 20)` before exponentiating to avoid
`OverflowError` on pathologically large losses early in training),
wall-clock time, and batch count.

### 2.6 `train(args)` — setup phase

- Resolves `device` (`cuda` if available else `cpu`), creates
  `output_dir`.
- Merges `TRAINING_DEFAULTS` with CLI overrides into `cfg`.
- Loads `BertTokenizerFast.from_pretrained("bert-base-uncased")` — note
  the tokenizer/vocab is bert-base's regardless of the model's own hidden
  size/layer count, since `vocab_size=30522` in `BERT_SMALL_CONFIG` is
  that tokenizer's vocab size.
- Builds `BertConfig(**BERT_SMALL_CONFIG)` and `BertForMaskedLM(bert_cfg)`
  — a randomly initialized model (no `from_pretrained` — this is
  pretraining from scratch, not fine-tuning).
- If `--custom_attention module.Class` was passed: splits the dotted path,
  inserts the module's parent dir onto `sys.path`, imports it, and calls
  `swap_attention_layers` (§2.3).
- `model.to(device)`.
- Calls `load_wikipedia_dataset(...)` (§1.2) to get `train_ds`, `val_ds`.
- Builds a `DataCollatorForLanguageModeling` once (shared between train
  and val loops).
- `loader_kwargs`: `num_workers`, `pin_memory=True` (faster host→device
  copy), `persistent_workers=num_workers>0` (keeps worker processes alive
  across the `for batch in loader` — `iter(train_loader)` calls instead of
  respawning them every epoch), `prefetch_factor=4` (each worker
  pre-fetches 4 batches ahead).
- `train_loader` uses `drop_last=True` (so gradient-accumulation math
  always sees full-size micro-batches — a partial final batch would skew
  the loss/accuracy accumulation); `val_loader` uses double the batch size
  (no gradients to store, so more fits in memory) and keeps the final
  partial batch.

### 2.7 Optimizer / scheduler / scaler

```python
no_decay = {"bias", "LayerNorm.weight"}
optimizer = AdamW([
    {"params": [... if not any(nd in n for nd in no_decay)], "weight_decay": cfg["weight_decay"]},
    {"params": [... if any(nd in n for nd in no_decay)],     "weight_decay": 0.0},
], lr=..., betas=..., eps=...)
```
Standard BERT-style weight-decay exclusion: bias terms and LayerNorm gains
don't get L2-regularized, matching the original BERT training recipe.

`scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps,
total_steps)` where `total_steps = cfg["max_steps"]`. Both
`warmup_steps`/`total_steps` are counted in **optimizer steps**
(`global_step`), not micro-batches — this is consistent with where
`scheduler.step()` is actually called (§2.9, only once per
`accum_steps` micro-batches).

`scaler = torch.amp.GradScaler('cuda')` — standard mixed-precision loss
scaling, works with the `torch.amp.autocast('cuda')` context used in both
the training and validation forward passes.

### 2.8 Resume logic

```python
resume_path = args.resume_from_checkpoint
if resume_path is None and args.auto_resume:
    latest = find_latest_checkpoint(out_dir)
    if latest: resume_path = str(latest)
if resume_path:
    start_epoch, start_global_step, best_val_loss = load_checkpoint(...)
```
`--resume_from_checkpoint` and `--auto_resume` are mutually exclusive CLI
flags (`argparse` mutually-exclusive group, §2.10). If neither is given,
training starts fresh from step 0. Note: resuming does **not** replay or
skip already-consumed batches from `train_loader` — `train_iter =
iter(train_loader)` (line 440) always starts a fresh pass over the
dataset regardless of `start_global_step`. Since `train_ds` is an
`IterableDataset` with no persistent cursor saved/restored, a resumed run
effectively restarts the current epoch's data order from the beginning
(article shuffle is deterministic via `seed=42`, so it's the same order,
just replayed from position 0 rather than wherever the crash happened).

### 2.9 Main training loop (lines 424-592)

State kept across iterations:
- `accum_steps`, `epoch` (last **completed** epoch, initialized from
  resume), `global_step` (optimizer steps, initialized from resume),
  `micro_step` (counts forward/backward calls, **never reset**, drives
  the `% accum_steps` gradient-accumulation boundary — note this is not
  restored from the checkpoint on resume, so it restarts at 0; since only
  `micro_step % accum_steps` matters and it's reset-invariant modulo
  arithmetic this doesn't misalign accumulation, it just means the first
  resumed accumulation window may be shorter/longer relative to the
  crash point, not a correctness issue).
- `ep_loss`, `ep_correct`, `ep_masked` — kept as **on-device** scalar
  tensors (`torch.zeros((), device=device)`), accumulated with in-place
  `+=` every micro-batch without ever calling `.item()`. The comment at
  lines 431-432 explains why: `.item()`/`.any()` force a host-device sync,
  which would stall the async CUDA kernel queue on every single
  micro-step. These are only materialized to Python floats
  (`.item()`) at epoch boundaries and at the (much rarer)
  `global_step % accum_steps == 0` boundary where an optimizer step
  actually happens — not every micro-step.

**Getting the next batch (lines 446-472)**:
```python
try:
    batch = next(train_iter)
except StopIteration:
    # epoch boundary: flush epoch-level train metrics, then...
    train_iter = iter(train_loader)
    batch = next(train_iter)
```
Because `train_ds` is an `IterableDataset` streaming from disk, one full
pass through it (all articles → all chunks) is one epoch; `StopIteration`
signals that boundary. On boundary: `epoch` is incremented, and if any
batches were accumulated (`ep_batches > 0`) a `train_m` metrics dict
(loss/accuracy/perplexity/time — same shape as validation's) is appended
to `run_meta["epochs"]` and printed, then the on-device accumulators are
zeroed and a new `train_iter` is created (re-shards workers, restarts the
shuffled article order — same seed, so same order again) so training
continues seamlessly into the next epoch without missing the batch that
triggered `StopIteration` being lost (the `batch = next(train_iter)` right
after gets the real first batch of the new epoch).

**Forward + profiling (lines 474-507)**: `model.train()`, then the
profiling counter/hook logic from §2.2, then the same
un-batch→collate→re-batch step as validation (§2.5), then forward pass
under `autocast`.

**Backward + accumulation (lines 509-517)**:
```python
scaler.scale(loss / accum_steps).backward()
```
Loss is divided by `accum_steps` **before** scaling/backward so that
summing gradients over `accum_steps` micro-batches produces the same
average gradient as one large batch would (standard grad-accumulation
normalization). Epoch accumulators are updated on-device as described
above.

**Optimizer step (lines 519-536)**:
```python
micro_step += 1
if micro_step % accum_steps == 0:
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad()
    global_step += 1
    ...
```
Gradient clipping happens on **unscaled** gradients (`scaler.unscale_`
must run before `clip_grad_norm_`, otherwise you'd be clipping to a norm
threshold that's off by the loss-scale factor). `scaler.step(optimizer)`
internally skips the actual `optimizer.step()` if any gradient was
inf/nan (AMP overflow) — in that case `global_step` and `scheduler.step()`
still advance regardless of whether the optimizer's `step()` was
effectively a no-op that iteration, since this code doesn't check
`scaler.get_scale()` before/after to detect a skipped step.

**Checkpoint + validation (lines 539-590)**, gated on
`global_step % CHECKPOINT_EVERY == 0 or global_step >= cfg["max_steps"]`
(the second condition guarantees a final checkpoint+validation exactly at
the last step even if `max_steps` isn't a multiple of `CHECKPOINT_EVERY`):
1. Runs `run_epoch(..., is_train=False)` on the full validation set.
2. Builds a `ckpt` dict with model/optimizer/scheduler/scaler state,
   current `val_loss`, and the **pre-update** `best_val_loss` (i.e. the
   best value from *before* this validation's comparison).
3. Saves it to `checkpoint_step_{global_step}.pt`, then deletes the
   previously saved periodic checkpoint (`last_ckpt_path`) — so at most
   one periodic checkpoint file exists on disk at a time (plus
   `best_model.pt`).
4. If this validation's loss beats `best_val_loss`, updates
   `best_val_loss`, patches `ckpt["best_val_loss"]` to the new value, and
   separately saves a trimmed subset of fields (no optimizer/scheduler/
   scaler state — smaller file, inference-only) to `best_model.pt`.
   Note: the periodic `checkpoint_step_N.pt` saved in step 3 was already
   written to disk *before* this update, so on a step where a new best is
   achieved, that periodic checkpoint's embedded `best_val_loss` field is
   one checkpoint-interval stale relative to its own `val_loss` field
   (informational only — resuming from it just starts the "is this a new
   best" comparison from a slightly stale threshold; `best_model.pt`
   itself is always correct).
5. Appends this step's validation metrics to `run_meta["checkpoints"]`,
   updates `run_meta["best_val_loss"]`, flushes `metrics.json`, and calls
   `gc.collect()`.

### 2.10 CLI (`parse_args`)

All `TRAINING_DEFAULTS` values are exposed as overridable flags
(`--max_steps`, `--batch_size`, `--lr`, `--max_seq_length`,
`--max_train_samples`, `--max_val_samples`, `--grad_accum_steps`).
`--output_dir` and `--data_dir` control where checkpoints/metrics and the
downloaded dataset cache live, respectively. `--num_workers` controls
`DataLoader` parallelism (0 = tokenize on the main process, blocking the
GPU between batches). `--custom_attention` and the resume flags
(`--resume_from_checkpoint` / `--auto_resume`, mutually exclusive) are
described in §2.3 and §2.8.
