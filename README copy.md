# BERT-Small Pretraining — Changes from Original Script

This document tracks the changes made to `train.py` and `data_loader.py`, in the order they were introduced, and why.

## Context

The original script trained a "BERT-Small" model that was actually configured with BERT-base dimensions (`hidden_size=768`, `num_hidden_layers=12`, etc.), trained epoch-by-epoch over a fixed 50k-sample slice of Wikipedia. The goal was to move it toward a step-driven, full-corpus BERT-style pretraining run, then adapt it to the actual hardware available (32GB GPU, 100,000-step target).

---

## 1. `TRAINING_DEFAULTS` — step-driven, not epoch-driven

**Before:**
```python
TRAINING_DEFAULTS = dict(
    ...
    max_seq_length    = 128,
    num_epochs        = 3,
    max_train_samples = 50_000,
    max_val_samples   = 5_000,
)
```

**After:**
```python
TRAINING_DEFAULTS = dict(
    ...
    batch_size                   = 32,        # micro-batch sized for 32GB VRAM + fp16
    gradient_accumulation_steps  = 8,          # 32 x 8 = effective batch 256
    max_seq_length                = 512,        # single phase (no 90/10 128/512 split)
    max_steps                    = 100_000,    # step-driven instead of epoch-driven
    max_train_samples            = None,       # None = stream full corpus
    max_val_samples               = 10_000,
    warmup_steps                 = 1_000,      # ~1% of 100k, proportional to paper's 10k/1M
)
```

Why: the original BERT paper trains by step count over the full corpus, not by epoch over a subsample. `num_epochs` was dropped as the primary control; `max_steps` now governs training length directly. `warmup_steps` was rescaled to keep the same ~1% proportion as the paper, adjusted for the 100k-step target instead of 1M.

## 2. Gradient accumulation + mixed precision (fp16/AMP)

Added because a 256 effective batch size at sequence length 512 does not fit in a single forward/backward pass on consumer/cloud GPUs.

- `gradient_accumulation_steps` accumulates gradients over N micro-batches before each optimizer step.
- `torch.cuda.amp.autocast` wraps the forward pass; `GradScaler` handles loss scaling, unscaling before gradient clipping, and scaled optimizer steps.
- `GradScaler` state is now saved/restored in checkpoints (`scaler_state_dict`) so resumed runs don't lose calibrated loss-scale values.

## 3. Step-driven training loop (replaces epoch-count loop)

**Before:** `for epoch in range(start_epoch + 1, cfg["num_epochs"] + 1):` — iterates a fixed number of epochs.

**After:** `while global_step < cfg["max_steps"]:` — runs full passes over the data until the target step count is reached. `run_epoch` now accepts and returns `global_step`, and breaks out of its batch loop early once `max_steps` is hit (so the final partial epoch doesn't run further than necessary).

Checkpoints and `find_latest_checkpoint` were renamed from `checkpoint_epoch_N.pt` to `checkpoint_step_N.pt` to reflect step-based progress, since epoch count is no longer the meaningful unit of progress at full-corpus scale.

## 4. Live training feedback: tqdm + ETA

- Added a `tqdm` progress bar around the batch loop in `run_epoch`, showing live loss, masked-token accuracy, and perplexity in the postfix, plus `step: global_step/max_steps` during training.
- Because the dataset is now a streaming `IterableDataset` with no `__len__`, tqdm cannot infer a total on its own. `total` is now passed explicitly as `(max_steps - global_step) * gradient_accumulation_steps` so tqdm can render a percentage bar and ETA during training. Validation (no `max_steps`) does not get an ETA unless `max_val_samples` is used to compute one.

## 5. `data_loader.py` — streaming instead of eager tokenization

**Before:** `load_wikipedia_dataset` called `load_dataset(...)` without `streaming=True`, sliced `dataset["text"][:total]`, and tokenized the entire train+val split into one in-memory `TokenDataset` up front.

**After:** `load_dataset(..., streaming=True)` returns an iterable stream. A new `StreamingTokenDataset(IterableDataset)` tokenizes each article lazily as it's pulled, rather than pre-encoding the full corpus into RAM. `dataset.take(max_val)` reserves the first N articles for validation, `dataset.skip(max_val)` provides the rest for training, optionally capped with `.take(max_train)` if `max_train_samples` is set, and `.shuffle(buffer_size=10_000, seed=42)` provides an approximate (buffer-based) shuffle suitable for streaming.

Why: the original eager approach would attempt to hold millions of tokenized articles in RAM simultaneously once `max_train_samples=None` (full corpus), which exceeds 32GB RAM on the target hardware. Streaming keeps memory usage bounded regardless of corpus size.

## 6. CLI changes

- `--epochs` removed; replaced with `--max_steps`.
- `--grad_accum_steps` added.
- `--resume_from_checkpoint` help text updated to reference `checkpoint_step_N.pt`.

## Known limitations / things not (yet) changed

- **No NSP**: the script trains `BertForMaskedLM` (MLM only). Original BERT pretraining also includes next-sentence prediction via `BertForPreTraining`. This was left as-is since it wasn't explicitly requested, but it's a real deviation from "full BERT-style pretraining" worth confirming intentional.
- **No two-phase sequence length schedule**: per explicit instruction, the 90% @ 128 / 10% @ 512 schedule from the paper was dropped in favor of a single `max_seq_length` throughout.
- **`len(train_loader)` no longer printed**: since the dataset is a streaming `IterableDataset`, there's no fixed "batches per epoch" to report at startup — only the live step counter during training.
- **Streaming shuffle is approximate**: the `buffer_size=10_000` shuffle is not equivalent to a full corpus shuffle, just a sliding-window approximation, which is standard practice for streaming pretraining but worth knowing about.
- **`num_attention_heads` currently `1`** in `BERT_SMALL_CONFIG` (uploaded file), not `8` as stated in the module docstring/paper-accurate spec — flagged but not changed, since it wasn't clear if this was intentional (e.g. testing single-head attention) or an accidental edit.
