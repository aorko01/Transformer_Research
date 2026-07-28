# GPT CLM training on TinyStories (single GPU)

Autoregressive next-token-prediction (causal language modelling) training of a
GPT-style model on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
dataset.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Prepare the data

Downloads TinyStories from the HuggingFace Hub, tokenizes it with the GPT-2 BPE
tokenizer (`tiktoken`), and writes `train.bin` / `val.bin` token shards:

```bash
python data/tinystories/prepare.py
```

## 2. Train

```bash
python train.py
```

Useful overrides:

```bash
python train.py --batch_size 8 --seq_len 512 --max_steps 20000 --lr 6e-4
```

Run `python train.py --help` for the full list of CLI options (data dir,
checkpoint dir, seed, `--no-compile`, etc). Config defaults live in
`GPT_Config.py` (`ModelConfig` for the architecture, `TrainingConfig` for
optimization/eval settings).

## What changed from the original scaffold

- `Dataloader.py` no longer hardcodes a local Shakespeare `.txt` file. It now
  streams from pre-tokenized `train.bin` / `val.bin` shards via `np.memmap`
  (see `data/tinystories/prepare.py`), and supports both sequential batches
  (`next_batch`, used for training) and randomly sampled batches
  (`random_batch`, used for validation).
- `train.py`:
  - Autocast dtype and `torch.cuda.synchronize()` are now guarded by an actual
    CUDA check instead of being hardcoded to `"cuda"`, so the script no longer
    crashes on CPU-only machines.
  - Optimizer is built before `torch.compile` wraps the model, to avoid
    compiled parameter name prefixes leaking into checkpoints.
  - Added a validation loop (`eval_interval` / `eval_iters`) with
    best-checkpoint saving to `out/ckpt.pt`.
  - Added linear warmup + cosine LR decay (`get_lr` in `training_utils.py`)
    instead of driving `CosineAnnealingLR` off a fixed, disconnected
    `T_max=50000`.
  - Added `argparse` CLI overrides and a fixed random seed for reproducibility.
- `training_utils.py`:
  - Fixed the `configure_optimizers` signature - the first argument was
    named `self` despite being a plain function (it worked, but was
    misleading); it's now named `model`.
  - Fixed the misspelled `claculate_loss` -> `calculate_loss`, with a
    backwards-compatible alias kept for anything still importing the old name.
  - Added `get_lr` (warmup + cosine schedule) and `estimate_loss`
    (validation-loss averaging helper).
- `GPT_Config.py`: added `TrainingConfig` fields needed by the above
  (`data_dir`, `out_dir`, `eval_interval`, `eval_iters`, `warmup_steps`,
  `min_lr`, `seed`).
