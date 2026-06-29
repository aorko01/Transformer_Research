"""
train.py  –  BERT-Small MLM on English Wikipedia
=================================================
Paper-accurate BERT-Small (Turc et al., 2019 / Well-Read Students…):
  hidden_size=512, num_hidden_layers=4, num_attention_heads=8, intermediate_size=2048

Features
--------
* Per-layer peak VRAM + forward-pass time profiling (forward hooks)
* Time-per-epoch, Cross-Entropy Loss, Masked Token Accuracy
* All metrics → metrics.json  (epoch-averaged layer profiles only)
* Checkpoint every epoch + best-model checkpoint
* Swappable attention:  --custom_attention path.to.MyClass
* Resumes automatically if --resume_from_checkpoint <path> is given,
  or auto-detects the latest checkpoint in --output_dir
"""

import argparse, json, os, time, math, gc, sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BertConfig, BertForMaskedLM, BertTokenizerFast,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from data_loader import load_wikipedia_dataset


# ─────────────────────────────────────────────────────────────────────────────
#  Paper-accurate BERT-Small hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

BERT_SMALL_CONFIG = dict(
    hidden_size                  = 512,
    num_hidden_layers            = 4,
    num_attention_heads          = 8,
    intermediate_size            = 2048,
    hidden_act                   = "gelu",
    hidden_dropout_prob          = 0.1,
    attention_probs_dropout_prob = 0.1,
    max_position_embeddings      = 512,
    type_vocab_size              = 2,
    initializer_range            = 0.02,
    layer_norm_eps               = 1e-12,
    vocab_size                   = 30522,   # bert-base-uncased vocab
)

TRAINING_DEFAULTS = dict(
    mlm_probability      = 0.15,
    learning_rate        = 1e-4,
    weight_decay         = 0.01,
    adam_beta1           = 0.9,
    adam_beta2           = 0.999,
    adam_epsilon         = 1e-6,
    max_grad_norm        = 1.0,
    warmup_steps         = 10_000,
    batch_size           = 32,
    max_seq_length       = 128,
    num_epochs           = 3,
    max_train_samples    = 50_000,
    max_val_samples      = 5_000,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Layer profiling: peak VRAM + forward-pass time
# ─────────────────────────────────────────────────────────────────────────────

class LayerProfileHook:
    """
    Forward hooks that record, per named submodule:
      • peak VRAM (MB) during the forward pass through that layer
      • wall-clock time (ms) for the forward pass

    Only tracks GPU memory; no CPU/RAM tracking.
    Accumulates across calls so epoch averages can be computed at the end.
    """

    def __init__(self):
        self.stats: Dict[str, Dict] = {}
        self._pending: Dict[str, float] = {}   # name → pre-call timestamp
        self._handles = []

    def attach(self, model: nn.Module):
        for name, module in model.named_modules():
            if not name:
                continue

            def _pre(mod, inp, _tag=name):
                if torch.cuda.is_available():
                    # Reset peak tracker so we measure only this layer's peak
                    torch.cuda.reset_peak_memory_stats()
                self._pending[_tag] = time.perf_counter()

            def _post(mod, inp, out, _tag=name):
                t_end = time.perf_counter()
                if _tag not in self._pending:
                    return

                elapsed_ms = (t_end - self._pending.pop(_tag)) * 1000.0

                peak_vram_mb = (
                    torch.cuda.max_memory_allocated() / 1024 ** 2
                    if torch.cuda.is_available()
                    else 0.0
                )

                entry = self.stats.setdefault(
                    _tag,
                    {"peak_vram_sum": 0.0, "time_ms_sum": 0.0, "n": 0},
                )
                entry["peak_vram_sum"] += peak_vram_mb
                entry["time_ms_sum"]   += elapsed_ms
                entry["n"]             += 1

            self._handles += [
                module.register_forward_pre_hook(_pre),
                module.register_forward_hook(_post),
            ]

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def averages(self) -> Dict:
    """Return per-layer averages over all calls seen since last reset()."""

        result = {}

        for name, s in self.stats.items():

            avg_peak_vram = round(
                s["peak_vram_sum"] / max(s["n"], 1),
                4
            )

            avg_forward = round(
                s["time_ms_sum"] / max(s["n"], 1),
                4
            )

            result[name] = {
                "avg_peak_vram_mb": avg_peak_vram,
                "avg_forward_ms": avg_forward,
                "total_calls": s["n"],
            }

        return result

    def reset(self):
        self.stats.clear()
        self._pending.clear()


# ─────────────────────────────────────────────────────────────────────────────
#  Swappable attention
# ─────────────────────────────────────────────────────────────────────────────

def swap_attention_layers(model: BertForMaskedLM, cls, config: BertConfig):
    """Replace every BertSelfAttention with cls(config)."""
    from transformers.models.bert.modeling_bert import BertAttention
    n = 0
    for _, module in model.named_modules():
        if isinstance(module, BertAttention):
            #create a new module that is defined by me 
            module.self = cls(config)
            n += 1
    print(f"[swap_attention] Replaced {n} layers → {cls.__name__}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(out_dir: Path) -> Optional[Path]:
    """Return the checkpoint with the highest epoch number in out_dir, or None."""
    ckpts = sorted(out_dir.glob("checkpoint_epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def load_checkpoint(path: Path, model, optimizer, scheduler, device):
    """
    Load a checkpoint saved by this script.
    Returns the epoch that was completed (so training resumes from epoch+1).
    """
    print(f"\n  Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch_done = ckpt["epoch"]
    best_val   = ckpt.get("best_val_loss", float("inf"))
    print(f"  Resumed after epoch {epoch_done}  (best val loss so far: {best_val:.4f})\n")
    return epoch_done, best_val


def load_existing_metrics(metrics_path: Path) -> dict:
    """Load existing metrics.json if it exists (for resume), else return empty shell."""
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
#  Training / validation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, scheduler, collator,
              device, cfg, is_train: bool) -> Dict:
    model.train(is_train)
    total_loss, total_correct, total_masked, n_batches = 0., 0, 0, 0
    t0 = time.perf_counter()

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            keys  = list(batch.keys())
            items = []
            for i in range(len(batch[keys[0]])):
                sample = {}

                for k in keys:
                    sample[k] = batch[k][i]

                items.append(sample)
            masked = collator(items)

            input_ids = masked["input_ids"].to(device)
            attn_mask = masked.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)
            labels = masked["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attn_mask,
                            labels=labels)
            loss   = outputs.loss
            logits = outputs.logits

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            n_batches  += 1

            mask = labels != -100
            if mask.any():
                total_correct += (logits.argmax(-1)[mask] == labels[mask]).sum().item()
                total_masked  += mask.sum().item()

    elapsed  = time.perf_counter() - t0
    avg_loss = total_loss / max(n_batches, 1)
    return {
        "loss":                  round(avg_loss, 6),
        "masked_token_accuracy": round(total_correct / max(total_masked, 1), 6),
        "perplexity":            round(math.exp(min(avg_loss, 20)), 4),
        "epoch_time_sec":        round(elapsed, 2),
        "batches":               n_batches,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"

    print(f"\n{'='*60}")
    print(f"  BERT-Small MLM Training")
    print(f"  Device  : {device}")
    print(f"  Started : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}\n")

    cfg = {
        **TRAINING_DEFAULTS,
        "num_epochs":        args.epochs,
        "batch_size":        args.batch_size,
        "learning_rate":     args.lr,
        "max_seq_length":    args.max_seq_length,
        "max_train_samples": args.max_train_samples,
        "max_val_samples":   args.max_val_samples,
    }

    # ── Tokenizer & model ────────────────────────────────────────────
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bert_cfg  = BertConfig(**BERT_SMALL_CONFIG)
    model     = BertForMaskedLM(bert_cfg)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model params : {n_params:.1f}M")

    if args.custom_attention:
        mod_str, cls_name = args.custom_attention.rsplit(".", 1)
        sys.path.insert(0, str(Path(mod_str).parent) if "/" in mod_str else ".")
        import importlib
        mod   = importlib.import_module(mod_str.split("/")[-1].replace(".py", ""))
        model = swap_attention_layers(model, getattr(mod, cls_name), bert_cfg)

    model.to(device)

    # ── Data ─────────────────────────────────────────────────────────
    train_ds, val_ds = load_wikipedia_dataset(
        tokenizer, cfg["max_seq_length"],
        cfg["max_train_samples"], cfg["max_val_samples"])

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True,
        mlm_probability=cfg["mlm_probability"],
        return_tensors="pt")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"] * 2,
                              shuffle=False, num_workers=0)

    # ── Optimizer & scheduler ────────────────────────────────────────
    no_decay  = {"bias", "LayerNorm.weight"}
    optimizer = AdamW(
        [
            {"params": [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": cfg["weight_decay"]},
            {"params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ],
        lr=cfg["learning_rate"],
        betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
        eps=cfg["adam_epsilon"],
    )

    total_steps  = len(train_loader) * cfg["num_epochs"]
    warmup_steps = min(cfg["warmup_steps"], total_steps // 10)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    # ── Resume from checkpoint ───────────────────────────────────────
    start_epoch   = 0          # epochs already completed
    best_val_loss = float("inf")

    resume_path = args.resume_from_checkpoint
    if resume_path is None and args.auto_resume:
        latest = find_latest_checkpoint(out_dir)
        if latest:
            resume_path = str(latest)

    if resume_path:
        start_epoch, best_val_loss = load_checkpoint(
            Path(resume_path), model, optimizer, scheduler, device)

    print(f"  Train batches/epoch : {len(train_loader)}")
    print(f"  Val   batches       : {len(val_loader)}")
    print(f"  Total steps         : {total_steps}")
    print(f"  Warmup steps        : {warmup_steps}")
    if start_epoch:
        print(f"  Skipping epochs 1–{start_epoch} (already done)")
    print()

    # ── Load or initialise the metrics document ──────────────────────
    run_meta = load_existing_metrics(metrics_path)
    if not run_meta:
        run_meta = {
            "model":               "bert-small",
            "params_M":            round(n_params, 2),
            "device":              str(device),
            "bert_config":         BERT_SMALL_CONFIG,
            "training_cfg":        cfg,
            "custom_attention":    args.custom_attention or "none",
            "started_at":          datetime.now().isoformat(),
            "epochs":              [],
            # layer_memory_profile is populated incrementally below
        }

    # ── Profiling hook ───────────────────────────────────────────────
    hook = LayerProfileHook()
    hook.attach(model)

    # ── Epoch loop ───────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, cfg["num_epochs"] + 1):
        print(f"\n{'─'*50}  Epoch {epoch}/{cfg['num_epochs']}  {'─'*50}")
        hook.reset()

        train_m = run_epoch(model, train_loader, optimizer, scheduler,
                            collator, device, cfg, is_train=True)

        # Capture layer profile from the training pass only
        layer_profile = hook.averages()

        val_m = run_epoch(model, val_loader, optimizer, scheduler,
                          collator, device, cfg, is_train=False)

        print(f"\n  [Train]  loss={train_m['loss']:.4f}  "
              f"acc={train_m['masked_token_accuracy']:.4f}  "
              f"ppl={train_m['perplexity']:.2f}  "
              f"time={train_m['epoch_time_sec']:.1f}s")
        print(f"  [Val  ]  loss={val_m['loss']:.4f}  "
              f"acc={val_m['masked_token_accuracy']:.4f}  "
              f"ppl={val_m['perplexity']:.2f}  "
              f"time={val_m['epoch_time_sec']:.1f}s")

        # ── Update metrics (epoch averages only; no per-epoch table) ─
        run_meta.setdefault("epochs", []).append({
            "epoch": epoch,
            "train": train_m,
            "val":   val_m,
        })

        # Store per-epoch layer profile in metrics.json;
        # also maintain a rolling average across all completed epochs.
        run_meta.setdefault("layer_profile_per_epoch", {})[f"epoch_{epoch}"] = layer_profile
        _update_overall_layer_avg(run_meta, layer_profile, epoch)

        # ── Checkpoint (includes best_val_loss for safe resume) ──────
        ckpt = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss":             val_m["loss"],
            "best_val_loss":        best_val_loss,   # carried forward
            "bert_config":          BERT_SMALL_CONFIG,
            "training_cfg":         cfg,
            "custom_attention":     args.custom_attention or "none",
        }
        ckpt_path = out_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(ckpt, ckpt_path)
        print(f"\n  Checkpoint → {ckpt_path}")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            ckpt["best_val_loss"] = best_val_loss
            torch.save(
                {k: ckpt[k] for k in (
                    "epoch", "model_state_dict", "bert_config",
                    "training_cfg", "val_loss", "best_val_loss",
                    "custom_attention")},
                out_dir / "best_model.pt",
            )
            print(f"  ✓ Best val loss {best_val_loss:.4f} → best_model.pt")

        # Flush metrics after every epoch so a crash loses nothing
        run_meta["best_val_loss"] = best_val_loss
        _save_metrics(run_meta, metrics_path)

        gc.collect()

    hook.detach()

    # ── Final metrics flush ──────────────────────────────────────────
    run_meta["finished_at"] = datetime.now().isoformat()
    _save_metrics(run_meta, metrics_path)

    print(f"\n{'='*60}")
    print(f"  Done.  Best val loss: {best_val_loss:.4f}")
    print(f"  Metrics → {metrics_path}")
    print(f"{'='*60}\n")
    return run_meta


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _update_overall_layer_avg(run_meta: dict, layer_profile: dict, epoch: int):
    """
    Maintain a cumulative average of peak_vram and forward_ms across epochs
    and store it under run_meta["layer_profile_overall_avg"].
    Uses an incremental mean formula so we never store raw sums in the JSON.
    """
    overall = run_meta.setdefault("layer_profile_overall_avg", {})
    n = epoch   # number of epochs averaged so far (1-indexed, all included)

    for name, stats in layer_profile.items():
        if name not in overall:
            overall[name] = {
                "avg_peak_vram_mb": stats["avg_peak_vram_mb"],
                "avg_forward_ms":   stats["avg_forward_ms"],
            }
        else:
            prev = overall[name]
            # Welford-style incremental mean
            prev["avg_peak_vram_mb"] = round(
                prev["avg_peak_vram_mb"] + (stats["avg_peak_vram_mb"] - prev["avg_peak_vram_mb"]) / n, 4)
            prev["avg_forward_ms"]   = round(
                prev["avg_forward_ms"]   + (stats["avg_forward_ms"]   - prev["avg_forward_ms"])   / n, 4)


def _save_metrics(run_meta: dict, path: Path):
    with open(path, "w") as f:
        json.dump(run_meta, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BERT-Small MLM training")
    p.add_argument("--output_dir",           default="./bert_small_output")
    p.add_argument("--epochs",     type=int, default=TRAINING_DEFAULTS["num_epochs"])
    p.add_argument("--batch_size", type=int, default=TRAINING_DEFAULTS["batch_size"])
    p.add_argument("--lr",         type=float, default=TRAINING_DEFAULTS["learning_rate"])
    p.add_argument("--max_seq_length",    type=int, default=TRAINING_DEFAULTS["max_seq_length"])
    p.add_argument("--max_train_samples", type=int, default=TRAINING_DEFAULTS["max_train_samples"])
    p.add_argument("--max_val_samples",   type=int, default=TRAINING_DEFAULTS["max_val_samples"])
    p.add_argument("--custom_attention",  type=str, default=None,
                   help="Module.ClassName of custom attention, "
                        "e.g. custom_attention_template.LinearAttention")

    # ── Resume options ───────────────────────────────────────────────
    resume = p.add_mutually_exclusive_group()
    resume.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a specific checkpoint_epoch_N.pt to resume from.")
    resume.add_argument("--auto_resume", action="store_true",
                        help="Automatically resume from the latest checkpoint in --output_dir.")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())