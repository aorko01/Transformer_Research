"""
train.py  –  BERT-Small MLM on English Wikipedia
=================================================
Paper-accurate BERT-Small (Turc et al., 2019 / Well-Read Students…):
  hidden_size=512, num_hidden_layers=4, num_attention_heads=8, intermediate_size=2048

Features
--------
* Per-layer CPU-RAM + VRAM delta profiling (forward hooks)
* Time-per-epoch, Cross-Entropy Loss, Masked Token Accuracy
* All metrics → metrics.json
* Checkpoint every epoch + best-model checkpoint
* Swappable attention:  --custom_attention path.to.MyClass
* Resumes via resume_training.py
"""

import argparse, json, os, time, math, gc, sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple

import psutil
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
    hidden_size             = 512,
    num_hidden_layers       = 4,
    num_attention_heads     = 8,
    intermediate_size       = 2048,
    hidden_act              = "gelu",
    hidden_dropout_prob     = 0.1,
    attention_probs_dropout_prob = 0.1,
    max_position_embeddings = 512,
    type_vocab_size         = 2,
    initializer_range       = 0.02,
    layer_norm_eps          = 1e-12,
    vocab_size              = 30522,   # bert-base-uncased vocab
)

TRAINING_DEFAULTS = dict(
    mlm_probability     = 0.15,
    learning_rate       = 1e-4,
    weight_decay        = 0.01,
    adam_beta1          = 0.9,
    adam_beta2          = 0.999,
    adam_epsilon        = 1e-6,
    max_grad_norm       = 1.0,
    warmup_steps        = 10_000,
    batch_size          = 32,
    max_seq_length      = 128,
    num_epochs          = 3,
    max_train_samples   = 50_000,
    max_val_samples     = 5_000,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Memory profiling
# ─────────────────────────────────────────────────────────────────────────────

def _ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

def _vram_mb() -> float:
    return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0


class LayerMemoryHook:
    """Forward hooks that record RAM/VRAM delta per named submodule."""

    def __init__(self):
        self.stats: Dict[str, Dict]      = {}
        self._pending: Dict[str, Tuple]  = {}
        self._handles                    = []

    def attach(self, model: nn.Module):
        for name, module in model.named_modules():
            if not name:
                continue
            def _pre(mod, inp, _tag=name):
                self._pending[_tag] = (_ram_mb(), _vram_mb())
            def _post(mod, inp, out, _tag=name):
                if _tag not in self._pending:
                    return
                r0, v0 = self._pending.pop(_tag)
                entry = self.stats.setdefault(_tag, {"ram_sum":0.,"vram_sum":0.,"n":0})
                entry["ram_sum"]  += _ram_mb() - r0
                entry["vram_sum"] += _vram_mb() - v0
                entry["n"]        += 1
            self._handles += [
                module.register_forward_pre_hook(_pre),
                module.register_forward_hook(_post),
            ]

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def averages(self) -> Dict:
        return {
            name: {
                "avg_ram_delta_mb":  round(s["ram_sum"]  / max(s["n"],1), 4),
                "avg_vram_delta_mb": round(s["vram_sum"] / max(s["n"],1), 4),
                "total_calls":       s["n"],
            }
            for name, s in self.stats.items()
        }

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
            module.self = cls(config)
            n += 1
    print(f"[swap_attention] Replaced {n} layers → {cls.__name__}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Training / validation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, scheduler, collator,
              device, cfg, is_train: bool,
              hook: Optional[LayerMemoryHook] = None,
              profile_batches: int = 10) -> Dict:
    model.train(is_train)
    total_loss, total_correct, total_masked, n_batches = 0., 0, 0, 0
    t0 = time.perf_counter()

    with torch.set_grad_enabled(is_train):
        for bidx, batch in enumerate(loader):
            # Reconstruct list-of-dicts for collator
            keys = list(batch.keys())
            items = [{k: batch[k][i] for k in keys} for i in range(len(batch[keys[0]]))]
            masked = collator(items)

            input_ids  = masked["input_ids"].to(device)
            attn_mask  = masked.get("attention_mask")
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

    elapsed = time.perf_counter() - t0
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BERT-Small MLM Training")
    print(f"  Device  : {device}")
    print(f"  Started : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}\n")

    cfg = {**TRAINING_DEFAULTS,
           "num_epochs":       args.epochs,
           "batch_size":       args.batch_size,
           "learning_rate":    args.lr,
           "max_seq_length":   args.max_seq_length,
           "max_train_samples":args.max_train_samples,
           "max_val_samples":  args.max_val_samples}

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
        mod = importlib.import_module(mod_str.split("/")[-1].replace(".py",""))
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
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"]*2,
                              shuffle=False, num_workers=0)

    # ── Optimizer & scheduler ────────────────────────────────────────
    no_decay = {"bias", "LayerNorm.weight"}
    optimizer = AdamW([
        {"params": [p for n,p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg["weight_decay"]},
        {"params": [p for n,p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ], lr=cfg["learning_rate"],
       betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
       eps=cfg["adam_epsilon"])

    total_steps  = len(train_loader) * cfg["num_epochs"]
    warmup_steps = min(cfg["warmup_steps"], total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    print(f"  Train batches/epoch : {len(train_loader)}")
    print(f"  Val   batches       : {len(val_loader)}")
    print(f"  Total steps         : {total_steps}")
    print(f"  Warmup steps        : {warmup_steps}\n")

    # ── Hook ─────────────────────────────────────────────────────────
    hook = LayerMemoryHook()
    hook.attach(model)

    run_meta = {
        "model":            "bert-small",
        "params_M":         round(n_params, 2),
        "device":           str(device),
        "bert_config":      BERT_SMALL_CONFIG,
        "training_cfg":     cfg,
        "custom_attention": args.custom_attention or "none",
        "started_at":       datetime.now().isoformat(),
        "epochs":           [],
        "layer_memory_profile": {},
    }

    best_val_loss = float("inf")

    # ── Epoch loop ───────────────────────────────────────────────────
    for epoch in range(1, cfg["num_epochs"] + 1):
        print(f"\n{'─'*50}  Epoch {epoch}/{cfg['num_epochs']}  {'─'*50}")
        hook.reset()

        train_m = run_epoch(model, train_loader, optimizer, scheduler,
                            collator, device, cfg, is_train=True,
                            hook=hook, profile_batches=args.profile_batches)
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

        # ── Layer memory table ───────────────────────────────────────
        print(f"\n  ── Top-20 layers by |avg_ram_delta_mb| ──")
        print(f"  {'Layer':<60}  {'RAM Δ (MB)':>12}  {'VRAM Δ (MB)':>12}  {'Calls':>6}")
        print(f"  {'─'*60}  {'─'*12}  {'─'*12}  {'─'*6}")
        top = sorted(layer_profile.items(),
                     key=lambda x: abs(x[1]["avg_ram_delta_mb"]), reverse=True)[:20]
        for lname, s in top:
            print(f"  {lname:<60}  {s['avg_ram_delta_mb']:>+12.4f}  "
                  f"{s['avg_vram_delta_mb']:>+12.4f}  {s['total_calls']:>6}")

        # ── Record ──────────────────────────────────────────────────
        run_meta["epochs"].append({"epoch": epoch,
                                   "train": train_m,
                                   "val":   val_m})
        run_meta["layer_memory_profile"][f"epoch_{epoch}"] = layer_profile

        # ── Checkpoint ──────────────────────────────────────────────
        ckpt = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss":             val_m["loss"],
            "bert_config":          BERT_SMALL_CONFIG,
            "training_cfg":         cfg,
            "custom_attention":     args.custom_attention or "none",
        }
        ckpt_path = out_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(ckpt, ckpt_path)
        print(f"\n  Checkpoint → {ckpt_path}")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save({k: ckpt[k] for k in
                        ("epoch","model_state_dict","bert_config",
                         "training_cfg","val_loss","custom_attention")},
                       out_dir / "best_model.pt")
            print(f"  ✓ Best val loss {best_val_loss:.4f} → best_model.pt")

        gc.collect()

    hook.detach()

    # ── Save metrics ─────────────────────────────────────────────────
    run_meta["finished_at"]  = datetime.now().isoformat()
    run_meta["best_val_loss"] = best_val_loss

    mp = out_dir / "metrics.json"
    with open(mp, "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done.  Best val loss: {best_val_loss:.4f}")
    print(f"  Metrics → {mp}")
    print(f"{'='*60}\n")
    return run_meta


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BERT-Small MLM training")
    p.add_argument("--output_dir",          default="./bert_small_output")
    p.add_argument("--epochs",    type=int, default=TRAINING_DEFAULTS["num_epochs"])
    p.add_argument("--batch_size",type=int, default=TRAINING_DEFAULTS["batch_size"])
    p.add_argument("--lr",        type=float, default=TRAINING_DEFAULTS["learning_rate"])
    p.add_argument("--max_seq_length",     type=int, default=TRAINING_DEFAULTS["max_seq_length"])
    p.add_argument("--max_train_samples",  type=int, default=TRAINING_DEFAULTS["max_train_samples"])
    p.add_argument("--max_val_samples",    type=int, default=TRAINING_DEFAULTS["max_val_samples"])
    p.add_argument("--profile_batches",    type=int, default=10)
    p.add_argument("--custom_attention",   type=str, default=None,
                   help="Module.ClassName of custom attention, e.g. custom_attention_template.LinearAttention")
    return p.parse_args()

if __name__ == "__main__":
    train(parse_args())
