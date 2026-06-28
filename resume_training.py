"""
resume_training.py
==================
Resume a BERT-Small MLM training run from a saved checkpoint.

Usage:
    python resume_training.py \
        --checkpoint ./bert_small_output/checkpoint_epoch_2.pt \
        --output_dir ./bert_small_output \
        --epochs 5            # total epochs (including already-done ones)
        [--custom_attention custom_attention_template.MyCustomAttention]
"""

import argparse
import torch
from pathlib import Path
from train import (
    BERT_SMALL_CONFIG, TRAINING_DEFAULTS,
    BertConfig, BertForMaskedLM, BertTokenizerFast,
    DataCollatorForLanguageModeling, AdamW,
    get_linear_schedule_with_warmup,
    load_wikipedia_dataset, run_epoch, LayerMemoryHook,
    swap_attention_layers,
)
import json
import gc
import math
from datetime import datetime


def resume(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nResuming from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    start_epoch  = ckpt["epoch"] + 1
    bert_cfg_d   = ckpt.get("bert_config", BERT_SMALL_CONFIG)
    training_cfg = ckpt.get("training_cfg", TRAINING_DEFAULTS)

    print(f"  Resuming at epoch {start_epoch}  (trained through {ckpt['epoch']})")

    # ── Rebuild model ────────────────────────────────────────────────
    bert_cfg = BertConfig(**bert_cfg_d)
    model    = BertForMaskedLM(bert_cfg)

    if args.custom_attention:
        import importlib, sys
        from pathlib import Path as P
        mod_path, cls_name = args.custom_attention.rsplit(".", 1)
        sys.path.insert(0, str(P(mod_path).parent))
        mod = importlib.import_module(P(mod_path).stem)
        custom_cls = getattr(mod, cls_name)
        model = swap_attention_layers(model, custom_cls, bert_cfg)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # ── Rebuild data ─────────────────────────────────────────────────
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_ds, val_ds = load_wikipedia_dataset(
        tokenizer,
        max_seq_length=training_cfg["max_seq_length"],
        max_train=training_cfg["max_train_samples"],
        max_val=training_cfg["max_val_samples"],
    )
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=training_cfg["batch_size"],
                              shuffle=True, num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=training_cfg["batch_size"] * 2,
                              shuffle=False, num_workers=2)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                               mlm_probability=training_cfg["mlm_probability"],
                                               return_tensors="pt")

    # ── Rebuild optimizer/scheduler ──────────────────────────────────
    no_decay = {"bias", "LayerNorm.weight"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": training_cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, lr=training_cfg["learning_rate"],
                      betas=(training_cfg["adam_beta1"], training_cfg["adam_beta2"]),
                      eps=training_cfg["adam_epsilon"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = min(training_cfg["warmup_steps"], total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # ── Load existing metrics ─────────────────────────────────────────
    out_dir      = Path(args.output_dir)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            run_meta = json.load(f)
    else:
        run_meta = {"epochs": [], "layer_memory_profile": {}}

    best_val_loss = ckpt.get("val_loss", float("inf"))
    hook = LayerMemoryHook()
    hook.attach(model)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        hook.reset()

        train_m = run_epoch(model, train_loader, optimizer, scheduler,
                            collator, device, training_cfg, is_train=True,
                            hook=hook, profile_batches=10)
        layer_profile = hook.averages()
        val_m   = run_epoch(model, val_loader, optimizer, scheduler,
                            collator, device, training_cfg, is_train=False)

        print(f"  [Train] {train_m}")
        print(f"  [Val  ] {val_m}")

        run_meta["epochs"].append({"epoch": epoch, "train": train_m, "val": val_m})
        run_meta["layer_memory_profile"][f"epoch_{epoch}"] = layer_profile

        ckpt_path = out_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_m["loss"],
            "bert_config": bert_cfg_d,
            "training_cfg": training_cfg,
        }, ckpt_path)

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "bert_config": bert_cfg_d, "val_loss": best_val_loss},
                       out_dir / "best_model.pt")
            print(f"  ✓ New best {best_val_loss:.4f}")

        gc.collect()

    hook.detach()
    run_meta["finished_at"] = datetime.now().isoformat()
    run_meta["best_val_loss"] = best_val_loss
    with open(metrics_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"\nDone. Metrics → {metrics_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", default="./bert_small_output")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--custom_attention", default=None)
    resume(p.parse_args())
