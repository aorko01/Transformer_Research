"""
compare_runs.py
===============
Print a side-by-side table comparing metrics.json files from multiple runs.
Useful for comparing standard BERT-Small attention vs. your custom attention.

Usage:
    python compare_runs.py \
        bert_small_output/metrics.json \
        bert_custom_attn_output/metrics.json \
        [--labels vanilla custom]
"""

import argparse, json, sys
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt(v):
    if isinstance(v, float):
        return f"{v:.5f}"
    return str(v)


def compare(paths, labels):
    runs = [load(p) for p in paths]
    if not labels:
        labels = [Path(p).parent.name or Path(p).name for p in paths]

    # Epoch table
    max_epochs = max(len(r["epochs"]) for r in runs)

    header = f"{'Epoch':>5}  " + "  ".join(
        f"{'[' + lb + '] train_loss':>20}  {'val_loss':>10}  {'val_acc':>10}  {'time(s)':>8}"
        for lb in labels
    )
    print(header)
    print("─" * len(header))

    for ep in range(1, max_epochs + 1):
        row = f"{ep:>5}  "
        for run in runs:
            ep_data = next((e for e in run["epochs"] if e["epoch"] == ep), None)
            if ep_data:
                row += (
                    f"  {ep_data['train']['loss']:>20.5f}"
                    f"  {ep_data['val']['loss']:>10.5f}"
                    f"  {ep_data['val']['masked_token_accuracy']:>10.5f}"
                    f"  {ep_data['train']['epoch_time_sec']:>8.1f}"
                )
            else:
                row += "  " + " " * 52
        print(row)

    print()
    print("Best val loss per run:")
    for lb, run in zip(labels, runs):
        print(f"  {lb}: {run.get('best_val_loss', 'N/A')}")

    # Layer memory comparison (first epoch)
    print("\nLayer memory profile (epoch_1, top-10 by abs RAM delta, first run):")
    first_run = runs[0]
    ep1 = first_run.get("layer_memory_profile", {}).get("epoch_1", {})
    top = sorted(ep1.items(), key=lambda x: abs(x[1]["avg_ram_delta_mb"]), reverse=True)[:10]
    for name, stats in top:
        print(f"  {name:60s}  RAM Δ={stats['avg_ram_delta_mb']:+8.3f} MB  VRAM Δ={stats['avg_vram_delta_mb']:+8.3f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("metrics", nargs="+", help="paths to metrics.json files")
    p.add_argument("--labels", nargs="*", default=None)
    args = p.parse_args()
    compare(args.metrics, args.labels or [])
