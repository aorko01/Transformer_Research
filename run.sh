#!/usr/bin/env bash
# run.sh — idempotent launcher: safe to invoke on first run, and safe to
# invoke again after a crash/reboot (picks up training from the last
# checkpoint automatically instead of starting over).
set -euo pipefail

# Always resolve paths relative to this script's own location, not the
# caller's cwd — required since a reboot auto-start mechanism (cron/launchd)
# won't necessarily invoke this from the repo directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="./run_vanilla"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/train.log"

# ── Environment setup (only on first run — a pre-existing .venv means deps
# are already installed, so reboot restarts skip straight to training) ──
if [ ! -d ".venv" ]; then
    echo "[run.sh] First run: creating virtualenv and installing dependencies..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi


# ── Train ────────────────────────────────────────────────────────────────
# --auto_resume: starts fresh if OUTPUT_DIR has no checkpoint yet (first
# run), or auto-detects and resumes from the latest checkpoint_step_*.pt
# otherwise (reboot/crash recovery) — same invocation covers both cases.
echo "[run.sh] $(date '+%Y-%m-%d %H:%M:%S')  Starting training (auto-resume) → $OUTPUT_DIR" | tee -a "$LOG_FILE"
python train.py \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 \
    --grad_accum_steps 16 \
    --auto_resume \
    >> "$LOG_FILE" 2>&1
