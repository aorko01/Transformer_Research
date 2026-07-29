#!/bin/bash

# Script to run two sequential training runs with checkpoint resumption support
# Designed for systemd-based execution with graceful restart capability
# Location: /home/aorko/workplace/Transformer_Research/run.sh

# Do NOT exit on error immediately - handle it in each training function
set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# State file for tracking progress (survives script restarts)
STATE_FILE=".training_state"
LOCK_FILE=".training_lock"

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "ERROR: venv directory not found. Please set up virtual environment first."
    exit 1
fi
source venv/bin/activate

# Create output directories
mkdir -p runs/vanilla runs/custom logs

# Log file for this script
SCRIPT_LOG="logs/training_runs.log"

# Function to log messages
log_msg() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $1" | tee -a "$SCRIPT_LOG"
}

# Function to get current training state
get_state() {
    if [ -f "$STATE_FILE" ]; then
        cat "$STATE_FILE"
    else
        echo "VANILLA_PENDING"
    fi
}

# Function to update training state (atomic write)
set_state() {
    local new_state=$1
    # Write to temp file first, then move atomically
    echo "$new_state" > "$STATE_FILE.tmp"
    mv -f "$STATE_FILE.tmp" "$STATE_FILE"
    log_msg "State updated: $new_state"
}

# Function to acquire lock (prevents concurrent runs)
acquire_lock() {
    local timeout=30
    local elapsed=0
    while [ -f "$LOCK_FILE" ] && [ $elapsed -lt $timeout ]; do
        log_msg "Waiting for previous run to complete... ($elapsed/$timeout seconds)"
        sleep 2
        elapsed=$((elapsed + 2))
    done

    if [ -f "$LOCK_FILE" ]; then
        log_msg "ERROR: Could not acquire lock after ${timeout}s. Another instance may be running."
        return 1
    fi

    echo "$$" > "$LOCK_FILE"
    return 0
}

# Function to release lock
release_lock() {
    rm -f "$LOCK_FILE"
}

# Cleanup on exit (for systemd graceful shutdown)
cleanup() {
    local exit_code=$?
    log_msg "Script interrupted with exit code: $exit_code"
    release_lock
    exit $exit_code
}

trap cleanup SIGTERM SIGINT

# Function to run training with checkpoint resumption
run_training() {
    local attention=$1
    local out_dir=$2
    local metrics_path=$3
    local max_steps=2000
    local run_name="${attention^}"

    log_msg "=========================================="
    log_msg "Training Run: $run_name Attention"
    log_msg "=========================================="
    log_msg "Steps: $max_steps"
    log_msg "Attention: $attention"
    log_msg "Output Directory: $out_dir"
    log_msg "Metrics: $metrics_path"

    mkdir -p "$out_dir"

    # Check if training is already complete (idempotency check)
    if [ -f "$out_dir/.training_complete" ]; then
        log_msg "✓ $run_name attention training already completed in previous run"
        log_msg "  Checkpoint: $out_dir/ckpt_latest.pt"
        log_msg "  Metrics: $metrics_path"
        return 0
    fi

    # Check if checkpoint exists
    if [ -f "$out_dir/ckpt_latest.pt" ]; then
        log_msg "Found existing checkpoint at $out_dir/ckpt_latest.pt"
        log_msg "Resuming from checkpoint..."
        resume_flag="--resume $out_dir/ckpt_latest.pt"
    else
        log_msg "No checkpoint found, starting from scratch..."
        resume_flag=""
    fi

    # Run training with automatic retry logic
    max_attempts=3
    attempt=1

    while [ $attempt -le $max_attempts ]; do
        log_msg "Attempt $attempt of $max_attempts"

        python3 -u train.py \
            --max_steps $max_steps \
            --attention $attention \
            --out_dir "$out_dir" \
            --metrics_path "$metrics_path" \
            $resume_flag \
            >> "$out_dir/train.log" 2>&1

        exit_code=$?

        if [ $exit_code -eq 0 ]; then
            log_msg "✓ $run_name attention training completed successfully"
            log_msg "  Checkpoint: $out_dir/ckpt_latest.pt"
            log_msg "  Metrics: $metrics_path"

            # Mark training as complete (for idempotency on restart)
            touch "$out_dir/.training_complete"
            return 0
        else
            log_msg "✗ $run_name attention training failed with exit code $exit_code"

            if [ $attempt -lt $max_attempts ]; then
                log_msg "Attempt $attempt failed. Will retry from checkpoint on next attempt..."
                attempt=$((attempt + 1))
                resume_flag="--resume $out_dir/ckpt_latest.pt"
                # Wait before retrying (systemd-friendly)
                sleep 5
            else
                log_msg "✗ All $max_attempts attempts failed for $run_name attention training"
                log_msg "See $out_dir/train.log for details"
                log_msg "Next run will resume from the latest checkpoint at: $out_dir/ckpt_latest.pt"
                return 1
            fi
        fi
    done

    return 1
}

# Start training runs
log_msg "=========================================="
log_msg "Starting sequential training runs..."
log_msg "Working directory: $SCRIPT_DIR"
log_msg "Current state: $(get_state)"
log_msg "=========================================="

# Acquire lock to prevent concurrent runs
if ! acquire_lock; then
    log_msg "ERROR: Failed to acquire lock. Exiting."
    exit 1
fi

# Get current state and decide where to resume
CURRENT_STATE=$(get_state)

# Run 1: Vanilla Attention (if not yet completed)
if [ "$CURRENT_STATE" = "VANILLA_PENDING" ] || [ "$CURRENT_STATE" = "VANILLA_FAILED" ]; then
    log_msg ""
    log_msg "Starting Vanilla attention training..."
    log_msg ""

    if run_training "vanilla" "runs/vanilla" "runs/vanilla/metrics.json"; then
        set_state "CUSTOM_PENDING"
        log_msg ""
        log_msg "Vanilla attention training completed. Proceeding to custom attention training..."
        log_msg ""
    else
        set_state "VANILLA_FAILED"
        log_msg ""
        log_msg "Vanilla attention training failed after all retry attempts."
        log_msg "State saved as VANILLA_FAILED. Run script again to retry from checkpoint."
        log_msg "See logs/training_runs.log for details"
        release_lock
        exit 1
    fi
fi

# Run 2: Custom Attention (if not yet completed)
if [ "$CURRENT_STATE" = "CUSTOM_PENDING" ] || [ "$CURRENT_STATE" = "CUSTOM_FAILED" ]; then
    log_msg ""
    log_msg "Starting Custom attention training..."
    log_msg ""

    if run_training "custom" "runs/custom" "runs/custom/metrics.json"; then
        set_state "COMPLETED"
        log_msg ""
        log_msg "=========================================="
        log_msg "All training runs completed successfully!"
        log_msg "=========================================="
        log_msg ""
        log_msg "Summary:"
        log_msg "  Vanilla:  runs/vanilla/metrics.json"
        log_msg "  Custom:   runs/custom/metrics.json"
        log_msg ""
        log_msg "Logs:"
        log_msg "  Vanilla:  runs/vanilla/train.log"
        log_msg "  Custom:   runs/custom/train.log"
        log_msg "  Script:   logs/training_runs.log"
        log_msg ""
        log_msg "Checkpoints:"
        log_msg "  Vanilla:  runs/vanilla/ckpt_latest.pt (ckpt_best.pt)"
        log_msg "  Custom:   runs/custom/ckpt_latest.pt (ckpt_best.pt)"
        log_msg ""
        release_lock
        exit 0
    else
        set_state "CUSTOM_FAILED"
        log_msg ""
        log_msg "Custom attention training failed."
        log_msg "State saved as CUSTOM_FAILED. Run script again to retry from checkpoint."
        log_msg "See logs/training_runs.log and runs/custom/train.log for details"
        release_lock
        exit 1
    fi
elif [ "$CURRENT_STATE" = "COMPLETED" ]; then
    log_msg ""
    log_msg "=========================================="
    log_msg "Training already completed in previous run!"
    log_msg "=========================================="
    log_msg ""
    log_msg "Summary:"
    log_msg "  Vanilla:  runs/vanilla/metrics.json"
    log_msg "  Custom:   runs/custom/metrics.json"
    log_msg ""
    log_msg "To start fresh training, remove state file:"
    log_msg "  rm $STATE_FILE"
    log_msg ""
    release_lock
    exit 0
fi

log_msg "ERROR: Unknown state: $CURRENT_STATE"
release_lock
exit 1