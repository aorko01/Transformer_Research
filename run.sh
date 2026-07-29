#!/bin/bash

# Script to run two sequential training runs with checkpoint resumption support
# Location: /home/aorko/workplace/Transformer_Research/run.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
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
            --out_dir $out_dir \
            --metrics_path $metrics_path \
            $resume_flag \
            >> "$out_dir/train.log" 2>&1
        
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            log_msg "✓ $run_name attention training completed successfully"
            log_msg "  Checkpoint: $out_dir/ckpt_latest.pt"
            log_msg "  Metrics: $metrics_path"
            return 0
        else
            log_msg "✗ $run_name attention training failed with exit code $exit_code"
            
            if [ $attempt -lt $max_attempts ]; then
                log_msg "Attempt $attempt failed. Will retry from checkpoint on next attempt..."
                attempt=$((attempt + 1))
                resume_flag="--resume $out_dir/ckpt_latest.pt"
                # Wait before retrying
                sleep 5
            else
                log_msg "✗ All $max_attempts attempts failed for $run_name attention training"
                log_msg "See $out_dir/train.log for details"
                return 1
            fi
        fi
    done
    
    return 1
}

# Start training runs
log_msg "Starting sequential training runs..."
log_msg "Working directory: $SCRIPT_DIR"

# Run 1: Vanilla Attention
if run_training "vanilla" "runs/vanilla" "runs/vanilla/metrics.json"; then
    log_msg ""
    log_msg "Vanilla attention training completed. Proceeding to custom attention training..."
    log_msg ""
else
    log_msg ""
    log_msg "Vanilla attention training failed after all retry attempts."
    log_msg "Exiting without starting custom attention training."
    log_msg "See logs/training_runs.log for details"
    exit 1
fi

# Run 2: Custom Attention
if run_training "custom" "runs/custom" "runs/custom/metrics.json"; then
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
    exit 0
else
    log_msg ""
    log_msg "Custom attention training failed."
    log_msg "See logs/training_runs.log and runs/custom/train.log for details"
    exit 1
fi