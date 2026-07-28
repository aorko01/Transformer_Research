import argparse
import os
import time

import torch

from GPT_Config import ModelConfig, TrainingConfig
from Dataloader import Dataloader
from model.GPT import GPT
from training_utils import calculate_loss, configure_optimizers, estimate_loss, get_lr


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT model for causal language modelling on TinyStories")
    parser.add_argument("--data_dir", type=str, default=TrainingConfig.data_dir,
                         help="directory containing train.bin / val.bin (see data/tinystories/prepare.py)")
    parser.add_argument("--out_dir", type=str, default=TrainingConfig.out_dir,
                         help="directory to write checkpoints to")
    parser.add_argument("--max_steps", type=int, default=TrainingConfig.max_steps)
    parser.add_argument("--batch_size", type=int, default=TrainingConfig.Batch)
    parser.add_argument("--seq_len", type=int, default=TrainingConfig.Sequence_length)
    parser.add_argument("--lr", type=float, default=TrainingConfig.lr)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--compile", action="store_true", default=True,
                         help="use torch.compile (only takes effect on CUDA)")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    max_steps = args.max_steps
    lr = args.lr
    Batch = args.batch_size
    Sequence_length = args.seq_len
    Total_batches = TrainingConfig.Total_batches

    assert Total_batches % (Batch * Sequence_length) == 0, "Total_batches must be divisible by Batch*Sequence_length"
    grad_accumulation_steps = Total_batches // (Batch * Sequence_length)
    print(f"grad_accumulation_steps: {grad_accumulation_steps}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = device
    # bf16 autocast only helps (and is only supported) on CUDA GPUs that support it;
    # fall back to plain fp32 everywhere else instead of hardcoding device_type="cuda".
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    ptdtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"using device: {device} | autocast dtype: {ptdtype}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Data
    train_loader = Dataloader(B=Batch, T=Sequence_length, split="train", data_dir=args.data_dir)
    val_loader = Dataloader(B=Batch, T=Sequence_length, split="val", data_dir=args.data_dir)

    # Model
    model = GPT(ModelConfig())
    model.to(device)

    # Build the optimizer against the un-compiled model (compiling first can prefix
    # parameter names, which is best avoided before we've grabbed named_parameters()).
    optimizer = configure_optimizers(
        model, weight_decay=0.1, learning_rate=lr, betas=(0.9, 0.95), device_type=device_type
    )

    raw_model = model  # keep an uncompiled reference for clean checkpointing
    if args.compile and device == "cuda":
        model = torch.compile(model)

    best_val_loss = float("inf")

    model.train()
    
    for step in range(max_steps):
        t0 = time.time()
        last_step = step == max_steps - 1

        if step % TrainingConfig.eval_interval == 0 or last_step:
            val_loss = estimate_loss(model, val_loader, TrainingConfig.eval_iters, device, device_type, ptdtype)
            print(f"step {step}: val_loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                    "model_config": ModelConfig(),
                }
                ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
                torch.save(checkpoint, ckpt_path)
                print(f"saved checkpoint to {ckpt_path}")

        # linear warmup + cosine decay
        step_lr = get_lr(step, TrainingConfig.warmup_steps, max_steps, lr, TrainingConfig.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = step_lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        # Gradient accumulation: sum gradients over several micro-batches before
        # stepping the optimizer, to simulate a larger effective batch size
        # (Total_batches tokens) than fits in memory at once.
        for micro_step in range(grad_accumulation_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device_type, dtype=ptdtype):
                logits = model(x)
                loss = calculate_loss(logits, y)
            loss_accum += loss.detach()
            # average the loss over accumulation steps since gradients add up otherwise
            loss = loss / grad_accumulation_steps
            loss.backward()

        # Clip the global gradient norm to 1.0 to prevent exploding gradients.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()  # wait for all queued CUDA kernels to finish before timing
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = Batch * Sequence_length * grad_accumulation_steps * 1000 / dt
        print(
            f"step {step} | loss: {loss_accum.item() / grad_accumulation_steps:.4f} "
            f"| lr: {step_lr:.2e} | norm: {norm.item():.2f} "
            f"| tok/sec: {tokens_per_sec:.2f} | time: {dt:.2f}ms"
        )


if __name__ == "__main__":
    main()