import argparse
import math
import os

import torch

from GPT_Config import ModelConfig, TrainingConfig
from Dataloader import Dataloader
from metrics import MetricsTracker
from model.GPT import GPT
from model.attention import Attention
from training_utils import calculate_loss, configure_optimizers, estimate_loss, get_lr


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT model for causal language modelling on WikiText-103")
    parser.add_argument("--data_dir", type=str, default=TrainingConfig.data_dir,
                         help="directory containing train.bin / val.bin (see data/wikitext103/prepare.py)")
    parser.add_argument("--out_dir", type=str, default=TrainingConfig.out_dir,
                         help="directory to write checkpoints to")
    parser.add_argument("--max_steps", type=int, default=TrainingConfig.max_steps)
    parser.add_argument("--batch_size", type=int, default=TrainingConfig.Batch)
    parser.add_argument("--seq_len", type=int, default=TrainingConfig.Sequence_length)
    parser.add_argument("--total_batches", type=int, default=TrainingConfig.Total_batches,
                         help="target tokens per optimizer step (drives gradient accumulation)")
    parser.add_argument("--lr", type=float, default=TrainingConfig.lr)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--compile", action="store_true", default=True,
                         help="use torch.compile (only takes effect on CUDA)")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--eval_interval", type=int, default=TrainingConfig.eval_interval,
                         help="run validation (and log val loss/perplexity) every N steps")
    parser.add_argument("--eval_iters", type=int, default=TrainingConfig.eval_iters,
                         help="number of random batches averaged per validation pass")
    parser.add_argument("--metrics_path", type=str, default=TrainingConfig.metrics_path,
                         help="where to write the metrics.json report")
    parser.add_argument("--log_interval", type=int, default=TrainingConfig.log_interval,
                         help="append train loss/perplexity to metrics.json every N steps")
    parser.add_argument("--metrics_warmup_steps", type=int, default=TrainingConfig.metrics_warmup_steps,
                         help="steps to discard before averaging step time / peak memory")
    parser.add_argument("--profile_attn", action="store_true", default=True,
                         help="measure time and peak memory inside the attention layers "
                              "(forces torch.compile off, since the hooks break the graph)")
    parser.add_argument("--no-profile_attn", dest="profile_attn", action="store_false")
    parser.add_argument("--ckpt_interval", type=int, default=getattr(TrainingConfig, "ckpt_interval", 500),
                         help="save a 'latest' checkpoint (for --resume) every N steps, in addition to "
                              "the best-val-loss checkpoint")
    parser.add_argument("--resume", type=str, default=None,
                         help="path to a checkpoint to resume from (restores model, optimizer, step, "
                              "dataloader position, and RNG state)")
    return parser.parse_args()


def build_checkpoint(raw_model, optimizer, step, val_loss, best_val_loss, train_loader):
    return {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "model_config": ModelConfig(),
        # sequential offset into train_loader's token stream, so resuming
        # continues from here instead of restarting at the beginning of the data
        "train_loader_position": train_loader.current_position,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


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
    Total_batches = args.total_batches

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

    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        print(f"resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        train_loader.current_position = ckpt.get("train_loader_position", 0)
        torch.set_rng_state(ckpt["rng_state"])
        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        print(f"resumed at step {start_step}, best_val_loss so far: {best_val_loss:.4f}")

    # Per-layer hooks force graph breaks, so attention profiling and torch.compile
    # can't both be on if the numbers are to describe one consistent execution mode.
    use_compile = args.compile and device == "cuda" and not args.profile_attn
    if args.compile and device == "cuda" and args.profile_attn:
        print("torch.compile disabled: --profile_attn instruments the attention layers "
              "with hooks, which break the compiled graph. Use --no-profile_attn for "
              "compiled-throughput numbers.")
    if use_compile:
        model = torch.compile(model)

    model_config = ModelConfig()
    tracker_kwargs = dict(
        path=args.metrics_path,
        run_info={
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "torch_version": torch.__version__,
            "autocast_dtype": str(ptdtype),
            "torch_compile": use_compile,
            "attention_profiling": args.profile_attn,
            "seed": args.seed,
            "num_parameters": sum(p.numel() for p in raw_model.parameters()),
            "num_parameters_non_embedding": sum(
                p.numel() for p in raw_model.parameters()
            ) - raw_model.transformer.wpe.weight.numel(),
            "model_config": vars(model_config),
            "batch_size": Batch,
            "seq_len": Sequence_length,
            "grad_accumulation_steps": grad_accumulation_steps,
            "max_steps": max_steps,
            "max_lr": lr,
            "min_lr": TrainingConfig.min_lr,
            "warmup_steps": TrainingConfig.warmup_steps,
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
            "data_dir": args.data_dir,
        },
        tokens_per_step=Batch * Sequence_length * grad_accumulation_steps,
        warmup_steps_skipped=args.metrics_warmup_steps,
        log_interval=args.log_interval,
        device_type=device_type,
    )
    if args.resume:
        tracker = MetricsTracker.resume(start_step=start_step, **tracker_kwargs)
    else:
        tracker = MetricsTracker(**tracker_kwargs)
    if args.profile_attn:
        tracker.attach_attention_profiler(raw_model, Attention)
    print(f"writing metrics to {os.path.abspath(args.metrics_path)}")

    # holds the most recently computed validation loss, so the periodic "latest"
    # checkpoint below always has a value to record even between eval_interval steps
    val_loss = best_val_loss if best_val_loss != float("inf") else float("nan")

    model.train()

    for step in range(start_step, max_steps):
        last_step = step == max_steps - 1

        if step % args.eval_interval == 0 or last_step:
            val_loss = estimate_loss(model, val_loader, args.eval_iters, device, device_type, ptdtype)
            val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
            tracker.log_validation(step, val_loss)
            tracker.write()
            print(f"step {step}: val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = os.path.join(args.out_dir, "ckpt_best.pt")
                torch.save(
                    build_checkpoint(raw_model, optimizer, step, val_loss, best_val_loss, train_loader),
                    best_ckpt_path,
                )
                print(f"saved best checkpoint to {best_ckpt_path}")

        # linear warmup + cosine decay
        step_lr = get_lr(step, TrainingConfig.warmup_steps, max_steps, lr, TrainingConfig.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = step_lr

        # Timed region starts here so the validation pass above is excluded from
        # the step time and the peak-memory high-water mark.
        tracker.start_step(step)

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        # Gradient accumulation: sum gradients over several micro-batches before
        # stepping the optimizer, to simulate a larger effective batch size
        # (Total_batches tokens) than fits in memory at once.
        for micro_step in range(grad_accumulation_steps):
            tracker.timer.start("data")
            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            tracker.timer.stop("data")

            tracker.timer.start("forward")
            with torch.autocast(device_type=device_type, dtype=ptdtype):
                logits = model(x)
                loss = calculate_loss(logits, y)
            tracker.timer.stop("forward")

            loss_accum += loss.detach()
            # average the loss over accumulation steps since gradients add up otherwise
            loss = loss / grad_accumulation_steps
            tracker.timer.start("backward")
            loss.backward()
            tracker.timer.stop("backward")

        # Clip the global gradient norm to 1.0 to prevent exploding gradients.
        tracker.timer.start("optimizer")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tracker.timer.stop("optimizer")

        train_loss = loss_accum.item() / grad_accumulation_steps
        # end_step does the single per-step cuda synchronize and resolves the timers
        dt, tokens_per_sec, phases = tracker.end_step(
            step, loss=train_loss, lr=step_lr, grad_norm=norm.item()
        )
        attn_ms = phases.get("attention_forward", 0.0) + phases.get("attention_backward", 0.0)
        attn_str = f" | attn: {attn_ms:.2f}ms" if args.profile_attn else ""
        print(
            f"step {step} | loss: {train_loss:.4f} "
            f"| lr: {step_lr:.2e} | norm: {norm.item():.2f} "
            f"| tok/sec: {tokens_per_sec:.2f} | time: {dt:.2f}ms{attn_str}"
        )

        if step % args.ckpt_interval == 0 or last_step:
            latest_ckpt_path = os.path.join(args.out_dir, "ckpt_latest.pt")
            torch.save(
                build_checkpoint(raw_model, optimizer, step, val_loss, best_val_loss, train_loader),
                latest_ckpt_path,
            )
            print(f"saved latest checkpoint to {latest_ckpt_path}")

    tracker.write(status="completed")
    print(f"wrote metrics to {os.path.abspath(args.metrics_path)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # metrics.json is rewritten on every logged step, so an interrupted run
        # still leaves the curves and aggregates collected so far on disk.
        print("interrupted; metrics.json holds everything up to the last logged step")