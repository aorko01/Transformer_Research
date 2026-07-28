"""
Metrics collection for training runs, serialised to `metrics.json`.

Two kinds of numbers are recorded, because they answer different questions:

* **Curves** - train loss / val loss / val perplexity, appended every N steps.
  These are per-step time series and are meant to be plotted.
* **Aggregates** - step time, tokens/sec, peak memory, and the time+memory spent
  inside the attention layers. A single training step is noisy (allocator warm-up,
  cuDNN autotuning, `torch.compile` recompiles), so the first
  `warmup_steps_skipped` steps are thrown away and everything after that is
  summarised as mean/std/median/min/max/p90 rather than stored per step.

Timing on CUDA uses CUDA events instead of `time.perf_counter()`: kernel launches
are asynchronous, so a host-side timer around a launch measures the launch, not
the work. Events are recorded in-stream and resolved once per step after a single
`torch.cuda.synchronize()`, which keeps the instrumentation off the critical path.
"""

import json
import math
import os
import statistics
import time
from datetime import datetime, timezone

import torch

MIB = 1024 ** 2


def _summarize(values):
    """mean/std/median/min/max/p90 for a list of samples (None if empty)."""
    if not values:
        return None
    vals = sorted(values)
    n = len(vals)
    return {
        "count": n,
        "mean": sum(vals) / n,
        "std": statistics.stdev(vals) if n > 1 else 0.0,
        "median": statistics.median(vals),
        "min": vals[0],
        "max": vals[-1],
        "p90": vals[min(n - 1, int(round(0.9 * (n - 1))))],
    }


class _Timer:
    """
    Accumulates named durations for the current step.

    On CUDA every `start`/`stop` pair records a pair of CUDA events; the elapsed
    times are only read in `drain()`, which the caller invokes after synchronising.
    Events are pooled and reused so a long run doesn't allocate thousands of them.
    Repeated start/stop of the same label within a step sum together (e.g. the
    forward pass runs once per gradient-accumulation micro-step).
    """

    def __init__(self, device_type):
        self.cuda = device_type == "cuda"
        self._pool = []
        self._in_use = 0
        self._pending = []
        self._open = {}

    def _event(self):
        if self._in_use == len(self._pool):
            self._pool.append(torch.cuda.Event(enable_timing=True))
        event = self._pool[self._in_use]
        self._in_use += 1
        return event

    def start(self, label):
        if self.cuda:
            event = self._event()
            event.record()
            self._open[label] = event
        else:
            self._open[label] = time.perf_counter()

    def stop(self, label):
        started = self._open.pop(label, None)
        if started is None:  # stop without a matching start (e.g. profiler paused mid-step)
            return
        if self.cuda:
            end = self._event()
            end.record()
            self._pending.append((label, started, end))
        else:
            self._pending.append((label, (time.perf_counter() - started) * 1000.0))

    def drain(self):
        """Resolve and clear everything recorded since the last drain. Call after sync."""
        totals = {}
        for item in self._pending:
            if self.cuda:
                label, start, end = item
                totals[label] = totals.get(label, 0.0) + start.elapsed_time(end)
            else:
                label, elapsed_ms = item
                totals[label] = totals.get(label, 0.0) + elapsed_ms
        self._pending.clear()
        self._open.clear()
        self._in_use = 0
        return totals


class AttentionProfiler:
    """
    Measures wall time and peak memory inside the attention layers only.

    Memory is tracked in *segments*: on entering attention we read the running
    allocator peak, fold it into the pipeline peak, then call
    `reset_peak_memory_stats()` (which rebases the peak to the current allocation).
    On exit we read the peak again - that value is the true high-water mark reached
    while inside attention. Taking a max over every segment therefore still yields
    an exact whole-pipeline peak, without the attention measurement and the
    pipeline measurement clobbering each other.

    Two attention memory numbers are reported:
      * `absolute`  - high-water mark of the whole process while inside attention,
                      which includes everything earlier layers are still holding.
      * `delta_over_entry` - peak minus what was already allocated on entry, i.e.
                      the memory attributable to the attention layer itself.

    Note: this attaches forward/backward hooks, which force `torch.compile` to
    break the graph. `train.py` disables compilation when this profiler is active
    so the reported timings describe one consistent execution mode.
    """

    def __init__(self, model, module_cls, timer, tracker):
        self.timer = timer
        self.tracker = tracker
        self.enabled = False
        self.layers = 0
        self._entry_alloc = 0
        self.reset_step()

        for module in model.modules():
            if isinstance(module, module_cls):
                self.layers += 1
                module.register_forward_pre_hook(self._forward_pre)
                module.register_forward_hook(self._forward_post)
                module.register_full_backward_pre_hook(self._backward_pre)
                module.register_full_backward_hook(self._backward_post)

    def reset_step(self):
        self.forward_calls = 0
        self.backward_calls = 0
        self.peak_absolute = 0
        self.peak_delta = 0

    # --- shared segment bookkeeping -------------------------------------------------
    def _open_segment(self):
        if not self.tracker.cuda:
            return
        self.tracker.observe_peak()
        torch.cuda.reset_peak_memory_stats()
        self._entry_alloc = torch.cuda.memory_allocated()

    def _close_segment(self):
        if not self.tracker.cuda:
            return
        peak = torch.cuda.max_memory_allocated()
        self.peak_absolute = max(self.peak_absolute, peak)
        self.peak_delta = max(self.peak_delta, peak - self._entry_alloc)
        self.tracker.observe_peak()
        torch.cuda.reset_peak_memory_stats()

    # --- hooks ----------------------------------------------------------------------
    def _forward_pre(self, module, args):
        if not self.enabled:
            return
        self._open_segment()
        self.timer.start("attention_forward")

    def _forward_post(self, module, args, output):
        if not self.enabled:
            return
        self.timer.stop("attention_forward")
        self._close_segment()
        self.forward_calls += 1

    def _backward_pre(self, module, grad_output):
        if not self.enabled:
            return
        self._open_segment()
        self.timer.start("attention_backward")

    def _backward_post(self, module, grad_input, grad_output):
        if not self.enabled:
            return
        self.timer.stop("attention_backward")
        self._close_segment()
        self.backward_calls += 1


class MetricsTracker:
    """
    Collects curves + aggregates and writes them to `metrics.json`.

    Typical use inside the training loop::

        tracker.start_step(step)
        ...  tracker.timer.start("forward") / .stop("forward") ...
        tracker.end_step(step, loss=..., lr=..., grad_norm=...)

    `end_step` performs the single per-step CUDA synchronize, so the caller should
    not add another one.
    """

    def __init__(self, path, run_info, tokens_per_step, warmup_steps_skipped,
                 log_interval, device_type):
        self.path = path
        self.tokens_per_step = tokens_per_step
        self.warmup_steps_skipped = warmup_steps_skipped
        self.log_interval = log_interval
        self.cuda = device_type == "cuda"
        self.timer = _Timer(device_type)
        self.attention = None

        self.run_info = dict(run_info)
        self.run_info.update({
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tokens_per_step": tokens_per_step,
            "warmup_steps_skipped": warmup_steps_skipped,
            "log_interval": log_interval,
        })

        self.train_curve = []
        self.val_curve = []

        # aggregate samples, only collected for steps >= warmup_steps_skipped
        self._step_ms = []
        self._tokens_per_sec = []
        self._phase_ms = {}
        self._attn_forward_ms = []
        self._attn_backward_ms = []
        self._step_peak_alloc = []
        self._step_peak_reserved = []
        self._attn_peak_absolute = 0
        self._attn_peak_delta = 0
        self._peak_alloc = 0
        self._peak_reserved = 0

        self._t0 = None
        self._measuring = False
        self._wall_start = time.time()

    def attach_attention_profiler(self, model, module_cls):
        self.attention = AttentionProfiler(model, module_cls, self.timer, self)
        self.run_info["attention_layers_profiled"] = self.attention.layers
        return self.attention

    # --- memory ---------------------------------------------------------------------
    def observe_peak(self):
        """Fold the allocator's current running peak into this step's high-water marks."""
        if not self.cuda:
            return
        self._peak_alloc = max(self._peak_alloc, torch.cuda.max_memory_allocated())
        self._peak_reserved = max(self._peak_reserved, torch.cuda.max_memory_reserved())

    # --- step lifecycle -------------------------------------------------------------
    def start_step(self, step):
        """Begin timing a step. Call *after* any validation pass so eval memory and
        eval time are excluded from the training-step numbers."""
        self._measuring = step >= self.warmup_steps_skipped
        self._peak_alloc = 0
        self._peak_reserved = 0
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
        if self.attention is not None:
            self.attention.enabled = True
            self.attention.reset_step()
        self._t0 = time.perf_counter()

    def end_step(self, step, loss, lr, grad_norm):
        """Close out a step: synchronize once, resolve timers, record samples."""
        if self.cuda:
            torch.cuda.synchronize()
        step_ms = (time.perf_counter() - self._t0) * 1000.0
        tokens_per_sec = self.tokens_per_step * 1000.0 / step_ms

        phases = self.timer.drain()
        self.observe_peak()
        if self.attention is not None:
            self.attention.enabled = False

        if self._measuring:
            self._step_ms.append(step_ms)
            self._tokens_per_sec.append(tokens_per_sec)
            for label, value in phases.items():
                self._phase_ms.setdefault(label, []).append(value)
            if self.cuda:
                self._step_peak_alloc.append(self._peak_alloc)
                self._step_peak_reserved.append(self._peak_reserved)
            if self.attention is not None:
                self._attn_forward_ms.append(phases.get("attention_forward", 0.0))
                self._attn_backward_ms.append(phases.get("attention_backward", 0.0))
                self._attn_peak_absolute = max(self._attn_peak_absolute, self.attention.peak_absolute)
                self._attn_peak_delta = max(self._attn_peak_delta, self.attention.peak_delta)

        if step % self.log_interval == 0 or step == self.run_info.get("max_steps", 0) - 1:
            self.train_curve.append({
                "step": step,
                "train_loss": loss,
                "train_perplexity": _safe_exp(loss),
                "lr": lr,
                "grad_norm": grad_norm,
                "step_time_ms": step_ms,
                "tokens_per_sec": tokens_per_sec,
                "tokens_seen": (step + 1) * self.tokens_per_step,
            })
            # Flush on every logged step. The file is small and a step takes seconds,
            # so this is free, and a killed run still leaves complete metrics behind.
            self.write()

        return step_ms, tokens_per_sec, phases

    def log_validation(self, step, val_loss):
        self.val_curve.append({
            "step": step,
            "val_loss": val_loss,
            "val_perplexity": _safe_exp(val_loss),
            "tokens_seen": step * self.tokens_per_step,
        })

    # --- output ---------------------------------------------------------------------
    def aggregates(self):
        out = {
            "measured_steps": len(self._step_ms),
            "note": (
                f"the first {self.warmup_steps_skipped} steps are excluded; "
                "times are per optimizer step (all gradient-accumulation micro-steps)"
            ),
            "step_time_ms": _summarize(self._step_ms),
            "tokens_per_sec": _summarize(self._tokens_per_sec),
            "phase_time_ms": {k: _summarize(v) for k, v in sorted(self._phase_ms.items())},
            "phase_time_note": (
                "data/forward/backward/optimizer partition the step; attention_forward "
                "and attention_backward are nested inside forward/backward, not additional to them"
            ),
        }

        if self.cuda:
            out["peak_memory_per_step_bytes"] = {
                "allocated": _summarize(self._step_peak_alloc),
                "reserved": _summarize(self._step_peak_reserved),
            }
            out["peak_memory_mib"] = {
                "allocated": max(self._step_peak_alloc, default=0) / MIB,
                "reserved": max(self._step_peak_reserved, default=0) / MIB,
            }

        if self.attention is not None and self._step_ms:
            fwd = _summarize(self._attn_forward_ms)
            bwd = _summarize(self._attn_backward_ms)
            total = [f + b for f, b in zip(self._attn_forward_ms, self._attn_backward_ms)]
            total_summary = _summarize(total)
            step_mean = out["step_time_ms"]["mean"] if out["step_time_ms"] else None
            attention = {
                "layers": self.attention.layers,
                "forward_ms_per_step": fwd,
                "backward_ms_per_step": bwd,
                "total_ms_per_step": total_summary,
                "fraction_of_step_time": (
                    total_summary["mean"] / step_mean if step_mean else None
                ),
            }
            if self.cuda:
                attention["peak_memory_bytes"] = {
                    "absolute": self._attn_peak_absolute,
                    "delta_over_entry": self._attn_peak_delta,
                }
                attention["peak_memory_mib"] = {
                    "absolute": self._attn_peak_absolute / MIB,
                    "delta_over_entry": self._attn_peak_delta / MIB,
                }
                attention["peak_memory_note"] = (
                    "'absolute' is the process high-water mark while inside an attention "
                    "layer (includes activations held by earlier layers); "
                    "'delta_over_entry' subtracts what was already allocated on entry and "
                    "is the memory attributable to attention itself"
                )
            out["attention"] = attention

        return out

    def snapshot(self):
        return {
            "run": self.run_info,
            "aggregates": self.aggregates(),
            "train": self.train_curve,
            "val": self.val_curve,
        }

    def write(self, status="running"):
        """Atomically (re)write metrics.json. Cheap enough to call at every eval."""
        payload = self.snapshot()
        payload["run"]["status"] = status
        payload["run"]["elapsed_wall_clock_sec"] = time.time() - self._wall_start
        payload["run"]["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, self.path)


def _safe_exp(loss):
    """exp(loss), guarding against overflow on an early/diverged run."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")
