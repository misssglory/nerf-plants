#!/usr/bin/env python3
"""Run Nerfstudio training and stop when held-out rendering quality plateaus.

The supervisor reads TensorBoard scalars written by Nerfstudio 1.1.x. It does
not modify Nerfstudio itself, so it remains compatible with the pinned Pixi
environment used by this project.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EvalPoint:
    step: int
    psnr: float
    ssim: float | None
    lpips: float
    train_loss: float | None
    eval_loss: float | None


@dataclass
class EarlyStopState:
    best_step: int | None = None
    best_psnr: float = -math.inf
    best_ssim: float = -math.inf
    best_lpips: float = math.inf
    best_eval_loss: float = math.inf
    stale_evaluations: int = 0
    evaluations_seen: int = 0
    stopped_early: bool = False
    stop_reason: str | None = None


def finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def latest_at_or_before(events: Sequence, step: int) -> float | None:
    candidates = [event for event in events if int(event.step) <= step]
    if not candidates:
        return None
    return finite_or_none(float(candidates[-1].value))


def events_by_step(events: Sequence) -> dict[int, float]:
    result: dict[int, float] = {}
    for event in events:
        value = finite_or_none(float(event.value))
        if value is not None:
            result[int(event.step)] = value
    return result


def pick_tag(tags: Iterable[str], *, suffix: str, contains: Sequence[str] = ()) -> str | None:
    suffix_lower = suffix.lower()
    required = tuple(value.lower() for value in contains)
    matches = []
    for tag in tags:
        lower = tag.lower()
        if lower.endswith(suffix_lower) and all(value in lower for value in required):
            matches.append(tag)
    if not matches:
        return None
    # Prefer the full all-images validation metric over similarly named tags.
    matches.sort(key=lambda tag: ("all images" not in tag.lower(), len(tag)))
    return matches[0]


def relative_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous == 0:
        return None
    return 100.0 * (current - previous) / abs(previous)


def fmt(value: float | None, digits: int = 5) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_delta(value: float | None, digits: int = 5) -> str:
    return "n/a" if value is None else f"{value:+.{digits}f}"


def decide_improvement(
    point: EvalPoint,
    state: EarlyStopState,
    *,
    min_steps: int,
    min_psnr_delta: float,
    min_lpips_delta: float,
) -> tuple[bool, str]:
    """Return whether a validation point materially improves rendering quality."""
    if point.step < min_steps:
        return False, "warmup"
    if state.best_step is None:
        return True, "baseline"

    psnr_gain = point.psnr - state.best_psnr
    lpips_gain = state.best_lpips - point.lpips
    reasons = []
    if psnr_gain >= min_psnr_delta:
        reasons.append(f"PSNR {psnr_gain:+.4f} dB")
    if lpips_gain >= min_lpips_delta:
        reasons.append(f"LPIPS {lpips_gain:+.5f}")
    return bool(reasons), ", ".join(reasons) if reasons else "plateau"


def nearest_checkpoint(checkpoint_dir: Path, step: int) -> Path | None:
    exact = checkpoint_dir / f"step-{step:09d}.ckpt"
    if exact.is_file():
        return exact

    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("step-*.ckpt"):
        try:
            candidate_step = int(path.stem.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if candidate_step <= step:
            candidates.append((candidate_step, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def preserve_best_checkpoint(run_dir: Path, step: int) -> Path | None:
    checkpoint_dir = run_dir / "nerfstudio_models"
    source = nearest_checkpoint(checkpoint_dir, step)
    if source is None:
        return None

    best_dir = run_dir / "best_checkpoint"
    best_dir.mkdir(parents=True, exist_ok=True)
    destination = best_dir / "best.ckpt"
    temporary = best_dir / ".best.ckpt.tmp"
    temporary.unlink(missing_ok=True)

    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)

    config = run_dir / "config.yml"
    if config.is_file():
        shutil.copy2(config, best_dir / "config.yml")

    (best_dir / "best.json").write_text(
        json.dumps(
            {
                "step": step,
                "checkpoint": str(source),
                "preserved_checkpoint": str(destination),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def append_csv(path: Path, point: EvalPoint, state: EarlyStopState, status: str, reason: str) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "train_loss",
                "eval_loss",
                "psnr",
                "ssim",
                "lpips",
                "status",
                "reason",
                "best_step",
                "best_psnr",
                "best_ssim",
                "best_lpips",
                "stale_evaluations",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "step": point.step,
                "train_loss": point.train_loss,
                "eval_loss": point.eval_loss,
                "psnr": point.psnr,
                "ssim": point.ssim,
                "lpips": point.lpips,
                "status": status,
                "reason": reason,
                "best_step": state.best_step,
                "best_psnr": None if state.best_step is None else state.best_psnr,
                "best_ssim": None if state.best_ssim == -math.inf else state.best_ssim,
                "best_lpips": None if state.best_step is None else state.best_lpips,
                "stale_evaluations": state.stale_evaluations,
            }
        )


def write_summary(path: Path, state: EarlyStopState, settings: dict, return_code: int | None) -> None:
    payload = {
        "state": {
            **asdict(state),
            "best_psnr": None if state.best_step is None else state.best_psnr,
            "best_ssim": None if state.best_ssim == -math.inf else state.best_ssim,
            "best_lpips": None if state.best_step is None else state.best_lpips,
            "best_eval_loss": None if state.best_eval_loss == math.inf else state.best_eval_loss,
        },
        "settings": settings,
        "training_return_code": return_code,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def terminate_process_group(process: subprocess.Popen, *, timeout: float = 30.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def load_points(run_dir: Path, last_step: int) -> tuple[list[EvalPoint], list[str]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:  # pragma: no cover - depends on Pixi environment
        raise RuntimeError(
            "TensorBoard is required for early stopping. Run `pixi install` after updating pixi.toml."
        ) from exc

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = list(accumulator.Tags().get("scalars", []))

    psnr_tag = pick_tag(tags, suffix="/psnr", contains=("eval", "all images"))
    lpips_tag = pick_tag(tags, suffix="/lpips", contains=("eval", "all images"))
    ssim_tag = pick_tag(tags, suffix="/ssim", contains=("eval", "all images"))

    missing = []
    if psnr_tag is None:
        missing.append("all-image PSNR")
    if lpips_tag is None:
        missing.append("all-image LPIPS")
    if missing:
        return [], missing

    psnr_by_step = events_by_step(accumulator.Scalars(psnr_tag))
    lpips_by_step = events_by_step(accumulator.Scalars(lpips_tag))
    ssim_by_step = events_by_step(accumulator.Scalars(ssim_tag)) if ssim_tag else {}

    train_events = accumulator.Scalars("Train Loss") if "Train Loss" in tags else []
    eval_events = accumulator.Scalars("Eval Loss") if "Eval Loss" in tags else []

    common_steps = sorted((set(psnr_by_step) & set(lpips_by_step)) - set(range(last_step + 1)))
    points = [
        EvalPoint(
            step=step,
            psnr=psnr_by_step[step],
            ssim=ssim_by_step.get(step),
            lpips=lpips_by_step[step],
            train_loss=latest_at_or_before(train_events, step),
            eval_loss=latest_at_or_before(eval_events, step),
        )
        for step in common_steps
    ]
    return points, []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-psnr-delta", type=float, default=0.03)
    parser.add_argument("--min-lpips-delta", type=float, default=0.002)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing training command after --")
    if args.min_steps < 0 or args.patience < 1:
        parser.error("min-steps must be >= 0 and patience must be >= 1")
    if args.min_psnr_delta < 0 or args.min_lpips_delta < 0:
        parser.error("quality deltas must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    run_dir: Path = args.run_dir.resolve()
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    state = EarlyStopState()
    history_path = run_dir / "early_stopping.csv"
    summary_path = run_dir / "early_stopping.json"
    settings = {
        "min_steps": args.min_steps,
        "patience": args.patience,
        "min_psnr_delta": args.min_psnr_delta,
        "min_lpips_delta": args.min_lpips_delta,
        "poll_seconds": args.poll_seconds,
        "run_dir": str(run_dir),
        "command": args.command,
    }

    print("\nEarly stopping supervisor")
    print(f"  run directory:  {run_dir}")
    print(f"  minimum steps:  {args.min_steps}")
    print(f"  patience:       {args.patience} full validations")
    print(f"  PSNR min gain:  {args.min_psnr_delta:.4f} dB")
    print(f"  LPIPS min gain: {args.min_lpips_delta:.5f}")
    print("  criterion:      PSNR improves OR LPIPS decreases\n", flush=True)

    process = subprocess.Popen(args.command, start_new_session=True)
    user_interrupted = False

    def forward_signal(signum, _frame):
        nonlocal user_interrupted
        user_interrupted = True
        print(f"\nForwarding signal {signum} to training process...", flush=True)
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)

    last_step = -1
    previous_point: EvalPoint | None = None
    missing_reported = False

    try:
        while process.poll() is None:
            if run_dir.exists():
                try:
                    points, missing = load_points(run_dir, last_step)
                except RuntimeError as exc:
                    terminate_process_group(process)
                    print(f"early-stop error: {exc}", file=sys.stderr)
                    return 2
                except Exception as exc:  # TensorBoard can observe a partially written event file.
                    print(f"[early-stop] metric read deferred: {exc}", flush=True)
                    points, missing = [], []

                if missing and not missing_reported:
                    print(
                        "[early-stop] waiting for validation metrics: " + ", ".join(missing),
                        flush=True,
                    )
                    missing_reported = True

                for point in points:
                    last_step = max(last_step, point.step)
                    state.evaluations_seen += 1
                    improved, reason = decide_improvement(
                        point,
                        state,
                        min_steps=args.min_steps,
                        min_psnr_delta=args.min_psnr_delta,
                        min_lpips_delta=args.min_lpips_delta,
                    )

                    train_delta = None if previous_point is None else (
                        None
                        if point.train_loss is None or previous_point.train_loss is None
                        else point.train_loss - previous_point.train_loss
                    )
                    eval_delta = None if previous_point is None else (
                        None
                        if point.eval_loss is None or previous_point.eval_loss is None
                        else point.eval_loss - previous_point.eval_loss
                    )
                    train_pct = relative_change(
                        point.train_loss,
                        None if previous_point is None else previous_point.train_loss,
                    )
                    eval_pct = relative_change(
                        point.eval_loss,
                        None if previous_point is None else previous_point.eval_loss,
                    )

                    status = "WARMUP"
                    if point.step >= args.min_steps:
                        if improved:
                            state.best_step = point.step
                            state.best_psnr = max(state.best_psnr, point.psnr)
                            state.best_ssim = max(state.best_ssim, point.ssim or -math.inf)
                            state.best_lpips = min(state.best_lpips, point.lpips)
                            if point.eval_loss is not None:
                                state.best_eval_loss = min(state.best_eval_loss, point.eval_loss)
                            state.stale_evaluations = 0
                            status = "IMPROVED"
                            saved = preserve_best_checkpoint(run_dir, point.step)
                            if saved is None:
                                print(
                                    f"[early-stop] checkpoint for step {point.step} is not visible yet; "
                                    "it will be retried at the next improvement.",
                                    flush=True,
                                )
                        else:
                            state.stale_evaluations += 1
                            status = "PLATEAU"

                    print(
                        "[quality] "
                        f"step={point.step:>7d}  "
                        f"train_loss={fmt(point.train_loss)} Δ={fmt_delta(train_delta)} "
                        f"({fmt_delta(train_pct, 2)}%)  "
                        f"eval_loss={fmt(point.eval_loss)} Δ={fmt_delta(eval_delta)} "
                        f"({fmt_delta(eval_pct, 2)}%)\n"
                        "          "
                        f"PSNR={point.psnr:.4f} dB  "
                        f"SSIM={fmt(point.ssim, 4)}  "
                        f"LPIPS={point.lpips:.5f}  "
                        f"status={status} ({reason})  "
                        f"patience={state.stale_evaluations}/{args.patience}",
                        flush=True,
                    )
                    append_csv(history_path, point, state, status, reason)
                    write_summary(summary_path, state, settings, process.poll())
                    previous_point = point

                    if point.step >= args.min_steps and state.stale_evaluations >= args.patience:
                        state.stopped_early = True
                        state.stop_reason = (
                            f"no PSNR gain >= {args.min_psnr_delta} dB and no LPIPS decrease "
                            f">= {args.min_lpips_delta} for {args.patience} validations"
                        )
                        print(
                            f"\n[early-stop] plateau detected at step {point.step}. "
                            f"Best validation step: {state.best_step}.\n"
                            "[early-stop] stopping Nerfstudio and preserving best_checkpoint/best.ckpt.",
                            flush=True,
                        )
                        terminate_process_group(process)
                        break

            if state.stopped_early:
                break
            time.sleep(max(args.poll_seconds, 0.5))
    finally:
        if user_interrupted and process.poll() is None:
            terminate_process_group(process)

    return_code = process.wait() if process.poll() is None else int(process.returncode or 0)
    write_summary(summary_path, state, settings, return_code)

    print("\nTraining summary")
    print(f"  evaluations observed: {state.evaluations_seen}")
    print(f"  stopped early:        {state.stopped_early}")
    if state.best_step is not None:
        print(f"  best step:            {state.best_step}")
        print(f"  best PSNR:            {state.best_psnr:.4f} dB")
        print(f"  best SSIM:            {fmt(None if state.best_ssim == -math.inf else state.best_ssim, 4)}")
        print(f"  best LPIPS:           {state.best_lpips:.5f}")
        print(f"  best checkpoint:      {run_dir / 'best_checkpoint' / 'best.ckpt'}")
    print(f"  metrics CSV:          {history_path}")
    print(f"  summary JSON:         {summary_path}")

    if user_interrupted:
        return 130
    if state.stopped_early:
        return 0
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
