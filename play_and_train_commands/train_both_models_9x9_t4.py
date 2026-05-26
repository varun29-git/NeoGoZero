from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run unattended 9x9 ResNet and ConvNeXt training on a single T4."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("training_runs/t4_9x9"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--self-play-games", type=int, default=40)
    parser.add_argument("--mcts-rounds", type=int, default=300)
    parser.add_argument("--training-steps", type=int, default=500)
    parser.add_argument("--evaluation-games", type=int, default=5)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    weights_dir = run_dir / "trained_model_weights"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    resnet_command = [
        sys.executable,
        "play_and_train_commands/train_zero.py",
        "--board-size",
        "9",
        "--history-length",
        "8",
        "--channels",
        str(args.channels),
        "--res-blocks",
        str(args.blocks),
        "--iterations",
        str(args.iterations),
        "--self-play-games",
        str(args.self_play_games),
        "--mcts-rounds",
        str(args.mcts_rounds),
        "--max-rollout-moves",
        "243",
        "--training-steps",
        str(args.training_steps),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--replay-buffer-size",
        str(args.replay_buffer_size),
        "--evaluation-games",
        str(args.evaluation_games),
        "--checkpoint-dir",
        str(run_dir / "checkpoints_resnet_policy_value"),
        "--metrics-path",
        str(run_dir / "metrics_resnet_policy_value.jsonl"),
        "--weights-export-dir",
        str(weights_dir),
        "--device",
        args.device,
        "--seed",
        "1",
    ]
    convnext_command = [
        sys.executable,
        "policy_value_networks/convnext_policy_value/train_convnext_zero.py",
        "--board-size",
        "9",
        "--history-length",
        "8",
        "--channels",
        str(args.channels),
        "--blocks",
        str(args.blocks),
        "--iterations",
        str(args.iterations),
        "--self-play-games",
        str(args.self_play_games),
        "--mcts-rounds",
        str(args.mcts_rounds),
        "--max-rollout-moves",
        "243",
        "--training-steps",
        str(args.training_steps),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--replay-buffer-size",
        str(args.replay_buffer_size),
        "--evaluation-games",
        str(args.evaluation_games),
        "--checkpoint-dir",
        str(run_dir / "checkpoints_convnext_policy_value"),
        "--metrics-path",
        str(run_dir / "metrics_convnext_policy_value.jsonl"),
        "--weights-export-dir",
        str(weights_dir),
        "--device",
        args.device,
        "--seed",
        "2",
    ]

    commands = [
        ("resnet_policy_value", resnet_command),
        ("convnext_policy_value", convnext_command),
    ]
    print(f"Run directory: {run_dir}")
    for name, command in commands:
        print(f"{name}: {' '.join(command)}")

    if args.dry_run:
        return

    total_started_at = time.monotonic()
    finished_durations: list[float] = []
    for index, (name, command) in enumerate(commands, start=1):
        duration = _run_and_log(
            name=name,
            command=command,
            log_path=logs_dir / f"{name}.log",
            model_index=index,
            total_models=len(commands),
            started_at=total_started_at,
            finished_durations=finished_durations,
            total_iterations=args.iterations,
        )
        finished_durations.append(duration)

    print(f"Done. Weights and download bundles are in: {weights_dir}")


def _run_and_log(
    name: str,
    command: list[str],
    log_path: Path,
    model_index: int,
    total_models: int,
    started_at: float,
    finished_durations: list[float],
    total_iterations: int,
) -> float:
    print(f"Starting {name}. Log: {log_path}", flush=True)
    model_started_at = time.monotonic()
    last_eta_at = model_started_at
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
            if _line_reports_iteration(line):
                _print_eta(
                    name=name,
                    model_index=model_index,
                    total_models=total_models,
                    finished_durations=finished_durations,
                    started_at=started_at,
                    model_started_at=model_started_at,
                    completed_iterations=_iteration_from_line(line),
                    total_iterations=total_iterations,
                )
                last_eta_at = time.monotonic()
            elif time.monotonic() - last_eta_at >= 300:
                _print_eta(
                    name=name,
                    model_index=model_index,
                    total_models=total_models,
                    finished_durations=finished_durations,
                    started_at=started_at,
                    model_started_at=model_started_at,
                    completed_iterations=None,
                    total_iterations=total_iterations,
                )
                last_eta_at = time.monotonic()
        return_code = process.wait()

    if return_code != 0:
        raise SystemExit(f"{name} failed with exit code {return_code}")
    duration = time.monotonic() - model_started_at
    print(f"Finished {name} in {_format_duration(duration)}", flush=True)
    return duration


def _line_reports_iteration(line: str) -> bool:
    return line.startswith("Iteration ") or line.startswith("ConvNeXt iteration ")


def _iteration_from_line(line: str) -> int | None:
    parts = line.split()
    if line.startswith("Iteration ") and len(parts) >= 2:
        return int(parts[1].rstrip(":"))
    if line.startswith("ConvNeXt iteration ") and len(parts) >= 3:
        return int(parts[2].rstrip(":"))
    return None


def _print_eta(
    name: str,
    model_index: int,
    total_models: int,
    finished_durations: list[float],
    started_at: float,
    model_started_at: float,
    completed_iterations: int | None,
    total_iterations: int,
) -> None:
    now = time.monotonic()
    total_elapsed = now - started_at
    model_elapsed = now - model_started_at
    remaining_current_model = None
    if completed_iterations is not None and completed_iterations > 0:
        seconds_per_iteration = model_elapsed / completed_iterations
        remaining_current_model = max(total_iterations - completed_iterations, 0) * seconds_per_iteration

    completed_models = model_index - 1
    remaining_models_after_current = total_models - model_index
    average_finished_model_duration = (
        sum(finished_durations) / len(finished_durations)
        if finished_durations
        else None
    )
    if average_finished_model_duration is None:
        estimated_future_models = remaining_models_after_current * model_elapsed
    else:
        estimated_future_models = remaining_models_after_current * average_finished_model_duration

    estimated_remaining = (remaining_current_model or 0.0) + estimated_future_models
    eta_clock = datetime.now() + timedelta(seconds=estimated_remaining)
    iteration_text = (
        f"{completed_iterations}/{total_iterations}"
        if completed_iterations is not None
        else "in progress"
    )
    print(
        "[ETA] "
        f"model={name} ({model_index}/{total_models}), "
        f"iteration={iteration_text}, "
        f"completed_models={completed_models}/{total_models}, "
        f"elapsed={_format_duration(total_elapsed)}, "
        f"remaining~={_format_duration(estimated_remaining)}, "
        f"finish~={eta_clock.strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


if __name__ == "__main__":
    main()
