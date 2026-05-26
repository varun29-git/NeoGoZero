from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convnext_impl.convnext_zero_loop import (
    ConvNeXtTrainingConfig,
    run_convnext_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ConvNeXt NeoGoZero loop.")
    parser.add_argument("--board-size", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--self-play-games", type=int, default=1)
    parser.add_argument("--mcts-rounds", type=int, default=2)
    parser.add_argument("--max-rollout-moves", type=int, default=12)
    parser.add_argument("--training-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--history-length", type=int, default=1)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--layer-scale-init", type=float, default=1e-6)
    parser.add_argument("--stochastic-depth-prob", type=float, default=0.1)
    parser.add_argument("--replay-buffer-size", type=int, default=1000)
    parser.add_argument("--evaluation-games", type=int, default=0)
    parser.add_argument("--promotion-threshold", type=float, default=0.55)
    parser.add_argument("--self-play-temperature", type=float, default=1.0)
    parser.add_argument("--temperature-drop-move", type=int, default=30)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.03)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("convnext_checkpoints"),
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    config = ConvNeXtTrainingConfig(
        board_size=args.board_size,
        iterations=args.iterations,
        self_play_games_per_iteration=args.self_play_games,
        mcts_rounds=args.mcts_rounds,
        max_rollout_moves=args.max_rollout_moves,
        training_steps_per_iteration=args.training_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        channels=args.channels,
        num_blocks=args.blocks,
        history_length=args.history_length,
        kernel_size=args.kernel_size,
        mlp_ratio=args.mlp_ratio,
        layer_scale_init=args.layer_scale_init,
        stochastic_depth_prob=args.stochastic_depth_prob,
        replay_buffer_size=args.replay_buffer_size,
        evaluation_games=args.evaluation_games,
        promotion_threshold=args.promotion_threshold,
        self_play_temperature=args.self_play_temperature,
        temperature_drop_move=args.temperature_drop_move,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        checkpoint_dir=args.checkpoint_dir,
        resume_checkpoint=args.resume_checkpoint,
        metrics_path=args.metrics_path,
        seed=args.seed,
    )
    result = run_convnext_training(config)

    for iteration in result.iterations:
        print(
            "ConvNeXt iteration "
            f"{iteration.iteration}: "
            f"examples={iteration.generated_examples}, "
            f"loss={iteration.mean_loss:.4f}, "
            f"candidate_win_rate={iteration.candidate_win_rate:.1%}, "
            f"promoted={iteration.promoted}"
        )
    print(f"Final checkpoint: {result.final_checkpoint_path}")


if __name__ == "__main__":
    main()
