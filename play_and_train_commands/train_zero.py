from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zero_training_pipeline.zero_loop import ZeroTrainingConfig, run_zero_training
from zero_training_pipeline.weight_exports import export_checkpoint_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NeoGoZero training loop.")
    parser.add_argument("--board-size", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--self-play-games", type=int, default=1)
    parser.add_argument("--mcts-rounds", type=int, default=2)
    parser.add_argument("--max-rollout-moves", type=int, default=12)
    parser.add_argument("--training-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--res-blocks", type=int, default=2)
    parser.add_argument("--history-length", type=int, default=1)
    parser.add_argument("--replay-buffer-size", type=int, default=1000)
    parser.add_argument("--evaluation-games", type=int, default=0)
    parser.add_argument("--promotion-threshold", type=float, default=0.55)
    parser.add_argument("--self-play-temperature", type=float, default=1.0)
    parser.add_argument("--temperature-drop-move", type=int, default=30)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.03)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--supervised-sgf-dir", type=Path, default=None)
    parser.add_argument("--supervised-steps", type=int, default=0)
    parser.add_argument("--supervised-max-examples", type=int, default=None)
    parser.add_argument("--supervised-batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--weights-export-dir", type=Path, default=Path("trained_model_weights"))
    parser.add_argument("--skip-weights-export", action="store_true")
    parser.add_argument("--auto-download-weights", action="store_true")
    args = parser.parse_args()

    config = ZeroTrainingConfig(
        board_size=args.board_size,
        iterations=args.iterations,
        self_play_games_per_iteration=args.self_play_games,
        mcts_rounds=args.mcts_rounds,
        max_rollout_moves=args.max_rollout_moves,
        training_steps_per_iteration=args.training_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        channels=args.channels,
        num_res_blocks=args.res_blocks,
        history_length=args.history_length,
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
        supervised_sgf_dir=args.supervised_sgf_dir,
        supervised_steps=args.supervised_steps,
        supervised_max_examples=args.supervised_max_examples,
        supervised_batch_size=args.supervised_batch_size,
        seed=args.seed,
        device=args.device,
    )
    result = run_zero_training(config)

    for iteration in result.iterations:
        print(
            "Iteration "
            f"{iteration.iteration}: "
            f"examples={iteration.generated_examples}, "
            f"loss={iteration.mean_loss:.4f}, "
            f"candidate_win_rate={iteration.candidate_win_rate:.1%}, "
            f"promoted={iteration.promoted}"
        )
    print(f"Final checkpoint: {result.final_checkpoint_path}")
    if not args.skip_weights_export:
        export = export_checkpoint_weights(
            checkpoint_path=result.final_checkpoint_path,
            architecture="resnet_policy_value",
            output_dir=args.weights_export_dir,
            auto_download=args.auto_download_weights,
        )
        print(f"Final weights: {export.weights_path}")
        print(f"Download bundle: {export.bundle_path}")
        if args.auto_download_weights and not export.auto_download_started:
            print("Auto-download is only available in Colab; bundle was still created.")


if __name__ == "__main__":
    main()
