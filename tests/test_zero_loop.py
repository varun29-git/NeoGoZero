from __future__ import annotations

import random
import zipfile

import pytest

torch = pytest.importorskip("torch")

from go_engine.types import Player
from zero_training_pipeline.self_play import TrainingExample
from zero_training_pipeline.zero_loop import (
    ReplayBuffer,
    ZeroTrainingConfig,
    load_checkpoint,
    run_zero_training,
)
from zero_training_pipeline.weight_exports import export_checkpoint_weights


def test_replay_buffer_keeps_capacity_and_samples() -> None:
    example = TrainingExample(
        board=(),
        player=Player.BLACK,
        visit_distribution={},
        winner=Player.BLACK,
    )
    buffer = ReplayBuffer(capacity=2)

    buffer.add_examples([example, example, example])
    batch = buffer.sample(batch_size=4, rng=random.Random(1))

    assert len(buffer) == 2
    assert len(batch) == 4


def test_zero_training_writes_loadable_checkpoint(tmp_path) -> None:
    config = ZeroTrainingConfig(
        board_size=3,
        iterations=1,
        self_play_games_per_iteration=1,
        mcts_rounds=1,
        max_rollout_moves=6,
        training_steps_per_iteration=1,
        batch_size=4,
        channels=8,
        num_res_blocks=1,
        history_length=2,
        evaluation_games=0,
        checkpoint_dir=tmp_path,
        seed=3,
    )

    result = run_zero_training(config)
    model, optimizer, loaded_config, replay_buffer, iteration = load_checkpoint(
        result.final_checkpoint_path
    )

    assert result.iterations[0].generated_examples > 0
    assert result.final_checkpoint_path.exists()
    assert iteration == 1
    assert loaded_config.board_size == 3
    assert loaded_config.history_length == 2
    assert len(replay_buffer) > 0
    assert optimizer.state_dict()
    assert model.policy_size == 10


def test_zero_training_exports_downloadable_weights_bundle(tmp_path) -> None:
    config = ZeroTrainingConfig(
        board_size=3,
        iterations=1,
        self_play_games_per_iteration=1,
        mcts_rounds=1,
        max_rollout_moves=6,
        training_steps_per_iteration=1,
        batch_size=4,
        channels=8,
        num_res_blocks=1,
        checkpoint_dir=tmp_path / "checkpoints",
        seed=7,
    )

    result = run_zero_training(config)
    export = export_checkpoint_weights(
        checkpoint_path=result.final_checkpoint_path,
        architecture="resnet_policy_value",
        output_dir=tmp_path / "exports",
    )

    assert export.weights_path.exists()
    assert export.manifest_path.exists()
    assert export.bundle_path.exists()
    with zipfile.ZipFile(export.bundle_path) as bundle:
        names = set(bundle.namelist())

    assert export.weights_path.name in names
    assert export.manifest_path.name in names


def test_zero_training_writes_metrics_and_can_resume(tmp_path) -> None:
    first_config = ZeroTrainingConfig(
        board_size=3,
        iterations=1,
        self_play_games_per_iteration=1,
        mcts_rounds=1,
        max_rollout_moves=6,
        training_steps_per_iteration=1,
        batch_size=4,
        channels=8,
        num_res_blocks=1,
        checkpoint_dir=tmp_path,
        metrics_path=tmp_path / "metrics.jsonl",
        seed=4,
    )
    first_result = run_zero_training(first_config)
    resumed_config = ZeroTrainingConfig(
        board_size=3,
        iterations=1,
        self_play_games_per_iteration=1,
        mcts_rounds=1,
        max_rollout_moves=6,
        training_steps_per_iteration=1,
        batch_size=4,
        channels=8,
        num_res_blocks=1,
        checkpoint_dir=tmp_path,
        resume_checkpoint=first_result.final_checkpoint_path,
        metrics_path=tmp_path / "metrics.jsonl",
        seed=5,
    )

    resumed_result = run_zero_training(resumed_config)
    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()

    assert resumed_result.iterations[0].iteration == 2
    assert len(lines) == 2
