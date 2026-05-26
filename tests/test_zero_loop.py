from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from myalphago.go.types import Player
from myalphago.training.self_play import TrainingExample
from myalphago.training.zero_loop import (
    ReplayBuffer,
    ZeroTrainingConfig,
    load_checkpoint,
    run_zero_training,
)


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
    assert len(replay_buffer) > 0
    assert optimizer.state_dict()
    assert model.policy_size == 10
