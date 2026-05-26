from __future__ import annotations

import random
import zipfile

import pytest

torch = pytest.importorskip("torch")

from policy_value_networks.convnext_policy_value.convnext_policy_value import (
    ConvNeXtBlock,
    ConvNeXtPolicyValueEvaluator,
    ConvNeXtPolicyValueNet,
    LayerNorm2d,
)
from policy_value_networks.convnext_policy_value.convnext_zero_loop import (
    ConvNeXtTrainingConfig,
    load_convnext_checkpoint,
    run_convnext_training,
)
from search_players.mcts_bot import MCTSBot
from go_engine.game import GameState
from zero_training_pipeline.self_play import generate_self_play_game
from zero_training_pipeline.torch_training import train_step
from zero_training_pipeline.weight_exports import export_checkpoint_weights


def test_convnext_forward_shapes() -> None:
    model = ConvNeXtPolicyValueNet(
        board_size=3,
        input_planes=3,
        channels=8,
        num_blocks=2,
    )
    boards = torch.zeros((2, 3, 3, 3), dtype=torch.float32)

    policy_logits, values = model(boards)

    assert policy_logits.shape == (2, 10)
    assert values.shape == (2,)


def test_convnext_block_uses_depthwise_conv_and_layer_norm() -> None:
    block = ConvNeXtBlock(channels=8, stochastic_depth_prob=0.0)

    assert block.depthwise_conv.groups == 8
    assert isinstance(block.norm, LayerNorm2d)


def test_convnext_evaluator_returns_legal_priors() -> None:
    game = GameState.new_game(board_size=3)
    model = ConvNeXtPolicyValueNet(
        board_size=3,
        input_planes=3,
        channels=8,
        num_blocks=1,
    )
    evaluator = ConvNeXtPolicyValueEvaluator(model)

    evaluation = evaluator.evaluate(game)

    assert set(evaluation.move_priors) == set(game.legal_moves())
    assert sum(evaluation.move_priors.values()) == pytest.approx(1.0)
    assert -1.0 <= evaluation.value <= 1.0


def test_convnext_train_step_returns_finite_loss() -> None:
    search_bot = MCTSBot(num_rounds=2, max_rollout_moves=6, rng=random.Random(1))
    game = generate_self_play_game(search_bot, board_size=3, max_moves=12)
    model = ConvNeXtPolicyValueNet(
        board_size=3,
        input_planes=3,
        channels=8,
        num_blocks=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    loss = train_step(model, optimizer, game.examples[:4], board_size=3)

    assert loss > 0
    assert torch.isfinite(torch.tensor(loss))


def test_convnext_training_writes_loadable_checkpoint(tmp_path) -> None:
    config = ConvNeXtTrainingConfig(
        board_size=3,
        iterations=1,
        self_play_games_per_iteration=1,
        mcts_rounds=1,
        max_rollout_moves=6,
        training_steps_per_iteration=1,
        batch_size=4,
        channels=8,
        num_blocks=1,
        evaluation_games=0,
        checkpoint_dir=tmp_path,
        metrics_path=tmp_path / "metrics.jsonl",
        seed=3,
    )

    result = run_convnext_training(config)
    model, optimizer, loaded_config, replay_buffer, iteration = load_convnext_checkpoint(
        result.final_checkpoint_path
    )

    assert result.final_checkpoint_path.exists()
    assert (tmp_path / "metrics.jsonl").exists()
    assert iteration == 1
    assert loaded_config.num_blocks == 1
    assert len(replay_buffer) > 0
    assert optimizer.state_dict()
    assert model.policy_size == 10


def test_convnext_training_exports_downloadable_weights_bundle(tmp_path) -> None:
    config = ConvNeXtTrainingConfig(
        board_size=3,
        iterations=1,
        self_play_games_per_iteration=1,
        mcts_rounds=1,
        max_rollout_moves=6,
        training_steps_per_iteration=1,
        batch_size=4,
        channels=8,
        num_blocks=1,
        evaluation_games=0,
        checkpoint_dir=tmp_path / "checkpoints",
        seed=5,
    )

    result = run_convnext_training(config)
    export = export_checkpoint_weights(
        checkpoint_path=result.final_checkpoint_path,
        architecture="convnext_policy_value",
        output_dir=tmp_path / "exports",
    )

    assert export.weights_path.exists()
    assert export.manifest_path.exists()
    assert export.bundle_path.exists()
    with zipfile.ZipFile(export.bundle_path) as bundle:
        names = set(bundle.namelist())

    assert export.weights_path.name in names
    assert export.manifest_path.name in names
