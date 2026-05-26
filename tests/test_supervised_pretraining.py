from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from go_engine.game import Move
from go_engine.types import Player, Point
from policy_value_networks.resnet_policy_value.policy_value import PolicyValueNet
from zero_training_pipeline.supervised_pretraining import (
    load_sgf_training_examples,
    parse_sgf_training_examples,
    run_supervised_pretraining,
)


def test_parse_sgf_training_examples_uses_expert_moves_and_winner() -> None:
    sgf = "(;GM[1]FF[4]SZ[9]RE[B+R];B[aa];W[bb];B[cc])"

    examples = parse_sgf_training_examples(
        sgf,
        board_size=9,
        history_length=2,
    )

    assert len(examples) == 3
    assert examples[0].player is Player.BLACK
    assert examples[0].winner is Player.BLACK
    assert examples[0].visit_distribution == {Move.play(Point(1, 1)): 1.0}
    assert examples[1].visit_distribution == {Move.play(Point(2, 2)): 1.0}
    assert len(examples[2].board_history) == 2


def test_load_sgf_training_examples_from_directory(tmp_path) -> None:
    (tmp_path / "game.sgf").write_text(
        "(;GM[1]FF[4]SZ[9]RE[W+2.5];B[aa];W[bb])",
        encoding="utf-8",
    )

    examples = load_sgf_training_examples(
        tmp_path,
        board_size=9,
        history_length=1,
    )

    assert len(examples) == 2
    assert all(example.winner is Player.WHITE for example in examples)


def test_run_supervised_pretraining_updates_model() -> None:
    examples = parse_sgf_training_examples(
        "(;GM[1]FF[4]SZ[3]RE[B+R];B[aa];W[bb];B[cc])",
        board_size=3,
        history_length=1,
    )
    model = PolicyValueNet(board_size=3, input_planes=3, channels=8, num_res_blocks=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = run_supervised_pretraining(
        model=model,
        optimizer=optimizer,
        examples=examples,
        board_size=3,
        history_length=1,
        steps=2,
        batch_size=2,
        device="cpu",
        rng=random.Random(1),
    )

    assert len(losses) == 2
    assert all(loss > 0 for loss in losses)
