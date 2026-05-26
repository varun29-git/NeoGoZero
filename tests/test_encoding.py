from __future__ import annotations

import pytest

from myalphago.go.board import Board
from myalphago.go.game import GameState, Move
from myalphago.go.types import Player, Point
from myalphago.training.encoding import (
    encode_board_history,
    encode_game_state,
    encode_policy,
    index_to_move,
    move_to_index,
)


def test_encode_game_state_uses_current_player_perspective() -> None:
    game = GameState(
        board=Board.from_grid(
            3,
            [
                (Point(1, 1), Player.BLACK),
                (Point(2, 2), Player.WHITE),
            ],
        ),
        next_player=Player.WHITE,
    )

    current_stones, opponent_stones, next_player_plane = encode_game_state(game)

    assert current_stones[1][1] == 1
    assert opponent_stones[0][0] == 1
    assert next_player_plane == (
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
    )


def test_move_policy_index_round_trip() -> None:
    board_size = 3

    for index in range(board_size * board_size + 1):
        move = index_to_move(index, board_size)
        assert move_to_index(move, board_size) == index


def test_encode_policy_includes_pass_move() -> None:
    policy = encode_policy(
        {
            Move.play(Point(1, 1)): 0.25,
            Move.pass_turn(): 0.75,
        },
        board_size=3,
    )

    assert len(policy) == 10
    assert policy[0] == 0.25
    assert policy[-1] == 0.75


def test_invalid_policy_index_is_rejected() -> None:
    with pytest.raises(ValueError):
        index_to_move(10, board_size=3)


def test_encode_game_state_supports_history_planes() -> None:
    game = GameState.new_game(board_size=3)
    game = game.apply_move(Move.play(Point(1, 1)))
    game = game.apply_move(Move.play(Point(2, 2)))

    planes = encode_game_state(game, history_length=2)

    assert len(planes) == 5
    assert planes[0][1][1] == 1
    assert planes[3][0][0] == 1


def test_encode_board_history_pads_missing_positions() -> None:
    history = (((1, 1, Player.BLACK.value),),)

    planes = encode_board_history(
        board_history=history,
        player=Player.BLACK,
        board_size=3,
        history_length=2,
    )

    assert len(planes) == 5
    assert planes[0][0][0] == 1
    assert planes[2] == (
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
    )
