from __future__ import annotations

import pytest

from go_engine.board import Board
from go_engine.game import GameState, Move
from go_engine.types import Player, Point


def test_black_captures_one_white_stone() -> None:
    board = Board.from_grid(
        5,
        [
            (Point(2, 2), Player.WHITE),
            (Point(1, 2), Player.BLACK),
            (Point(2, 1), Player.BLACK),
            (Point(2, 3), Player.BLACK),
        ],
    )
    game = GameState(board=board, next_player=Player.BLACK)

    next_state = game.apply_move(Move.play(Point(3, 2)))

    assert next_state.board.get(Point(2, 2)) is None
    assert next_state.board.get(Point(3, 2)) is Player.BLACK


def test_black_captures_a_white_group() -> None:
    board = Board.from_grid(
        5,
        [
            (Point(2, 2), Player.WHITE),
            (Point(2, 3), Player.WHITE),
            (Point(1, 2), Player.BLACK),
            (Point(1, 3), Player.BLACK),
            (Point(2, 1), Player.BLACK),
            (Point(2, 4), Player.BLACK),
            (Point(3, 2), Player.BLACK),
        ],
    )
    game = GameState(board=board, next_player=Player.BLACK)

    next_state = game.apply_move(Move.play(Point(3, 3)))

    assert next_state.board.get(Point(2, 2)) is None
    assert next_state.board.get(Point(2, 3)) is None


def test_suicide_move_is_invalid() -> None:
    board = Board.from_grid(
        5,
        [
            (Point(1, 2), Player.WHITE),
            (Point(2, 1), Player.WHITE),
            (Point(2, 3), Player.WHITE),
            (Point(3, 2), Player.WHITE),
        ],
    )
    game = GameState(board=board, next_player=Player.BLACK)

    assert not game.is_valid_move(Move.play(Point(2, 2)))
    with pytest.raises(ValueError):
        game.apply_move(Move.play(Point(2, 2)))


def test_capture_that_fills_last_liberty_is_not_suicide() -> None:
    board = Board.from_grid(
        5,
        [
            (Point(1, 2), Player.WHITE),
            (Point(2, 1), Player.WHITE),
            (Point(2, 3), Player.WHITE),
            (Point(3, 2), Player.WHITE),
            (Point(1, 1), Player.BLACK),
            (Point(1, 3), Player.BLACK),
            (Point(2, 4), Player.BLACK),
            (Point(3, 1), Player.BLACK),
            (Point(3, 3), Player.BLACK),
        ],
    )
    game = GameState(board=board, next_player=Player.BLACK)

    assert game.is_valid_move(Move.play(Point(2, 2)))


def test_ko_prevents_immediate_recapture() -> None:
    board = Board.from_grid(
        5,
        [
            (Point(1, 2), Player.BLACK),
            (Point(2, 1), Player.BLACK),
            (Point(2, 3), Player.BLACK),
            (Point(3, 2), Player.BLACK),
            (Point(1, 3), Player.WHITE),
            (Point(2, 4), Player.WHITE),
            (Point(3, 3), Player.WHITE),
        ],
    )
    game = GameState(board=board, next_player=Player.WHITE)

    after_capture = game.apply_move(Move.play(Point(2, 2)))

    assert after_capture.board.get(Point(2, 3)) is None
    assert not after_capture.is_valid_move(Move.play(Point(2, 3)))


def test_board_position_hash_is_stable_and_order_independent() -> None:
    stones = [
        (Point(1, 1), Player.BLACK),
        (Point(2, 2), Player.WHITE),
    ]
    board_a = Board.from_grid(3, stones)
    board_b = Board.from_grid(3, reversed(stones))

    assert board_a.position_hash() == board_b.position_hash()
    assert board_a.snapshot_key() == board_b.snapshot_key()


def test_board_position_hash_changes_after_move() -> None:
    board = Board(size=3)
    next_board = board.place_stone(Player.BLACK, Point(1, 1))

    assert board.position_hash() != next_board.position_hash()


def test_two_passes_end_the_game() -> None:
    game = GameState.new_game(board_size=5)

    game = game.apply_move(Move.pass_turn())
    assert not game.is_over()

    game = game.apply_move(Move.pass_turn())
    assert game.is_over()
    assert game.legal_moves() == ()


def test_area_scoring_counts_stones_and_owned_empty_regions() -> None:
    board = Board.from_grid(
        5,
        [
            (Point(2, 3), Player.BLACK),
            (Point(3, 2), Player.BLACK),
            (Point(3, 4), Player.BLACK),
            (Point(4, 3), Player.BLACK),
            (Point(5, 5), Player.WHITE),
        ],
    )
    game = GameState(board=board, next_player=Player.BLACK)

    score = game.score(komi=0)

    assert score.black == 5
    assert score.white == 1


def test_winner_rejects_tied_score() -> None:
    game = GameState.new_game(board_size=3)

    with pytest.raises(ValueError, match="tied"):
        game.winner(komi=0)


def test_random_bots_can_finish_a_small_game() -> None:
    from search_players.random_bot import RandomBot

    game = GameState.new_game(board_size=3)
    bots = {
        Player.BLACK: RandomBot(),
        Player.WHITE: RandomBot(),
    }

    for _ in range(200):
        if game.is_over():
            break
        game = game.apply_move(bots[game.next_player].select_move(game))

    assert game.is_over()
