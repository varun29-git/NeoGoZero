from __future__ import annotations

import random
import re
from pathlib import Path

import torch

from go_engine.game import GameState, Move
from go_engine.types import Player, Point
from zero_training_pipeline.self_play import BoardSnapshot, TrainingExample
from zero_training_pipeline.torch_training import train_step


_PROPERTY_RE = re.compile(r"([A-Za-z]+)((?:\[(?:\\.|[^\]])*\])+)") 
_VALUE_RE = re.compile(r"\[((?:\\.|[^\]])*)\]")


def load_sgf_training_examples(
    sgf_dir: Path,
    board_size: int,
    history_length: int,
    max_examples: int | None = None,
) -> tuple[TrainingExample, ...]:
    sgf_paths = sorted(Path(sgf_dir).rglob("*.sgf"))
    if not sgf_paths:
        raise ValueError(f"no SGF files found in {sgf_dir}")

    examples: list[TrainingExample] = []
    for sgf_path in sgf_paths:
        examples.extend(
            parse_sgf_training_examples(
                sgf_path.read_text(encoding="utf-8", errors="ignore"),
                board_size=board_size,
                history_length=history_length,
            )
        )
        if max_examples is not None and len(examples) >= max_examples:
            return tuple(examples[:max_examples])

    if not examples:
        raise ValueError(f"no usable {board_size}x{board_size} examples found in {sgf_dir}")
    return tuple(examples)


def parse_sgf_training_examples(
    sgf_text: str,
    board_size: int,
    history_length: int,
) -> tuple[TrainingExample, ...]:
    nodes = _parse_sgf_nodes(sgf_text)
    if not nodes:
        return ()

    root = nodes[0]
    sgf_board_size = int(root.get("SZ", [str(board_size)])[0])
    if sgf_board_size != board_size:
        return ()

    game = GameState.new_game(board_size=board_size)
    pending: list[tuple[BoardSnapshot, Player, Move, tuple[BoardSnapshot, ...]]] = []
    result_winner = _winner_from_result(root.get("RE", [""])[0])

    for node in nodes[1:]:
        move_player, move = _move_from_node(node, board_size)
        if move_player is None or move is None:
            continue
        if move_player is not game.next_player:
            return ()
        if not game.is_valid_move(move):
            return ()

        pending.append(
            (
                game.board.zobrist_key(),
                game.next_player,
                move,
                _snapshot_history(game, history_length),
            )
        )
        game = game.apply_move(move)

    if not pending:
        return ()

    winner = result_winner or game.winner()
    return tuple(
        TrainingExample(
            board=board,
            player=player,
            visit_distribution={move: 1.0},
            winner=winner,
            board_history=board_history,
        )
        for board, player, move, board_history in pending
    )


def run_supervised_pretraining(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    examples: tuple[TrainingExample, ...],
    board_size: int,
    history_length: int,
    steps: int,
    batch_size: int,
    device: torch.device | str,
    rng: random.Random,
) -> list[float]:
    if steps < 1:
        return []
    if not examples:
        raise ValueError("supervised pretraining needs at least one example")

    losses = []
    for _ in range(steps):
        batch = _sample_examples(examples, batch_size, rng)
        losses.append(
            train_step(
                model=model,  # type: ignore[arg-type]
                optimizer=optimizer,
                examples=batch,
                board_size=board_size,
                history_length=history_length,
                device=device,
            )
        )
    return losses


def _sample_examples(
    examples: tuple[TrainingExample, ...],
    batch_size: int,
    rng: random.Random,
) -> list[TrainingExample]:
    if batch_size <= len(examples):
        return rng.sample(list(examples), batch_size)
    return [rng.choice(examples) for _ in range(batch_size)]


def _parse_sgf_nodes(sgf_text: str) -> list[dict[str, list[str]]]:
    nodes: list[dict[str, list[str]]] = []
    for raw_node in sgf_text.split(";")[1:]:
        properties: dict[str, list[str]] = {}
        for key, raw_values in _PROPERTY_RE.findall(raw_node):
            properties[key] = [_unescape_sgf_value(value) for value in _VALUE_RE.findall(raw_values)]
        if properties:
            nodes.append(properties)
    return nodes


def _unescape_sgf_value(value: str) -> str:
    return value.replace(r"\]", "]").replace(r"\\", "\\").strip()


def _winner_from_result(result: str) -> Player | None:
    normalized = result.strip().upper()
    if normalized.startswith("B+"):
        return Player.BLACK
    if normalized.startswith("W+"):
        return Player.WHITE
    return None


def _move_from_node(
    node: dict[str, list[str]],
    board_size: int,
) -> tuple[Player | None, Move | None]:
    if "B" in node:
        return Player.BLACK, _move_from_sgf_coord(node["B"][0], board_size)
    if "W" in node:
        return Player.WHITE, _move_from_sgf_coord(node["W"][0], board_size)
    return None, None


def _move_from_sgf_coord(coord: str, board_size: int) -> Move:
    if coord == "" or coord.lower() == "tt":
        return Move.pass_turn()
    if len(coord) != 2:
        raise ValueError(f"invalid SGF coordinate: {coord!r}")

    col = ord(coord[0].lower()) - ord("a") + 1
    row = ord(coord[1].lower()) - ord("a") + 1
    if not (1 <= row <= board_size and 1 <= col <= board_size):
        raise ValueError(f"SGF coordinate is outside {board_size}x{board_size}: {coord!r}")
    return Move.play(Point(row=row, col=col))


def _snapshot_history(
    game_state: GameState,
    history_length: int,
) -> tuple[BoardSnapshot, ...]:
    history = []
    state: GameState | None = game_state
    while state is not None and len(history) < history_length:
        history.append(state.board.zobrist_key())
        state = state.previous_state
    return tuple(history)
