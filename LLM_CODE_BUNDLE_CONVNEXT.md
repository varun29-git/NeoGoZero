# NeoGoZero ConvNeXt Code Bundle

This file is a direct code bundle for LLM review. Each section starts with the source file path, followed by the full file contents.

## pyproject.toml

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "NeoGoZero"
version = "0.1.0"
description = "A sequential AlphaGo Zero-style project starting with a Go rules engine."
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8"]
ml = ["torch>=2.2"]

[tool.pytest.ini_options]
testpaths = ["tests", "policy_value_networks/convnext_policy_value"]
pythonpath = ["."]
```

## go_engine/__init__.py

```python
"""Core Go game logic."""

from go_engine.board import Board
from go_engine.game import GameState, Move
from go_engine.types import Player, Point

__all__ = ["Board", "GameState", "Move", "Player", "Point"]
```

## go_engine/types.py

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Player(Enum):
    BLACK = "black"
    WHITE = "white"

    @property
    def other(self) -> Player:
        return Player.WHITE if self is Player.BLACK else Player.BLACK

    @property
    def marker(self) -> str:
        return "X" if self is Player.BLACK else "O"


@dataclass(frozen=True, order=True)
class Point:
    row: int
    col: int

    def neighbors(self, size: int) -> tuple[Point, ...]:
        candidates = (
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        )
        return tuple(
            point
            for point in candidates
            if 1 <= point.row <= size and 1 <= point.col <= size
        )
```

## go_engine/board.py

```python
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from go_engine.types import Player, Point


@dataclass(frozen=True)
class Board:
    size: int = 9
    grid: frozenset[tuple[Point, Player]] = frozenset()
    _stones_by_point: dict[Point, Player] = field(init=False, repr=False, compare=False)
    _position_hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("board size must be at least 2")
        stones_by_point = dict(self.grid)
        if len(stones_by_point) != len(self.grid):
            raise ValueError("a point cannot contain more than one stone")
        for point, _ in self.grid:
            if not self.is_on_grid(point):
                raise ValueError(f"point is outside the board: {point}")
        object.__setattr__(self, "_stones_by_point", stones_by_point)
        object.__setattr__(
            self,
            "_position_hash",
            _position_hash(self.size, self.grid),
        )

    @classmethod
    def from_grid(
        cls, size: int, stones: Iterable[tuple[Point, Player]]
    ) -> Board:
        return cls(size=size, grid=frozenset(stones))

    def is_on_grid(self, point: Point) -> bool:
        return 1 <= point.row <= self.size and 1 <= point.col <= self.size

    def get(self, point: Point) -> Player | None:
        return self._stones_by_point.get(point)

    def is_empty(self, point: Point) -> bool:
        return self.get(point) is None

    def place_stone(self, player: Player, point: Point) -> Board:
        if not self.is_on_grid(point):
            raise ValueError(f"point is outside the board: {point}")
        if not self.is_empty(point):
            raise ValueError(f"point is already occupied: {point}")

        stones = dict(self._stones_by_point)
        stones[point] = player

        board_after_placement = Board.from_grid(self.size, stones.items())
        captured: set[Point] = set()
        for neighbor in point.neighbors(self.size):
            if board_after_placement.get(neighbor) is player.other:
                group = board_after_placement.group_at(neighbor)
                if not board_after_placement.liberties(group):
                    captured.update(group)

        for captured_point in captured:
            stones.pop(captured_point)

        return Board.from_grid(self.size, stones.items())

    def group_at(self, point: Point) -> frozenset[Point]:
        player = self.get(point)
        if player is None:
            raise ValueError(f"empty point has no group: {point}")

        group: set[Point] = set()
        frontier = [point]
        while frontier:
            current = frontier.pop()
            if current in group:
                continue
            group.add(current)
            for neighbor in current.neighbors(self.size):
                if self.get(neighbor) is player and neighbor not in group:
                    frontier.append(neighbor)
        return frozenset(group)

    def liberties(self, group: Iterable[Point]) -> frozenset[Point]:
        liberties: set[Point] = set()
        for point in group:
            for neighbor in point.neighbors(self.size):
                if self.is_empty(neighbor):
                    liberties.add(neighbor)
        return frozenset(liberties)

    def has_liberties(self, point: Point) -> bool:
        return bool(self.liberties(self.group_at(point)))

    def stone_counts(self) -> dict[Player, int]:
        counts = {Player.BLACK: 0, Player.WHITE: 0}
        for _, player in self.grid:
            counts[player] += 1
        return counts

    def empty_points(self) -> tuple[Point, ...]:
        return tuple(
            Point(row, col)
            for row in range(1, self.size + 1)
            for col in range(1, self.size + 1)
            if self.is_empty(Point(row, col))
        )

    def snapshot_key(self) -> tuple[tuple[int, int, str], ...]:
        return tuple(
            sorted((point.row, point.col, player.value) for point, player in self.grid)
        )

    def zobrist_key(self) -> tuple[tuple[int, int, str], ...]:
        return self.snapshot_key()

    def position_hash(self) -> int:
        return self._position_hash

    def __str__(self) -> str:
        rows = []
        for row in range(1, self.size + 1):
            cells = []
            for col in range(1, self.size + 1):
                player = self.get(Point(row, col))
                cells.append(player.marker if player else ".")
            rows.append(" ".join(cells))
        return "\n".join(rows)


def _position_hash(
    size: int,
    grid: frozenset[tuple[Point, Player]],
) -> int:
    value = 0
    for point, player in grid:
        value ^= _stone_hash(size, point, player)
    return value


def _stone_hash(size: int, point: Point, player: Player) -> int:
    color_id = 1 if player is Player.BLACK else 2
    key = (
        size * 0x9E3779B185EBCA87
        ^ point.row * 0xC2B2AE3D27D4EB4F
        ^ point.col * 0x165667B19E3779F9
        ^ color_id * 0x85EBCA77C2B2AE63
    ) & ((1 << 64) - 1)
    key ^= key >> 30
    key = (key * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    key ^= key >> 27
    key = (key * 0x94D049BB133111EB) & ((1 << 64) - 1)
    key ^= key >> 31
    return key
```

## go_engine/game.py

```python
from __future__ import annotations

from dataclasses import dataclass

from go_engine.board import Board
from go_engine.types import Player, Point


@dataclass(frozen=True)
class Move:
    point: Point | None = None
    is_pass: bool = False

    @classmethod
    def play(cls, point: Point) -> Move:
        return cls(point=point)

    @classmethod
    def pass_turn(cls) -> Move:
        return cls(is_pass=True)

    def __post_init__(self) -> None:
        if self.is_pass and self.point is not None:
            raise ValueError("pass moves cannot have a point")
        if not self.is_pass and self.point is None:
            raise ValueError("play moves must have a point")


@dataclass(frozen=True)
class GameState:
    board: Board
    next_player: Player
    previous_state: GameState | None = None
    last_move: Move | None = None

    @classmethod
    def new_game(cls, board_size: int = 9) -> GameState:
        return cls(board=Board(size=board_size), next_player=Player.BLACK)

    def apply_move(self, move: Move) -> GameState:
        if not self.is_valid_move(move):
            raise ValueError(f"illegal move: {move}")

        if move.is_pass:
            next_board = self.board
        else:
            assert move.point is not None
            next_board = self.board.place_stone(self.next_player, move.point)

        return GameState(
            board=next_board,
            next_player=self.next_player.other,
            previous_state=self,
            last_move=move,
        )

    def is_over(self) -> bool:
        if self.last_move is None or self.previous_state is None:
            return False
        return self.last_move.is_pass and self.previous_state.last_move is not None and (
            self.previous_state.last_move.is_pass
        )

    def legal_moves(self) -> tuple[Move, ...]:
        if self.is_over():
            return ()

        moves = [
            Move.play(point)
            for point in self.board.empty_points()
            if self.is_valid_move(Move.play(point))
        ]
        moves.append(Move.pass_turn())
        return tuple(moves)

    def is_valid_move(self, move: Move) -> bool:
        if self.is_over():
            return False
        if move.is_pass:
            return True
        assert move.point is not None
        if not self.board.is_on_grid(move.point) or not self.board.is_empty(move.point):
            return False

        next_board = self.board.place_stone(self.next_player, move.point)
        placed_player = next_board.get(move.point)
        if placed_player is not self.next_player:
            return False
        if not next_board.has_liberties(move.point):
            return False
        if self._violates_ko(next_board):
            return False
        return True

    def winner(self, komi: float = 7.5) -> Player:
        result = self.score(komi=komi)
        if result.black == result.white:
            raise ValueError("game is tied")
        return Player.BLACK if result.black > result.white else Player.WHITE

    def score(self, komi: float = 7.5) -> Score:
        black = 0.0
        white = komi
        visited: set[Point] = set()

        for point, player in self.board.grid:
            if player is Player.BLACK:
                black += 1
            else:
                white += 1
            visited.add(point)

        for point in self.board.empty_points():
            if point in visited:
                continue
            region, bordering_players = self._collect_empty_region(point)
            visited.update(region)
            if len(bordering_players) == 1:
                owner = next(iter(bordering_players))
                if owner is Player.BLACK:
                    black += len(region)
                else:
                    white += len(region)

        return Score(black=black, white=white)

    def _collect_empty_region(
        self, start: Point
    ) -> tuple[frozenset[Point], frozenset[Player]]:
        region: set[Point] = set()
        bordering_players: set[Player] = set()
        frontier = [start]

        while frontier:
            point = frontier.pop()
            if point in region:
                continue
            region.add(point)

            for neighbor in point.neighbors(self.board.size):
                player = self.board.get(neighbor)
                if player is None:
                    if neighbor not in region:
                        frontier.append(neighbor)
                else:
                    bordering_players.add(player)

        return frozenset(region), frozenset(bordering_players)

    def _violates_ko(self, next_board: Board) -> bool:
        next_situation = (self.next_player.other, next_board.position_hash())
        state = self.previous_state
        while state is not None:
            if (state.next_player, state.board.position_hash()) == next_situation:
                return True
            state = state.previous_state
        return False


@dataclass(frozen=True)
class Score:
    black: float
    white: float

    @property
    def margin(self) -> float:
        return abs(self.black - self.white)
```

## search_players/__init__.py

```python
"""Bot implementations."""

from search_players.mcts_bot import MCTSBot
from search_players.random_bot import RandomBot

__all__ = ["MCTSBot", "RandomBot"]
```

## search_players/random_bot.py

```python
from __future__ import annotations

import random
from dataclasses import dataclass, field

from go_engine.game import GameState, Move


@dataclass
class RandomBot:
    rng: random.Random = field(default_factory=random.Random)

    def select_move(self, game_state: GameState) -> Move:
        return self.rng.choice(game_state.legal_moves())
```

## search_players/mcts_bot.py

```python
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Protocol

from go_engine.game import GameState, Move
from go_engine.types import Player


@dataclass(frozen=True)
class Evaluation:
    move_priors: dict[Move, float]
    value: float

    @classmethod
    def from_priors(
        cls,
        game_state: GameState,
        move_priors: dict[Move, float],
        value: float,
    ) -> Evaluation:
        legal_moves = set(game_state.legal_moves())
        legal_priors = {
            move: max(prior, 0.0)
            for move, prior in move_priors.items()
            if move in legal_moves
        }
        total = sum(legal_priors.values())
        if total == 0:
            return cls.uniform(game_state, value=value)

        return cls(
            move_priors={
                move: prior / total for move, prior in legal_priors.items()
            },
            value=max(-1.0, min(1.0, value)),
        )

    @classmethod
    def uniform(cls, game_state: GameState, value: float) -> Evaluation:
        legal_moves = game_state.legal_moves()
        if not legal_moves:
            return cls(move_priors={}, value=max(-1.0, min(1.0, value)))

        prior = 1 / len(legal_moves)
        return cls(
            move_priors={move: prior for move in legal_moves},
            value=max(-1.0, min(1.0, value)),
        )


class Evaluator(Protocol):
    def evaluate(self, game_state: GameState) -> Evaluation:
        """Return move priors and value from the next player's perspective."""


class BatchEvaluator(Evaluator, Protocol):
    def evaluate_many(self, game_states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        """Return evaluations for multiple game states in one model call."""


@dataclass
class RandomRolloutEvaluator:
    rng: random.Random = field(default_factory=random.Random)
    max_rollout_moves: int | None = None

    def evaluate(self, game_state: GameState) -> Evaluation:
        return Evaluation.uniform(
            game_state,
            value=self._rollout_value(game_state),
        )

    def _rollout_value(self, game_state: GameState) -> float:
        player = game_state.next_player
        rollout_state = game_state
        max_moves = self.max_rollout_moves
        if max_moves is None:
            max_moves = game_state.board.size * game_state.board.size * 2

        for _ in range(max_moves):
            if rollout_state.is_over():
                break
            legal_moves = rollout_state.legal_moves()
            if not legal_moves:
                break
            move = self.rng.choice(legal_moves)
            rollout_state = rollout_state.apply_move(move)

        return 1.0 if rollout_state.winner() is player else -1.0


@dataclass(frozen=True)
class SearchResult:
    selected_move: Move
    visit_counts: dict[Move, int]

    def visit_distribution(self) -> dict[Move, float]:
        total_visits = sum(self.visit_counts.values())
        if total_visits == 0:
            if not self.visit_counts:
                return {}
            probability = 1 / len(self.visit_counts)
            return {move: probability for move in self.visit_counts}
        return {
            move: visits / total_visits
            for move, visits in self.visit_counts.items()
        }


@dataclass
class MCTSNode:
    game_state: GameState
    parent: MCTSNode | None = None
    move: Move | None = None
    prior_probability: float = 1.0
    value_sum: float = 0.0
    num_rollouts: int = 0
    virtual_visits: int = 0
    children: list[MCTSNode] = field(default_factory=list)

    def is_expanded(self) -> bool:
        return bool(self.children)

    def expand(self, move_priors: dict[Move, float]) -> None:
        if self.children or self.game_state.is_over():
            return
        for move, prior_probability in move_priors.items():
            self.children.append(
                MCTSNode(
                    game_state=self.game_state.apply_move(move),
                    parent=self,
                    move=move,
                    prior_probability=prior_probability,
                )
            )

    def record_visit(self, value: float) -> None:
        self.value_sum += value
        self.num_rollouts += 1

    def q_value(self) -> float:
        if self.num_rollouts == 0:
            return 0.0
        return self.value_sum / self.num_rollouts

    def select_child(self, c_puct: float) -> MCTSNode:
        if not self.children:
            raise ValueError("cannot select a child from a leaf node")

        parent_visits = math.sqrt(max(self.num_rollouts + self.virtual_visits, 1))

        def puct_score(child: MCTSNode) -> float:
            value_score = -child.q_value()
            child_visits = child.num_rollouts + child.virtual_visits
            prior_score = (
                c_puct
                * child.prior_probability
                * parent_visits
                / (1 + child_visits)
            )
            return value_score + prior_score

        return max(self.children, key=puct_score)

    def most_visited_child(self) -> MCTSNode:
        if not self.children:
            raise ValueError("cannot choose a move from a leaf node")
        return max(self.children, key=lambda child: child.num_rollouts)

    def add_dirichlet_noise(
        self,
        alpha: float,
        epsilon: float,
        rng: random.Random,
    ) -> None:
        if not self.children:
            return
        samples = [rng.gammavariate(alpha, 1.0) for _ in self.children]
        total = sum(samples)
        if total == 0:
            samples = [1.0 for _ in self.children]
            total = len(samples)

        for child, sample in zip(self.children, samples):
            noise = sample / total
            child.prior_probability = (
                (1 - epsilon) * child.prior_probability + epsilon * noise
            )


@dataclass
class MCTSBot:
    num_rounds: int = 100
    c_puct: float = 1.5
    max_rollout_moves: int | None = None
    evaluator: Evaluator | None = None
    inference_batch_size: int = 1
    dirichlet_alpha: float = 0.03
    dirichlet_epsilon: float = 0.25
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.num_rounds < 1:
            raise ValueError("num_rounds must be at least 1")
        if self.c_puct < 0:
            raise ValueError("c_puct cannot be negative")
        if self.max_rollout_moves is not None and self.max_rollout_moves < 1:
            raise ValueError("max_rollout_moves must be at least 1")
        if self.inference_batch_size < 1:
            raise ValueError("inference_batch_size must be at least 1")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError("dirichlet_epsilon must be in [0, 1]")
        if self.evaluator is None:
            self.evaluator = RandomRolloutEvaluator(
                rng=self.rng,
                max_rollout_moves=self.max_rollout_moves,
            )

    def select_move(self, game_state: GameState) -> Move:
        return self.search(game_state, temperature=0.0).selected_move

    def search(
        self,
        game_state: GameState,
        temperature: float = 0.0,
        add_dirichlet_noise: bool = False,
    ) -> SearchResult:
        if temperature < 0:
            raise ValueError("temperature cannot be negative")

        legal_moves = game_state.legal_moves()
        if not legal_moves:
            raise ValueError("cannot select a move after the game is over")
        if len(legal_moves) == 1:
            return SearchResult(
                selected_move=legal_moves[0],
                visit_counts={legal_moves[0]: 1},
            )

        root = MCTSNode(game_state)
        added_root_noise = False
        remaining_rounds = self.num_rounds

        if not root.is_expanded():
            assert self.evaluator is not None
            evaluation = self.evaluator.evaluate(root.game_state)
            root.expand(evaluation.move_priors)
            if add_dirichlet_noise and root.children:
                root.add_dirichlet_noise(
                    alpha=self.dirichlet_alpha,
                    epsilon=self.dirichlet_epsilon,
                    rng=self.rng,
                )
                added_root_noise = True
            _backup_value(root, evaluation.value)
            remaining_rounds -= 1

        while remaining_rounds > 0:
            batch_size = min(self.inference_batch_size, remaining_rounds)
            leaves_and_paths = [
                _select_leaf_with_virtual_visits(root, self.c_puct)
                for _ in range(batch_size)
            ]
            leaves = [leaf for leaf, _ in leaves_and_paths]
            assert self.evaluator is not None
            evaluations = _evaluate_many(self.evaluator, [leaf.game_state for leaf in leaves])

            for (leaf, path), evaluation in zip(leaves_and_paths, evaluations):
                _release_virtual_visits(path)
                leaf.expand(evaluation.move_priors)
                if (
                    add_dirichlet_noise
                    and leaf is root
                    and not added_root_noise
                    and root.children
                ):
                    root.add_dirichlet_noise(
                        alpha=self.dirichlet_alpha,
                        epsilon=self.dirichlet_epsilon,
                        rng=self.rng,
                    )
                    added_root_noise = True
                _backup_value(leaf, evaluation.value)

            remaining_rounds -= batch_size

        best_child = root.most_visited_child()
        assert best_child.move is not None
        visit_counts = {
            child.move: child.num_rollouts
            for child in root.children
            if child.move is not None
        }
        selected_move = _select_move_from_visit_counts(
            visit_counts=visit_counts,
            temperature=temperature,
            rng=self.rng,
        )
        return SearchResult(
            selected_move=selected_move,
            visit_counts=visit_counts,
        )


def _select_leaf_with_virtual_visits(
    root: MCTSNode,
    c_puct: float,
) -> tuple[MCTSNode, tuple[MCTSNode, ...]]:
    node = root
    path = [node]

    while node.is_expanded() and not node.game_state.is_over():
        node = node.select_child(c_puct)
        path.append(node)

    for path_node in path:
        path_node.virtual_visits += 1
    return node, tuple(path)


def _release_virtual_visits(path: tuple[MCTSNode, ...]) -> None:
    for node in path:
        node.virtual_visits = max(node.virtual_visits - 1, 0)


def _backup_value(node: MCTSNode, value: float) -> None:
    while node is not None:
        node.record_visit(value)
        value = -value
        node = node.parent


def _evaluate_many(
    evaluator: Evaluator,
    game_states: Sequence[GameState],
) -> tuple[Evaluation, ...]:
    evaluate_many = getattr(evaluator, "evaluate_many", None)
    if callable(evaluate_many):
        return tuple(evaluate_many(game_states))
    return tuple(evaluator.evaluate(game_state) for game_state in game_states)


def _select_move_from_visit_counts(
    visit_counts: dict[Move, int],
    temperature: float,
    rng: random.Random,
) -> Move:
    if not visit_counts:
        raise ValueError("cannot select from empty visit counts")
    if temperature == 0:
        return max(visit_counts, key=visit_counts.get)

    weighted_moves = []
    total = 0.0
    exponent = 1 / temperature
    for move, visits in visit_counts.items():
        weight = visits**exponent
        weighted_moves.append((move, weight))
        total += weight

    if total == 0:
        return rng.choice(tuple(visit_counts))

    threshold = rng.random() * total
    cumulative = 0.0
    for move, weight in weighted_moves:
        cumulative += weight
        if cumulative >= threshold:
            return move

    return weighted_moves[-1][0]
```

## match_evaluation/__init__.py

```python
"""Bot evaluation and match helpers."""

from match_evaluation.match import GameResult, MatchResult, play_game, play_match

__all__ = ["GameResult", "MatchResult", "play_game", "play_match"]
```

## match_evaluation/match.py

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from go_engine.game import GameState, Move, Score
from go_engine.types import Player


class Bot(Protocol):
    def select_move(self, game_state: GameState) -> Move:
        """Choose one legal move for the current game state."""


@dataclass(frozen=True)
class GameResult:
    final_state: GameState
    moves: tuple[Move, ...]
    score: Score
    winner: Player
    reached_move_limit: bool = False

    @property
    def num_moves(self) -> int:
        return len(self.moves)


@dataclass(frozen=True)
class MatchResult:
    games: tuple[GameResult, ...]

    @property
    def black_wins(self) -> int:
        return sum(1 for game in self.games if game.winner is Player.BLACK)

    @property
    def white_wins(self) -> int:
        return sum(1 for game in self.games if game.winner is Player.WHITE)

    @property
    def average_moves(self) -> float:
        if not self.games:
            return 0.0
        return sum(game.num_moves for game in self.games) / len(self.games)

    @property
    def average_margin(self) -> float:
        if not self.games:
            return 0.0
        return sum(game.score.margin for game in self.games) / len(self.games)

    def win_rate(self, player: Player) -> float:
        if not self.games:
            return 0.0
        wins = self.black_wins if player is Player.BLACK else self.white_wins
        return wins / len(self.games)


BotFactory = Callable[[int], Bot]


def play_game(
    black_bot: Bot,
    white_bot: Bot,
    board_size: int = 9,
    komi: float = 7.5,
    max_moves: int | None = None,
) -> GameResult:
    game = GameState.new_game(board_size=board_size)
    bots = {
        Player.BLACK: black_bot,
        Player.WHITE: white_bot,
    }
    moves: list[Move] = []
    move_limit = max_moves if max_moves is not None else board_size * board_size * 3
    reached_move_limit = False

    for _ in range(move_limit):
        if game.is_over():
            break

        move = bots[game.next_player].select_move(game)
        if not game.is_valid_move(move):
            raise ValueError(f"{game.next_player.value} bot selected illegal move: {move}")

        moves.append(move)
        game = game.apply_move(move)
    else:
        reached_move_limit = not game.is_over()

    score = game.score(komi=komi)
    winner = Player.BLACK if score.black > score.white else Player.WHITE
    return GameResult(
        final_state=game,
        moves=tuple(moves),
        score=score,
        winner=winner,
        reached_move_limit=reached_move_limit,
    )


def play_match(
    black_bot_factory: BotFactory,
    white_bot_factory: BotFactory,
    num_games: int,
    board_size: int = 9,
    komi: float = 7.5,
    max_moves: int | None = None,
) -> MatchResult:
    if num_games < 1:
        raise ValueError("num_games must be at least 1")

    games = tuple(
        play_game(
            black_bot=black_bot_factory(game_index),
            white_bot=white_bot_factory(game_index),
            board_size=board_size,
            komi=komi,
            max_moves=max_moves,
        )
        for game_index in range(num_games)
    )
    return MatchResult(games=games)
```

## zero_training_pipeline/__init__.py

```python
"""Self-play and training data helpers."""

from zero_training_pipeline.encoding import (
    encode_board_snapshot,
    encode_game_state,
    encode_policy,
    index_to_move,
    move_to_index,
)
from zero_training_pipeline.self_play import SelfPlayGame, TrainingExample, generate_self_play_game

__all__ = [
    "SelfPlayGame",
    "TrainingExample",
    "encode_board_snapshot",
    "encode_game_state",
    "encode_policy",
    "generate_self_play_game",
    "index_to_move",
    "move_to_index",
]
```

## zero_training_pipeline/encoding.py

```python
from __future__ import annotations

from go_engine.game import GameState, Move
from go_engine.types import Player, Point

BoardPlanes = tuple[tuple[tuple[int, ...], ...], ...]
PolicyVector = tuple[float, ...]


BoardSnapshot = tuple[tuple[int, int, str], ...]


def encode_game_state(game_state: GameState, history_length: int = 1) -> BoardPlanes:
    history = []
    state: GameState | None = game_state
    while state is not None and len(history) < history_length:
        history.append(state.board.snapshot_key())
        state = state.previous_state

    return encode_board_history(
        board_history=tuple(history),
        player=game_state.next_player,
        board_size=game_state.board.size,
        history_length=history_length,
    )


def encode_board_snapshot(
    board: BoardSnapshot,
    player: Player,
    board_size: int,
    history_length: int = 1,
    board_history: tuple[BoardSnapshot, ...] | None = None,
) -> BoardPlanes:
    history = board_history if board_history is not None else (board,)
    return encode_board_history(
        board_history=history,
        player=player,
        board_size=board_size,
        history_length=history_length,
    )


def encode_board_history(
    board_history: tuple[BoardSnapshot, ...],
    player: Player,
    board_size: int,
    history_length: int = 1,
) -> BoardPlanes:
    if history_length < 1:
        raise ValueError("history_length must be at least 1")

    planes = []
    opponent = player.other
    for index in range(history_length):
        board = board_history[index] if index < len(board_history) else ()
        stones = {(row, col): color for row, col, color in board}
        planes.append(_snapshot_plane(stones, player, board_size))
        planes.append(_snapshot_plane(stones, opponent, board_size))

    planes.append(_filled_plane(board_size, 1 if player is Player.BLACK else 0))
    return tuple(planes)


def encode_policy(
    visit_distribution: dict[Move, float],
    board_size: int,
) -> PolicyVector:
    vector = [0.0] * (board_size * board_size + 1)
    for move, probability in visit_distribution.items():
        vector[move_to_index(move, board_size)] = probability
    return tuple(vector)


def move_to_index(move: Move, board_size: int) -> int:
    if move.is_pass:
        return board_size * board_size
    assert move.point is not None
    if not (1 <= move.point.row <= board_size and 1 <= move.point.col <= board_size):
        raise ValueError(f"move is outside a {board_size}x{board_size} board: {move}")
    return (move.point.row - 1) * board_size + (move.point.col - 1)


def index_to_move(index: int, board_size: int) -> Move:
    pass_index = board_size * board_size
    if index == pass_index:
        return Move.pass_turn()
    if index < 0 or index > pass_index:
        raise ValueError(f"policy index is outside a {board_size}x{board_size} board: {index}")

    row = index // board_size + 1
    col = index % board_size + 1
    return Move.play(Point(row, col))


def _snapshot_plane(
    stones: dict[tuple[int, int], str],
    player: Player,
    board_size: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            1 if stones.get((row, col)) == player.value else 0
            for col in range(1, board_size + 1)
        )
        for row in range(1, board_size + 1)
    )


def _filled_plane(size: int, value: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(value for _ in range(size)) for _ in range(size))
```

## zero_training_pipeline/self_play.py

```python
from __future__ import annotations

from dataclasses import dataclass

from search_players.mcts_bot import MCTSBot
from go_engine.game import GameState, Move
from go_engine.types import Player


BoardSnapshot = tuple[tuple[int, int, str], ...]


@dataclass(frozen=True)
class TrainingExample:
    board: BoardSnapshot
    player: Player
    visit_distribution: dict[Move, float]
    winner: Player
    board_history: tuple[BoardSnapshot, ...] = ()

    @property
    def value(self) -> float:
        return 1.0 if self.winner is self.player else -1.0


@dataclass(frozen=True)
class SelfPlayGame:
    examples: tuple[TrainingExample, ...]
    winner: Player
    moves: tuple[Move, ...]
    final_state: GameState


@dataclass(frozen=True)
class _PendingExample:
    board: BoardSnapshot
    player: Player
    visit_distribution: dict[Move, float]
    board_history: tuple[BoardSnapshot, ...]


def generate_self_play_game(
    bot: MCTSBot,
    board_size: int = 9,
    max_moves: int | None = None,
    history_length: int = 1,
    temperature: float = 1.0,
    temperature_drop_move: int = 30,
    add_dirichlet_noise: bool = True,
) -> SelfPlayGame:
    game = GameState.new_game(board_size=board_size)
    move_limit = max_moves if max_moves is not None else board_size * board_size * 3
    pending_examples: list[_PendingExample] = []
    moves: list[Move] = []

    for _ in range(move_limit):
        if game.is_over():
            break

        move_temperature = temperature if len(moves) < temperature_drop_move else 0.0
        search_result = bot.search(
            game,
            temperature=move_temperature,
            add_dirichlet_noise=add_dirichlet_noise,
        )
        board_history = _snapshot_history(game, history_length)
        pending_examples.append(
            _PendingExample(
                board=_snapshot_board(game),
                player=game.next_player,
                visit_distribution=search_result.visit_distribution(),
                board_history=board_history,
            )
        )
        moves.append(search_result.selected_move)
        game = game.apply_move(search_result.selected_move)

    if not game.is_over():
        game = game.apply_move(Move.pass_turn())
        if not game.is_over():
            game = game.apply_move(Move.pass_turn())

    winner = game.winner()
    examples = tuple(
        TrainingExample(
            board=example.board,
            player=example.player,
            visit_distribution=example.visit_distribution,
            winner=winner,
            board_history=example.board_history,
        )
        for example in pending_examples
    )

    return SelfPlayGame(
        examples=examples,
        winner=winner,
        moves=tuple(moves),
        final_state=game,
    )


def _snapshot_board(game_state: GameState) -> BoardSnapshot:
    return game_state.board.snapshot_key()


def _snapshot_history(
    game_state: GameState,
    history_length: int,
) -> tuple[BoardSnapshot, ...]:
    history = []
    state: GameState | None = game_state
    while state is not None and len(history) < history_length:
        history.append(_snapshot_board(state))
        state = state.previous_state
    return tuple(history)
```

## zero_training_pipeline/supervised_pretraining.py

```python
from __future__ import annotations

import random
import re
import time
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
                game.board.snapshot_key(),
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
    max_seconds: float | None = None,
) -> list[float]:
    if steps < 1:
        return []
    if not examples:
        raise ValueError("supervised pretraining needs at least one example")

    losses = []
    started_at = time.monotonic()
    for _ in range(steps):
        if max_seconds is not None and time.monotonic() - started_at >= max_seconds:
            break
        batch = _sample_examples(examples, batch_size, rng)
        losses.append(
            train_step(
                model=model,  # type: ignore[arg-type]
                optimizer=optimizer,
                examples=batch,
                board_size=board_size,
                history_length=history_length,
                device=device,
                rng=rng,
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
        history.append(state.board.snapshot_key())
        state = state.previous_state
    return tuple(history)
```

## zero_training_pipeline/torch_training.py

```python
from __future__ import annotations

from collections.abc import Sequence
import random
from typing import Protocol

import torch
import torch.nn.functional as F

from zero_training_pipeline.encoding import BoardPlanes, PolicyVector
from zero_training_pipeline.encoding import encode_board_snapshot, encode_policy
from zero_training_pipeline.self_play import TrainingExample


class PolicyValueModel(Protocol):
    def train(self, mode: bool = True) -> PolicyValueModel:
        ...

    def __call__(self, board_planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...


def examples_to_tensors(
    examples: Sequence[TrainingExample],
    board_size: int,
    history_length: int = 1,
    device: torch.device | str = "cpu",
    augment: bool = False,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boards = []
    policies = []
    augmentation_rng = rng or random
    for example in examples:
        board_planes = encode_board_snapshot(
            example.board,
            example.player,
            board_size,
            history_length=history_length,
            board_history=example.board_history or (example.board,),
        )
        policy = encode_policy(example.visit_distribution, board_size)
        if augment:
            symmetry = augmentation_rng.randrange(8)
            board_planes = transform_board_planes(board_planes, symmetry)
            policy = transform_policy(policy, board_size, symmetry)
        boards.append(board_planes)
        policies.append(policy)
    values = [example.value for example in examples]

    return (
        torch.tensor(boards, dtype=torch.float32, device=device),
        torch.tensor(policies, dtype=torch.float32, device=device),
        torch.tensor(values, dtype=torch.float32, device=device),
    )


def policy_value_loss(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
) -> torch.Tensor:
    policy_loss = -(target_policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1)
    value_loss = F.mse_loss(values, target_values, reduction="none")
    return (policy_loss + value_loss).mean()


def train_step(
    model: PolicyValueModel,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[TrainingExample],
    board_size: int,
    history_length: int = 1,
    device: torch.device | str = "cpu",
    augment: bool = True,
    rng: random.Random | None = None,
) -> float:
    model.train()
    boards, target_policies, target_values = examples_to_tensors(
        examples,
        board_size=board_size,
        history_length=history_length,
        device=device,
        augment=augment,
        rng=rng,
    )

    optimizer.zero_grad()
    policy_logits, values = model(boards)
    loss = policy_value_loss(policy_logits, values, target_policies, target_values)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def transform_board_planes(
    board_planes: BoardPlanes,
    symmetry: int,
) -> BoardPlanes:
    if not 0 <= symmetry < 8:
        raise ValueError("symmetry must be in [0, 7]")
    size = len(board_planes[0])
    transformed_planes = []
    for plane in board_planes:
        transformed = [[0 for _ in range(size)] for _ in range(size)]
        for row in range(size):
            for col in range(size):
                new_row, new_col = _transform_coords(row, col, size, symmetry)
                transformed[new_row][new_col] = plane[row][col]
        transformed_planes.append(tuple(tuple(row_values) for row_values in transformed))
    return tuple(transformed_planes)


def transform_policy(
    policy: PolicyVector,
    board_size: int,
    symmetry: int,
) -> PolicyVector:
    if len(policy) != board_size * board_size + 1:
        raise ValueError("policy length does not match board size")

    transformed = [0.0 for _ in policy]
    for row in range(board_size):
        for col in range(board_size):
            old_index = row * board_size + col
            new_row, new_col = _transform_coords(row, col, board_size, symmetry)
            new_index = new_row * board_size + new_col
            transformed[new_index] = policy[old_index]
    transformed[-1] = policy[-1]
    return tuple(transformed)


def _transform_coords(
    row: int,
    col: int,
    size: int,
    symmetry: int,
) -> tuple[int, int]:
    if symmetry == 0:
        return row, col
    if symmetry == 1:
        return col, size - 1 - row
    if symmetry == 2:
        return size - 1 - row, size - 1 - col
    if symmetry == 3:
        return size - 1 - col, row
    if symmetry == 4:
        return row, size - 1 - col
    if symmetry == 5:
        return size - 1 - row, col
    if symmetry == 6:
        return col, row
    if symmetry == 7:
        return size - 1 - col, size - 1 - row
    raise ValueError("symmetry must be in [0, 7]")
```

## zero_training_pipeline/weight_exports.py

```python
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class WeightExportResult:
    weights_path: Path
    manifest_path: Path
    bundle_path: Path
    auto_download_started: bool


def export_checkpoint_weights(
    checkpoint_path: Path,
    architecture: str,
    output_dir: Path = Path("trained_model_weights"),
    auto_download: bool = False,
) -> WeightExportResult:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    iteration = int(checkpoint["iteration"])
    output_dir = Path(output_dir)
    architecture_dir = output_dir / architecture
    architecture_dir.mkdir(parents=True, exist_ok=True)

    stem = f"neogozero_{architecture}_iteration_{iteration:04d}"
    weights_path = architecture_dir / f"{stem}_weights.pt"
    manifest_path = architecture_dir / f"{stem}_manifest.json"
    bundle_path = architecture_dir / f"{stem}_download_bundle.zip"

    weights_payload = {
        "architecture": architecture,
        "iteration": iteration,
        "model_state": checkpoint["model_state"],
        "config": checkpoint["config"],
        "source_checkpoint": str(checkpoint_path),
        "promoted": checkpoint.get("promoted"),
        "candidate_win_rate": checkpoint.get("candidate_win_rate"),
    }
    torch.save(weights_payload, weights_path)

    manifest = {
        "architecture": architecture,
        "iteration": iteration,
        "weights_file": weights_path.name,
        "source_checkpoint": str(checkpoint_path),
        "config": checkpoint["config"],
        "promoted": checkpoint.get("promoted"),
        "candidate_win_rate": checkpoint.get("candidate_win_rate"),
    }
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(weights_path, arcname=weights_path.name)
        bundle.write(manifest_path, arcname=manifest_path.name)

    auto_download_started = _try_colab_download(bundle_path) if auto_download else False
    return WeightExportResult(
        weights_path=weights_path,
        manifest_path=manifest_path,
        bundle_path=bundle_path,
        auto_download_started=auto_download_started,
    )


def _try_colab_download(path: Path) -> bool:
    try:
        from google.colab import files  # type: ignore[import-not-found]
    except ImportError:
        return False

    files.download(str(path))
    return True


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
```

## policy_value_networks/__init__.py

```python
"""Policy-value network implementations."""
```

## policy_value_networks/resnet_policy_value/__init__.py

```python
"""Neural network models and evaluators."""

from policy_value_networks.resnet_policy_value.policy_value import (
    PolicyValueNet,
    ResidualBlock,
    TorchPolicyValueEvaluator,
)

__all__ = ["PolicyValueNet", "ResidualBlock", "TorchPolicyValueEvaluator"]
```

## policy_value_networks/resnet_policy_value/policy_value.py

```python
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import nn

from search_players.mcts_bot import Evaluation
from go_engine.game import GameState
from zero_training_pipeline.encoding import encode_game_state, move_to_index


def history_length_from_input_planes(input_planes: int) -> int:
    if input_planes < 1 or input_planes % 2 == 0:
        raise ValueError("input_planes must be a positive odd number")
    return (input_planes - 1) // 2


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        return self.relu(outputs + residual)


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        board_size: int = 9,
        input_planes: int = 17,
        channels: int = 256,
        num_res_blocks: int = 20,
    ) -> None:
        super().__init__()
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be at least 1")
        history_length_from_input_planes(input_planes)

        self.board_size = board_size
        self.policy_size = board_size * board_size + 1
        self.input_planes = input_planes
        self.channels = channels
        self.num_res_blocks = num_res_blocks

        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.residual_tower = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(num_res_blocks))
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, self.policy_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, board_planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stem(board_planes)
        features = self.residual_tower(features)
        policy_logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return policy_logits, values


@dataclass
class TorchPolicyValueEvaluator:
    model: PolicyValueNet
    device: torch.device | str = "cpu"

    def evaluate(self, game_state: GameState) -> Evaluation:
        return self.evaluate_many((game_state,))[0]

    def evaluate_many(self, game_states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        if not game_states:
            return ()

        self.model.eval()
        history_length = history_length_from_input_planes(self.model.input_planes)
        board_planes = torch.tensor(
            [
                encode_game_state(game_state, history_length=history_length)
                for game_state in game_states
            ],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            policy_logits, values = self.model(board_planes)
            probabilities = torch.softmax(policy_logits, dim=1)

        evaluations = []
        for index, game_state in enumerate(game_states):
            move_priors = {
                move: float(probabilities[index, move_to_index(move, game_state.board.size)].item())
                for move in game_state.legal_moves()
            }
            evaluations.append(
                Evaluation.from_priors(
                    game_state,
                    move_priors=move_priors,
                    value=float(values[index].item()),
                )
            )
        return tuple(evaluations)
```

## zero_training_pipeline/zero_loop.py

```python
from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path

import torch

from search_players.mcts_bot import MCTSBot
from match_evaluation.match import play_game
from go_engine.types import Player
from policy_value_networks.resnet_policy_value.policy_value import PolicyValueNet, TorchPolicyValueEvaluator
from zero_training_pipeline.self_play import TrainingExample, generate_self_play_game
from zero_training_pipeline.supervised_pretraining import (
    load_sgf_training_examples,
    run_supervised_pretraining,
)
from zero_training_pipeline.torch_training import train_step


@dataclass(frozen=True)
class ZeroTrainingConfig:
    board_size: int = 3
    iterations: int = 1
    self_play_games_per_iteration: int = 1
    mcts_rounds: int = 2
    mcts_inference_batch_size: int = 1
    max_rollout_moves: int | None = 12
    training_steps_per_iteration: int = 2
    batch_size: int = 8
    learning_rate: float = 0.01
    channels: int = 16
    num_res_blocks: int = 2
    history_length: int = 1
    replay_buffer_size: int = 1_000
    evaluation_games: int = 0
    promotion_threshold: float = 0.55
    self_play_temperature: float = 1.0
    temperature_drop_move: int = 30
    dirichlet_alpha: float = 0.03
    dirichlet_epsilon: float = 0.25
    checkpoint_dir: Path = Path("checkpoints")
    resume_checkpoint: Path | None = None
    metrics_path: Path | None = None
    self_play_records_path: Path | None = None
    max_training_seconds: float | None = None
    supervised_sgf_dir: Path | None = None
    supervised_steps: int = 0
    supervised_max_seconds: float | None = None
    supervised_max_examples: int | None = None
    supervised_batch_size: int | None = None
    seed: int = 1
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.board_size < 2:
            raise ValueError("board_size must be at least 2")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.self_play_games_per_iteration < 1:
            raise ValueError("self_play_games_per_iteration must be at least 1")
        if self.mcts_rounds < 1:
            raise ValueError("mcts_rounds must be at least 1")
        if self.mcts_inference_batch_size < 1:
            raise ValueError("mcts_inference_batch_size must be at least 1")
        if self.max_rollout_moves is not None and self.max_rollout_moves < 1:
            raise ValueError("max_rollout_moves must be at least 1")
        if self.training_steps_per_iteration < 1:
            raise ValueError("training_steps_per_iteration must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.channels < 1:
            raise ValueError("channels must be at least 1")
        if self.num_res_blocks < 1:
            raise ValueError("num_res_blocks must be at least 1")
        if self.history_length < 1:
            raise ValueError("history_length must be at least 1")
        if self.replay_buffer_size < 1:
            raise ValueError("replay_buffer_size must be at least 1")
        if self.evaluation_games < 0:
            raise ValueError("evaluation_games cannot be negative")
        if not 0 <= self.promotion_threshold <= 1:
            raise ValueError("promotion_threshold must be in [0, 1]")
        if self.self_play_temperature < 0:
            raise ValueError("self_play_temperature cannot be negative")
        if self.temperature_drop_move < 0:
            raise ValueError("temperature_drop_move cannot be negative")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError("dirichlet_epsilon must be in [0, 1]")
        if self.supervised_steps < 0:
            raise ValueError("supervised_steps cannot be negative")
        if self.supervised_max_seconds is not None and self.supervised_max_seconds <= 0:
            raise ValueError("supervised_max_seconds must be positive")
        if self.max_training_seconds is not None and self.max_training_seconds <= 0:
            raise ValueError("max_training_seconds must be positive")
        if self.supervised_max_examples is not None and self.supervised_max_examples < 1:
            raise ValueError("supervised_max_examples must be at least 1")
        if self.supervised_batch_size is not None and self.supervised_batch_size < 1:
            raise ValueError("supervised_batch_size must be at least 1")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["checkpoint_dir"] = str(self.checkpoint_dir)
        data["resume_checkpoint"] = (
            str(self.resume_checkpoint) if self.resume_checkpoint is not None else None
        )
        data["metrics_path"] = str(self.metrics_path) if self.metrics_path is not None else None
        data["self_play_records_path"] = (
            str(self.self_play_records_path) if self.self_play_records_path is not None else None
        )
        data["supervised_sgf_dir"] = (
            str(self.supervised_sgf_dir) if self.supervised_sgf_dir is not None else None
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ZeroTrainingConfig:
        defaults = {}
        for config_field in fields(cls):
            if config_field.default is not MISSING:
                defaults[config_field.name] = config_field.default
            elif config_field.default_factory is not MISSING:
                defaults[config_field.name] = config_field.default_factory()
        converted = {**defaults, **data}
        converted["checkpoint_dir"] = Path(str(converted["checkpoint_dir"]))
        if converted.get("resume_checkpoint") is not None:
            converted["resume_checkpoint"] = Path(str(converted["resume_checkpoint"]))
        if converted.get("metrics_path") is not None:
            converted["metrics_path"] = Path(str(converted["metrics_path"]))
        if converted.get("self_play_records_path") is not None:
            converted["self_play_records_path"] = Path(str(converted["self_play_records_path"]))
        if converted.get("supervised_sgf_dir") is not None:
            converted["supervised_sgf_dir"] = Path(str(converted["supervised_sgf_dir"]))
        return cls(**converted)


@dataclass
class ReplayBuffer:
    capacity: int
    examples: list[TrainingExample] = field(default_factory=list)

    def add_examples(self, examples: list[TrainingExample] | tuple[TrainingExample, ...]) -> None:
        self.examples.extend(examples)
        if len(self.examples) > self.capacity:
            self.examples = self.examples[-self.capacity :]

    def sample(self, batch_size: int, rng: random.Random) -> list[TrainingExample]:
        if not self.examples:
            raise ValueError("cannot sample from an empty replay buffer")
        if batch_size <= len(self.examples):
            return rng.sample(self.examples, batch_size)
        return [rng.choice(self.examples) for _ in range(batch_size)]

    def __len__(self) -> int:
        return len(self.examples)


@dataclass(frozen=True)
class TrainingIterationResult:
    iteration: int
    generated_examples: int
    mean_loss: float
    candidate_win_rate: float
    promoted: bool
    checkpoint_path: Path


@dataclass(frozen=True)
class TrainingRunResult:
    iterations: tuple[TrainingIterationResult, ...]
    final_checkpoint_path: Path


def run_zero_training(config: ZeroTrainingConfig) -> TrainingRunResult:
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    training_started_at = time.monotonic()
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_iteration = 0
    if config.resume_checkpoint is not None:
        model, optimizer, loaded_config, replay_buffer, start_iteration = load_checkpoint(
            config.resume_checkpoint
        )
        _ensure_resume_compatible(config, loaded_config)
        replay_buffer.capacity = config.replay_buffer_size
    else:
        model = _new_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        replay_buffer = ReplayBuffer(capacity=config.replay_buffer_size)
        if _maybe_run_supervised_pretraining(config, model, optimizer, rng):
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                iteration=0,
                model=model,
                optimizer=optimizer,
                config=config,
                replay_buffer=replay_buffer,
                promoted=True,
                candidate_win_rate=1.0,
            )

    iteration_results: list[TrainingIterationResult] = []

    for offset in range(1, config.iterations + 1):
        if _time_budget_exhausted(config, training_started_at):
            print("Time budget reached before starting next ResNet iteration.", flush=True)
            break

        iteration = start_iteration + offset
        champion_model = _clone_model(model, config)
        generated_examples = _run_self_play(
            config,
            model,
            replay_buffer,
            rng,
            iteration,
            training_started_at,
        )
        losses = _train_from_replay(
            config,
            model,
            optimizer,
            replay_buffer,
            rng,
            training_started_at,
        )

        if _time_budget_exhausted(config, training_started_at):
            candidate_win_rate = 1.0
        else:
            candidate_win_rate = _evaluate_candidate(
                config=config,
                candidate_model=model,
                champion_model=champion_model,
                rng=rng,
            )
        promoted = candidate_win_rate >= config.promotion_threshold
        if not promoted:
            model.load_state_dict(champion_model.state_dict())

        checkpoint_path = save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            iteration=iteration,
            model=model,
            optimizer=optimizer,
            config=config,
            replay_buffer=replay_buffer,
            promoted=promoted,
            candidate_win_rate=candidate_win_rate,
        )
        iteration_results.append(
            TrainingIterationResult(
                iteration=iteration,
                generated_examples=generated_examples,
                mean_loss=sum(losses) / len(losses) if losses else 0.0,
                candidate_win_rate=candidate_win_rate,
                promoted=promoted,
                checkpoint_path=checkpoint_path,
            )
        )
        _write_metrics(config, iteration_results[-1], len(replay_buffer))

    if not iteration_results:
        checkpoint_path = save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            iteration=start_iteration,
            model=model,
            optimizer=optimizer,
            config=config,
            replay_buffer=replay_buffer,
            promoted=True,
            candidate_win_rate=1.0,
        )
        return TrainingRunResult(iterations=(), final_checkpoint_path=checkpoint_path)

    return TrainingRunResult(
        iterations=tuple(iteration_results),
        final_checkpoint_path=iteration_results[-1].checkpoint_path,
    )


def save_checkpoint(
    checkpoint_dir: Path,
    iteration: int,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    config: ZeroTrainingConfig,
    replay_buffer: ReplayBuffer,
    promoted: bool,
    candidate_win_rate: float,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"iteration_{iteration:04d}.pt"
    torch.save(
        {
            "iteration": iteration,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config.to_dict(),
            "replay_examples": replay_buffer.examples,
            "promoted": promoted,
            "candidate_win_rate": candidate_win_rate,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
) -> tuple[PolicyValueNet, torch.optim.Optimizer, ZeroTrainingConfig, ReplayBuffer, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = ZeroTrainingConfig.from_dict(checkpoint["config"])
    model = _new_model(config)
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    replay_buffer = ReplayBuffer(
        capacity=config.replay_buffer_size,
        examples=list(checkpoint["replay_examples"]),
    )
    return model, optimizer, config, replay_buffer, int(checkpoint["iteration"])


def _run_self_play(
    config: ZeroTrainingConfig,
    model: PolicyValueNet,
    replay_buffer: ReplayBuffer,
    rng: random.Random,
    iteration: int,
    started_at: float,
) -> int:
    generated_examples = 0
    evaluator = TorchPolicyValueEvaluator(model, device=config.device)

    for game_index in range(1, config.self_play_games_per_iteration + 1):
        if _time_budget_exhausted(config, started_at):
            print("Time budget reached during ResNet self-play.", flush=True)
            break

        bot = MCTSBot(
            num_rounds=config.mcts_rounds,
            inference_batch_size=config.mcts_inference_batch_size,
            max_rollout_moves=config.max_rollout_moves,
            evaluator=evaluator,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            rng=random.Random(rng.randrange(1_000_000_000)),
        )
        game = generate_self_play_game(
            bot,
            board_size=config.board_size,
            max_moves=config.board_size * config.board_size * 3,
            history_length=config.history_length,
            temperature=config.self_play_temperature,
            temperature_drop_move=config.temperature_drop_move,
            add_dirichlet_noise=True,
        )
        replay_buffer.add_examples(game.examples)
        generated_examples += len(game.examples)
        _write_self_play_record(config, iteration, game_index, game)

    return generated_examples


def _write_self_play_record(
    config: ZeroTrainingConfig,
    iteration: int,
    game_index: int,
    game,
) -> None:
    if config.self_play_records_path is None:
        return

    records_path = Path(config.self_play_records_path)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    score = game.final_state.score()
    with records_path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "architecture": "resnet_policy_value",
                    "iteration": iteration,
                    "game_index": game_index,
                    "board_size": config.board_size,
                    "history_length": config.history_length,
                    "mcts_rounds": config.mcts_rounds,
                    "mcts_inference_batch_size": config.mcts_inference_batch_size,
                    "temperature": config.self_play_temperature,
                    "temperature_drop_move": config.temperature_drop_move,
                    "dirichlet_alpha": config.dirichlet_alpha,
                    "dirichlet_epsilon": config.dirichlet_epsilon,
                    "winner": game.winner.value,
                    "num_moves": len(game.moves),
                    "num_examples": len(game.examples),
                    "black_score": score.black,
                    "white_score": score.white,
                    "score_margin": score.margin,
                    "moves": [_move_to_data(move) for move in game.moves],
                }
            )
            + "\n"
        )


def _move_to_data(move) -> dict[str, int | str]:
    if move.is_pass:
        return {"type": "pass"}
    assert move.point is not None
    return {
        "type": "play",
        "row": move.point.row,
        "col": move.point.col,
    }


def _time_budget_exhausted(
    config: ZeroTrainingConfig,
    started_at: float,
) -> bool:
    if config.max_training_seconds is None:
        return False
    return time.monotonic() - started_at >= config.max_training_seconds


def _maybe_run_supervised_pretraining(
    config: ZeroTrainingConfig,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
) -> bool:
    if config.supervised_sgf_dir is None or config.supervised_steps == 0:
        return False

    examples = load_sgf_training_examples(
        sgf_dir=config.supervised_sgf_dir,
        board_size=config.board_size,
        history_length=config.history_length,
        max_examples=config.supervised_max_examples,
    )
    batch_size = config.supervised_batch_size or config.batch_size
    losses = run_supervised_pretraining(
        model=model,
        optimizer=optimizer,
        examples=examples,
        board_size=config.board_size,
        history_length=config.history_length,
        steps=config.supervised_steps,
        batch_size=batch_size,
        device=config.device,
        rng=rng,
        max_seconds=config.supervised_max_seconds,
    )
    mean_loss = sum(losses) / len(losses) if losses else 0.0
    print(
        "Supervised pretraining complete: "
        f"examples={len(examples)}, "
        f"steps={len(losses)}/{config.supervised_steps}, "
        f"loss={mean_loss:.4f}",
        flush=True,
    )
    return True


def _train_from_replay(
    config: ZeroTrainingConfig,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    rng: random.Random,
    started_at: float,
) -> list[float]:
    losses = []
    for _ in range(config.training_steps_per_iteration):
        if _time_budget_exhausted(config, started_at):
            print("Time budget reached during ResNet replay training.", flush=True)
            break
        batch = replay_buffer.sample(config.batch_size, rng)
        losses.append(
            train_step(
                model=model,
                optimizer=optimizer,
                examples=batch,
                board_size=config.board_size,
                history_length=config.history_length,
                device=config.device,
                rng=rng,
            )
        )
    return losses


def _evaluate_candidate(
    config: ZeroTrainingConfig,
    candidate_model: PolicyValueNet,
    champion_model: PolicyValueNet,
    rng: random.Random,
) -> float:
    if config.evaluation_games == 0:
        return 1.0

    candidate_wins = 0
    total_games = config.evaluation_games * 2
    candidate_evaluator = TorchPolicyValueEvaluator(candidate_model, device=config.device)
    champion_evaluator = TorchPolicyValueEvaluator(champion_model, device=config.device)

    for game_index in range(config.evaluation_games):
        candidate_black = MCTSBot(
            num_rounds=config.mcts_rounds,
            inference_batch_size=config.mcts_inference_batch_size,
            evaluator=candidate_evaluator,
            rng=random.Random(rng.randrange(1_000_000_000)),
        )
        champion_white = MCTSBot(
            num_rounds=config.mcts_rounds,
            inference_batch_size=config.mcts_inference_batch_size,
            evaluator=champion_evaluator,
            rng=random.Random(rng.randrange(1_000_000_000)),
        )
        result = play_game(
            black_bot=candidate_black,
            white_bot=champion_white,
            board_size=config.board_size,
            max_moves=config.board_size * config.board_size * 3,
        )
        if result.winner is Player.BLACK:
            candidate_wins += 1

        champion_black = MCTSBot(
            num_rounds=config.mcts_rounds,
            inference_batch_size=config.mcts_inference_batch_size,
            evaluator=champion_evaluator,
            rng=random.Random(rng.randrange(1_000_000_000)),
        )
        candidate_white = MCTSBot(
            num_rounds=config.mcts_rounds,
            inference_batch_size=config.mcts_inference_batch_size,
            evaluator=candidate_evaluator,
            rng=random.Random(rng.randrange(1_000_000_000)),
        )
        result = play_game(
            black_bot=champion_black,
            white_bot=candidate_white,
            board_size=config.board_size,
            max_moves=config.board_size * config.board_size * 3,
        )
        if result.winner is Player.WHITE:
            candidate_wins += 1

    return candidate_wins / total_games


def _new_model(config: ZeroTrainingConfig) -> PolicyValueNet:
    input_planes = 2 * config.history_length + 1
    return PolicyValueNet(
        board_size=config.board_size,
        input_planes=input_planes,
        channels=config.channels,
        num_res_blocks=config.num_res_blocks,
    ).to(config.device)


def _clone_model(model: PolicyValueNet, config: ZeroTrainingConfig) -> PolicyValueNet:
    clone = _new_model(config)
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    clone.eval()
    return clone


def _write_metrics(
    config: ZeroTrainingConfig,
    result: TrainingIterationResult,
    replay_buffer_size: int,
) -> None:
    metrics_path = config.metrics_path or Path(config.checkpoint_dir) / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "iteration": result.iteration,
                    "generated_examples": result.generated_examples,
                    "mean_loss": result.mean_loss,
                    "candidate_win_rate": result.candidate_win_rate,
                    "promoted": result.promoted,
                    "checkpoint_path": str(result.checkpoint_path),
                    "replay_buffer_size": replay_buffer_size,
                }
            )
            + "\n"
        )


def _ensure_resume_compatible(
    config: ZeroTrainingConfig,
    loaded_config: ZeroTrainingConfig,
) -> None:
    fields = (
        "board_size",
        "channels",
        "num_res_blocks",
        "history_length",
        "device",
    )
    for field_name in fields:
        if getattr(config, field_name) != getattr(loaded_config, field_name):
            raise ValueError(
                "resume checkpoint is incompatible: "
                f"{field_name}={getattr(loaded_config, field_name)!r} in checkpoint, "
                f"{getattr(config, field_name)!r} requested"
            )
```

## policy_value_networks/convnext_policy_value/__init__.py

```python
"""ConvNeXt-based NeoGoZero variant."""

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

__all__ = [
    "ConvNeXtBlock",
    "ConvNeXtPolicyValueEvaluator",
    "ConvNeXtPolicyValueNet",
    "ConvNeXtTrainingConfig",
    "LayerNorm2d",
    "load_convnext_checkpoint",
    "run_convnext_training",
]
```

## policy_value_networks/convnext_policy_value/convnext_policy_value.py

```python
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from search_players.mcts_bot import Evaluation
from go_engine.game import GameState
from zero_training_pipeline.encoding import encode_game_state, move_to_index


def history_length_from_input_planes(input_planes: int) -> int:
    if input_planes < 1 or input_planes % 2 == 0:
        raise ValueError("input_planes must be a positive odd number")
    return (input_planes - 1) // 2


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.permute(0, 2, 3, 1)
        outputs = self.layer_norm(outputs)
        return outputs.permute(0, 3, 1, 2)


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob must be in [0, 1)")
        self.drop_prob = drop_prob

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.dropout(
            inputs,
            p=self.drop_prob,
            training=self.training,
        )


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        layer_scale_init: float = 1e-6,
        stochastic_depth_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        hidden_channels = channels * mlp_ratio
        self.depthwise_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = LayerNorm2d(channels)
        self.pointwise_expand = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.pointwise_project = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.layer_scale = nn.Parameter(
            torch.ones(channels, 1, 1) * layer_scale_init
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.depthwise_conv(inputs)
        outputs = self.norm(outputs)
        outputs = self.pointwise_expand(outputs)
        outputs = self.activation(outputs)
        outputs = self.pointwise_project(outputs)
        outputs = self.layer_scale * outputs
        outputs = self.stochastic_depth(outputs)
        return residual + outputs


class ConvNeXtPolicyValueNet(nn.Module):
    def __init__(
        self,
        board_size: int = 9,
        input_planes: int = 17,
        channels: int = 256,
        num_blocks: int = 20,
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        layer_scale_init: float = 1e-6,
        stochastic_depth_prob: float = 0.1,
    ) -> None:
        super().__init__()
        history_length_from_input_planes(input_planes)
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")

        self.board_size = board_size
        self.policy_size = board_size * board_size + 1
        self.input_planes = input_planes
        self.channels = channels
        self.num_blocks = num_blocks

        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, channels, kernel_size=3, padding=1),
            LayerNorm2d(channels),
        )
        block_drop_probs = torch.linspace(0, stochastic_depth_prob, num_blocks).tolist()
        self.convnext_tower = nn.Sequential(
            *(
                ConvNeXtBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    layer_scale_init=layer_scale_init,
                    stochastic_depth_prob=block_drop_probs[index],
                )
                for index in range(num_blocks)
            )
        )
        self.final_norm = LayerNorm2d(channels)
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, self.policy_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, channels),
            nn.GELU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, board_planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stem(board_planes)
        features = self.convnext_tower(features)
        features = self.final_norm(features)
        policy_logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return policy_logits, values


@dataclass
class ConvNeXtPolicyValueEvaluator:
    model: ConvNeXtPolicyValueNet
    device: torch.device | str = "cpu"

    def evaluate(self, game_state: GameState) -> Evaluation:
        return self.evaluate_many((game_state,))[0]

    def evaluate_many(self, game_states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        if not game_states:
            return ()

        self.model.eval()
        history_length = history_length_from_input_planes(self.model.input_planes)
        board_planes = torch.tensor(
            [
                encode_game_state(game_state, history_length=history_length)
                for game_state in game_states
            ],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            policy_logits, values = self.model(board_planes)
            probabilities = torch.softmax(policy_logits, dim=1)

        evaluations = []
        for index, game_state in enumerate(game_states):
            move_priors = {
                move: float(probabilities[index, move_to_index(move, game_state.board.size)].item())
                for move in game_state.legal_moves()
            }
            evaluations.append(
                Evaluation.from_priors(
                    game_state,
                    move_priors=move_priors,
                    value=float(values[index].item()),
                )
            )
        return tuple(evaluations)
```

## policy_value_networks/convnext_policy_value/convnext_zero_loop.py

```python
from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path

import torch

from policy_value_networks.convnext_policy_value.convnext_policy_value import (
    ConvNeXtPolicyValueEvaluator,
    ConvNeXtPolicyValueNet,
)
from search_players.mcts_bot import MCTSBot
from match_evaluation.match import play_game
from go_engine.types import Player
from zero_training_pipeline.self_play import TrainingExample, generate_self_play_game
from zero_training_pipeline.supervised_pretraining import (
    load_sgf_training_examples,
    run_supervised_pretraining,
)
from zero_training_pipeline.torch_training import train_step
from zero_training_pipeline.zero_loop import (
    ReplayBuffer,
    TrainingIterationResult,
    TrainingRunResult,
)


@dataclass(frozen=True)
class ConvNeXtTrainingConfig:
    board_size: int = 3
    iterations: int = 1
    self_play_games_per_iteration: int = 1
    mcts_rounds: int = 2
    mcts_inference_batch_size: int = 1
    max_rollout_moves: int | None = 12
    training_steps_per_iteration: int = 2
    batch_size: int = 8
    learning_rate: float = 0.01
    channels: int = 16
    num_blocks: int = 2
    history_length: int = 1
    kernel_size: int = 7
    mlp_ratio: int = 4
    layer_scale_init: float = 1e-6
    stochastic_depth_prob: float = 0.1
    replay_buffer_size: int = 1_000
    evaluation_games: int = 0
    promotion_threshold: float = 0.55
    self_play_temperature: float = 1.0
    temperature_drop_move: int = 30
    dirichlet_alpha: float = 0.03
    dirichlet_epsilon: float = 0.25
    checkpoint_dir: Path = Path("checkpoints_convnext_policy_value")
    resume_checkpoint: Path | None = None
    metrics_path: Path | None = None
    self_play_records_path: Path | None = None
    max_training_seconds: float | None = None
    supervised_sgf_dir: Path | None = None
    supervised_steps: int = 0
    supervised_max_seconds: float | None = None
    supervised_max_examples: int | None = None
    supervised_batch_size: int | None = None
    seed: int = 1
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.board_size < 2:
            raise ValueError("board_size must be at least 2")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.self_play_games_per_iteration < 1:
            raise ValueError("self_play_games_per_iteration must be at least 1")
        if self.mcts_rounds < 1:
            raise ValueError("mcts_rounds must be at least 1")
        if self.mcts_inference_batch_size < 1:
            raise ValueError("mcts_inference_batch_size must be at least 1")
        if self.max_rollout_moves is not None and self.max_rollout_moves < 1:
            raise ValueError("max_rollout_moves must be at least 1")
        if self.training_steps_per_iteration < 1:
            raise ValueError("training_steps_per_iteration must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.channels < 1:
            raise ValueError("channels must be at least 1")
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        if self.history_length < 1:
            raise ValueError("history_length must be at least 1")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        if self.mlp_ratio < 1:
            raise ValueError("mlp_ratio must be at least 1")
        if self.replay_buffer_size < 1:
            raise ValueError("replay_buffer_size must be at least 1")
        if self.evaluation_games < 0:
            raise ValueError("evaluation_games cannot be negative")
        if not 0 <= self.promotion_threshold <= 1:
            raise ValueError("promotion_threshold must be in [0, 1]")
        if self.self_play_temperature < 0:
            raise ValueError("self_play_temperature cannot be negative")
        if self.temperature_drop_move < 0:
            raise ValueError("temperature_drop_move cannot be negative")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError("dirichlet_epsilon must be in [0, 1]")
        if self.supervised_steps < 0:
            raise ValueError("supervised_steps cannot be negative")
        if self.supervised_max_seconds is not None and self.supervised_max_seconds <= 0:
            raise ValueError("supervised_max_seconds must be positive")
        if self.max_training_seconds is not None and self.max_training_seconds <= 0:
            raise ValueError("max_training_seconds must be positive")
        if self.supervised_max_examples is not None and self.supervised_max_examples < 1:
            raise ValueError("supervised_max_examples must be at least 1")
        if self.supervised_batch_size is not None and self.supervised_batch_size < 1:
            raise ValueError("supervised_batch_size must be at least 1")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["checkpoint_dir"] = str(self.checkpoint_dir)
        data["resume_checkpoint"] = (
            str(self.resume_checkpoint) if self.resume_checkpoint is not None else None
        )
        data["metrics_path"] = str(self.metrics_path) if self.metrics_path is not None else None
        data["self_play_records_path"] = (
            str(self.self_play_records_path) if self.self_play_records_path is not None else None
        )
        data["supervised_sgf_dir"] = (
            str(self.supervised_sgf_dir) if self.supervised_sgf_dir is not None else None
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ConvNeXtTrainingConfig:
        defaults = {}
        for config_field in fields(cls):
            if config_field.default is not MISSING:
                defaults[config_field.name] = config_field.default
            elif config_field.default_factory is not MISSING:
                defaults[config_field.name] = config_field.default_factory()
        converted = {**defaults, **data}
        converted["checkpoint_dir"] = Path(str(converted["checkpoint_dir"]))
        if converted.get("resume_checkpoint") is not None:
            converted["resume_checkpoint"] = Path(str(converted["resume_checkpoint"]))
        if converted.get("metrics_path") is not None:
            converted["metrics_path"] = Path(str(converted["metrics_path"]))
        if converted.get("self_play_records_path") is not None:
            converted["self_play_records_path"] = Path(str(converted["self_play_records_path"]))
        if converted.get("supervised_sgf_dir") is not None:
            converted["supervised_sgf_dir"] = Path(str(converted["supervised_sgf_dir"]))
        return cls(**converted)


def run_convnext_training(config: ConvNeXtTrainingConfig) -> TrainingRunResult:
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    training_started_at = time.monotonic()
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_iteration = 0
    if config.resume_checkpoint is not None:
        model, optimizer, loaded_config, replay_buffer, start_iteration = (
            load_convnext_checkpoint(config.resume_checkpoint)
        )
        _ensure_resume_compatible(config, loaded_config)
        replay_buffer.capacity = config.replay_buffer_size
    else:
        model = _new_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        replay_buffer = ReplayBuffer(capacity=config.replay_buffer_size)
        if _maybe_run_supervised_pretraining(config, model, optimizer, rng):
            save_convnext_checkpoint(
                checkpoint_dir=checkpoint_dir,
                iteration=0,
                model=model,
                optimizer=optimizer,
                config=config,
                replay_buffer=replay_buffer,
                promoted=True,
                candidate_win_rate=1.0,
            )

    iteration_results: list[TrainingIterationResult] = []

    for offset in range(1, config.iterations + 1):
        if _time_budget_exhausted(config, training_started_at):
            print("Time budget reached before starting next ConvNeXt iteration.", flush=True)
            break

        iteration = start_iteration + offset
        champion_model = _clone_model(model, config)
        generated_examples = _run_self_play(
            config,
            model,
            replay_buffer,
            rng,
            iteration,
            training_started_at,
        )
        losses = _train_from_replay(
            config,
            model,
            optimizer,
            replay_buffer,
            rng,
            training_started_at,
        )
        if _time_budget_exhausted(config, training_started_at):
            candidate_win_rate = 1.0
        else:
            candidate_win_rate = _evaluate_candidate(config, model, champion_model, rng)
        promoted = candidate_win_rate >= config.promotion_threshold
        if not promoted:
            model.load_state_dict(champion_model.state_dict())

        checkpoint_path = save_convnext_checkpoint(
            checkpoint_dir=checkpoint_dir,
            iteration=iteration,
            model=model,
            optimizer=optimizer,
            config=config,
            replay_buffer=replay_buffer,
            promoted=promoted,
            candidate_win_rate=candidate_win_rate,
        )
        result = TrainingIterationResult(
            iteration=iteration,
            generated_examples=generated_examples,
            mean_loss=sum(losses) / len(losses) if losses else 0.0,
            candidate_win_rate=candidate_win_rate,
            promoted=promoted,
            checkpoint_path=checkpoint_path,
        )
        iteration_results.append(result)
        _write_metrics(config, result, len(replay_buffer))

    if not iteration_results:
        checkpoint_path = save_convnext_checkpoint(
            checkpoint_dir=checkpoint_dir,
            iteration=start_iteration,
            model=model,
            optimizer=optimizer,
            config=config,
            replay_buffer=replay_buffer,
            promoted=True,
            candidate_win_rate=1.0,
        )
        return TrainingRunResult(iterations=(), final_checkpoint_path=checkpoint_path)

    return TrainingRunResult(
        iterations=tuple(iteration_results),
        final_checkpoint_path=iteration_results[-1].checkpoint_path,
    )


def save_convnext_checkpoint(
    checkpoint_dir: Path,
    iteration: int,
    model: ConvNeXtPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    config: ConvNeXtTrainingConfig,
    replay_buffer: ReplayBuffer,
    promoted: bool,
    candidate_win_rate: float,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"convnext_iteration_{iteration:04d}.pt"
    torch.save(
        {
            "iteration": iteration,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config.to_dict(),
            "replay_examples": replay_buffer.examples,
            "promoted": promoted,
            "candidate_win_rate": candidate_win_rate,
            "architecture": "convnext",
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_convnext_checkpoint(
    checkpoint_path: Path,
) -> tuple[
    ConvNeXtPolicyValueNet,
    torch.optim.Optimizer,
    ConvNeXtTrainingConfig,
    ReplayBuffer,
    int,
]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = ConvNeXtTrainingConfig.from_dict(checkpoint["config"])
    model = _new_model(config)
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    replay_buffer = ReplayBuffer(
        capacity=config.replay_buffer_size,
        examples=list(checkpoint["replay_examples"]),
    )
    return model, optimizer, config, replay_buffer, int(checkpoint["iteration"])


def _run_self_play(
    config: ConvNeXtTrainingConfig,
    model: ConvNeXtPolicyValueNet,
    replay_buffer: ReplayBuffer,
    rng: random.Random,
    iteration: int,
    started_at: float,
) -> int:
    generated_examples = 0
    evaluator = ConvNeXtPolicyValueEvaluator(model, device=config.device)

    for game_index in range(1, config.self_play_games_per_iteration + 1):
        if _time_budget_exhausted(config, started_at):
            print("Time budget reached during ConvNeXt self-play.", flush=True)
            break

        bot = MCTSBot(
            num_rounds=config.mcts_rounds,
            inference_batch_size=config.mcts_inference_batch_size,
            max_rollout_moves=config.max_rollout_moves,
            evaluator=evaluator,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            rng=random.Random(rng.randrange(1_000_000_000)),
        )
        game = generate_self_play_game(
            bot,
            board_size=config.board_size,
            max_moves=config.board_size * config.board_size * 3,
            history_length=config.history_length,
            temperature=config.self_play_temperature,
            temperature_drop_move=config.temperature_drop_move,
            add_dirichlet_noise=True,
        )
        replay_buffer.add_examples(game.examples)
        generated_examples += len(game.examples)
        _write_self_play_record(config, iteration, game_index, game)

    return generated_examples


def _write_self_play_record(
    config: ConvNeXtTrainingConfig,
    iteration: int,
    game_index: int,
    game,
) -> None:
    if config.self_play_records_path is None:
        return

    records_path = Path(config.self_play_records_path)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    score = game.final_state.score()
    with records_path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "architecture": "convnext_policy_value",
                    "iteration": iteration,
                    "game_index": game_index,
                    "board_size": config.board_size,
                    "history_length": config.history_length,
                    "mcts_rounds": config.mcts_rounds,
                    "mcts_inference_batch_size": config.mcts_inference_batch_size,
                    "temperature": config.self_play_temperature,
                    "temperature_drop_move": config.temperature_drop_move,
                    "dirichlet_alpha": config.dirichlet_alpha,
                    "dirichlet_epsilon": config.dirichlet_epsilon,
                    "winner": game.winner.value,
                    "num_moves": len(game.moves),
                    "num_examples": len(game.examples),
                    "black_score": score.black,
                    "white_score": score.white,
                    "score_margin": score.margin,
                    "moves": [_move_to_data(move) for move in game.moves],
                }
            )
            + "\n"
        )


def _move_to_data(move) -> dict[str, int | str]:
    if move.is_pass:
        return {"type": "pass"}
    assert move.point is not None
    return {
        "type": "play",
        "row": move.point.row,
        "col": move.point.col,
    }


def _time_budget_exhausted(
    config: ConvNeXtTrainingConfig,
    started_at: float,
) -> bool:
    if config.max_training_seconds is None:
        return False
    return time.monotonic() - started_at >= config.max_training_seconds


def _maybe_run_supervised_pretraining(
    config: ConvNeXtTrainingConfig,
    model: ConvNeXtPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
) -> bool:
    if config.supervised_sgf_dir is None or config.supervised_steps == 0:
        return False

    examples = load_sgf_training_examples(
        sgf_dir=config.supervised_sgf_dir,
        board_size=config.board_size,
        history_length=config.history_length,
        max_examples=config.supervised_max_examples,
    )
    batch_size = config.supervised_batch_size or config.batch_size
    losses = run_supervised_pretraining(
        model=model,
        optimizer=optimizer,
        examples=examples,
        board_size=config.board_size,
        history_length=config.history_length,
        steps=config.supervised_steps,
        batch_size=batch_size,
        device=config.device,
        rng=rng,
        max_seconds=config.supervised_max_seconds,
    )
    mean_loss = sum(losses) / len(losses) if losses else 0.0
    print(
        "Supervised pretraining complete: "
        f"examples={len(examples)}, "
        f"steps={len(losses)}/{config.supervised_steps}, "
        f"loss={mean_loss:.4f}",
        flush=True,
    )
    return True


def _train_from_replay(
    config: ConvNeXtTrainingConfig,
    model: ConvNeXtPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    rng: random.Random,
    started_at: float,
) -> list[float]:
    losses = []
    for _ in range(config.training_steps_per_iteration):
        if _time_budget_exhausted(config, started_at):
            print("Time budget reached during ConvNeXt replay training.", flush=True)
            break
        batch = replay_buffer.sample(config.batch_size, rng)
        losses.append(
            train_step(
                model=model,
                optimizer=optimizer,
                examples=batch,
                board_size=config.board_size,
                history_length=config.history_length,
                device=config.device,
                rng=rng,
            )
        )
    return losses


def _evaluate_candidate(
    config: ConvNeXtTrainingConfig,
    candidate_model: ConvNeXtPolicyValueNet,
    champion_model: ConvNeXtPolicyValueNet,
    rng: random.Random,
) -> float:
    if config.evaluation_games == 0:
        return 1.0

    candidate_wins = 0
    total_games = config.evaluation_games * 2
    candidate_evaluator = ConvNeXtPolicyValueEvaluator(candidate_model, device=config.device)
    champion_evaluator = ConvNeXtPolicyValueEvaluator(champion_model, device=config.device)

    for _ in range(config.evaluation_games):
        result = play_game(
            black_bot=MCTSBot(
                num_rounds=config.mcts_rounds,
                inference_batch_size=config.mcts_inference_batch_size,
                evaluator=candidate_evaluator,
                rng=random.Random(rng.randrange(1_000_000_000)),
            ),
            white_bot=MCTSBot(
                num_rounds=config.mcts_rounds,
                inference_batch_size=config.mcts_inference_batch_size,
                evaluator=champion_evaluator,
                rng=random.Random(rng.randrange(1_000_000_000)),
            ),
            board_size=config.board_size,
            max_moves=config.board_size * config.board_size * 3,
        )
        if result.winner is Player.BLACK:
            candidate_wins += 1

        result = play_game(
            black_bot=MCTSBot(
                num_rounds=config.mcts_rounds,
                inference_batch_size=config.mcts_inference_batch_size,
                evaluator=champion_evaluator,
                rng=random.Random(rng.randrange(1_000_000_000)),
            ),
            white_bot=MCTSBot(
                num_rounds=config.mcts_rounds,
                inference_batch_size=config.mcts_inference_batch_size,
                evaluator=candidate_evaluator,
                rng=random.Random(rng.randrange(1_000_000_000)),
            ),
            board_size=config.board_size,
            max_moves=config.board_size * config.board_size * 3,
        )
        if result.winner is Player.WHITE:
            candidate_wins += 1

    return candidate_wins / total_games


def _new_model(config: ConvNeXtTrainingConfig) -> ConvNeXtPolicyValueNet:
    input_planes = 2 * config.history_length + 1
    return ConvNeXtPolicyValueNet(
        board_size=config.board_size,
        input_planes=input_planes,
        channels=config.channels,
        num_blocks=config.num_blocks,
        kernel_size=config.kernel_size,
        mlp_ratio=config.mlp_ratio,
        layer_scale_init=config.layer_scale_init,
        stochastic_depth_prob=config.stochastic_depth_prob,
    ).to(config.device)


def _clone_model(
    model: ConvNeXtPolicyValueNet,
    config: ConvNeXtTrainingConfig,
) -> ConvNeXtPolicyValueNet:
    clone = _new_model(config)
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    clone.eval()
    return clone


def _write_metrics(
    config: ConvNeXtTrainingConfig,
    result: TrainingIterationResult,
    replay_buffer_size: int,
) -> None:
    metrics_path = config.metrics_path or Path(config.checkpoint_dir) / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "architecture": "convnext",
                    "iteration": result.iteration,
                    "generated_examples": result.generated_examples,
                    "mean_loss": result.mean_loss,
                    "candidate_win_rate": result.candidate_win_rate,
                    "promoted": result.promoted,
                    "checkpoint_path": str(result.checkpoint_path),
                    "replay_buffer_size": replay_buffer_size,
                }
            )
            + "\n"
        )


def _ensure_resume_compatible(
    config: ConvNeXtTrainingConfig,
    loaded_config: ConvNeXtTrainingConfig,
) -> None:
    fields_to_check = (
        "board_size",
        "channels",
        "num_blocks",
        "history_length",
        "kernel_size",
        "mlp_ratio",
        "device",
    )
    for field_name in fields_to_check:
        if getattr(config, field_name) != getattr(loaded_config, field_name):
            raise ValueError(
                "resume checkpoint is incompatible: "
                f"{field_name}={getattr(loaded_config, field_name)!r} in checkpoint, "
                f"{getattr(config, field_name)!r} requested"
            )
```

## policy_value_networks/convnext_policy_value/train_convnext_zero.py

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from policy_value_networks.convnext_policy_value.convnext_zero_loop import (
    ConvNeXtTrainingConfig,
    run_convnext_training,
)
from zero_training_pipeline.weight_exports import export_checkpoint_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ConvNeXt NeoGoZero loop.")
    parser.add_argument("--board-size", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--self-play-games", type=int, default=1)
    parser.add_argument("--mcts-rounds", type=int, default=2)
    parser.add_argument("--mcts-inference-batch-size", type=int, default=1)
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
        default=Path("checkpoints_convnext_policy_value"),
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--self-play-records-path", type=Path, default=None)
    parser.add_argument("--max-training-seconds", type=float, default=None)
    parser.add_argument("--supervised-sgf-dir", type=Path, default=None)
    parser.add_argument("--supervised-steps", type=int, default=0)
    parser.add_argument("--supervised-max-seconds", type=float, default=None)
    parser.add_argument("--supervised-max-examples", type=int, default=None)
    parser.add_argument("--supervised-batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--weights-export-dir", type=Path, default=Path("trained_model_weights"))
    parser.add_argument("--skip-weights-export", action="store_true")
    parser.add_argument("--auto-download-weights", action="store_true")
    args = parser.parse_args()

    config = ConvNeXtTrainingConfig(
        board_size=args.board_size,
        iterations=args.iterations,
        self_play_games_per_iteration=args.self_play_games,
        mcts_rounds=args.mcts_rounds,
        mcts_inference_batch_size=args.mcts_inference_batch_size,
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
        self_play_records_path=args.self_play_records_path,
        max_training_seconds=args.max_training_seconds,
        supervised_sgf_dir=args.supervised_sgf_dir,
        supervised_steps=args.supervised_steps,
        supervised_max_seconds=args.supervised_max_seconds,
        supervised_max_examples=args.supervised_max_examples,
        supervised_batch_size=args.supervised_batch_size,
        seed=args.seed,
        device=args.device,
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
    if not result.iterations:
        print("No self-play iterations completed; exporting the latest checkpoint.")
    print(f"Final checkpoint: {result.final_checkpoint_path}")
    if not args.skip_weights_export:
        export = export_checkpoint_weights(
            checkpoint_path=result.final_checkpoint_path,
            architecture="convnext_policy_value",
            output_dir=args.weights_export_dir,
            auto_download=args.auto_download_weights,
        )
        print(f"Final weights: {export.weights_path}")
        print(f"Download bundle: {export.bundle_path}")
        if args.auto_download_weights and not export.auto_download_started:
            print("Auto-download is only available in Colab; bundle was still created.")


if __name__ == "__main__":
    main()
```

## policy_value_networks/convnext_policy_value/test_convnext_policy_value.py

```python
from __future__ import annotations

import random
import zipfile
import json

import pytest

torch = pytest.importorskip("torch")

from policy_value_networks.convnext_policy_value.convnext_policy_value import (
    ConvNeXtBlock,
    ConvNeXtPolicyValueEvaluator,
    ConvNeXtPolicyValueNet,
    LayerNorm2d,
    StochasticDepth,
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


def test_stochastic_depth_preserves_shape() -> None:
    layer = StochasticDepth(drop_prob=0.5)
    layer.train()
    inputs = torch.ones((4, 3, 3, 3), dtype=torch.float32)

    outputs = layer(inputs)

    assert outputs.shape == inputs.shape


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


def test_convnext_training_writes_self_play_records(tmp_path) -> None:
    records_path = tmp_path / "self_play_records.jsonl"
    config = ConvNeXtTrainingConfig(
        board_size=3,
        iterations=1,
        self_play_games_per_iteration=2,
        mcts_rounds=1,
        max_rollout_moves=6,
        training_steps_per_iteration=1,
        batch_size=4,
        channels=8,
        num_blocks=1,
        evaluation_games=0,
        checkpoint_dir=tmp_path / "checkpoints",
        self_play_records_path=records_path,
        seed=6,
    )

    run_convnext_training(config)
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 2
    assert records[0]["architecture"] == "convnext_policy_value"
    assert records[0]["mcts_rounds"] == 1
    assert records[0]["moves"]
```

## policy_value_networks/convnext_policy_value/README.md

```markdown
# ConvNeXt NeoGoZero Variant

This folder contains a separate policy-value implementation that replaces the
ResNet tower with a ConvNeXt-style tower.

## Research Notes

ConvNeXt modernizes ConvNets by borrowing design choices that made Vision
Transformers strong while staying fully convolutional. The key block design we
use here is:

1. depthwise 7x7 convolution
2. LayerNorm over channels
3. inverted bottleneck expansion, usually 4x
4. GELU activation
5. projection back to the original channel count
6. learnable layer scale
7. residual connection with optional stochastic depth

The original image ConvNeXt uses patch/downsampling stages. For Go, we keep a
single spatial resolution because the policy head must preserve one logit per
board point plus pass. This is the same reason the ResNet version keeps the
board grid intact.

## Files

- `convnext_policy_value.py`: ConvNeXt policy-value network and evaluator.
- `convnext_zero_loop.py`: repeatable training loop with checkpoints and metrics.
- `train_convnext_zero.py`: CLI entrypoint.
- `test_convnext_policy_value.py`: smoke and checkpoint tests.

## Smoke Run

```bash
python3 policy_value_networks/convnext_policy_value/train_convnext_zero.py
```

The command exports final weights and a zipped download bundle under
`trained_model_weights/convnext_policy_value/` when training finishes. In Colab,
add `--auto-download-weights` to trigger a browser download.

## Larger 9x9 Shape

```bash
python3 policy_value_networks/convnext_policy_value/train_convnext_zero.py \
  --board-size 9 \
  --history-length 8 \
  --channels 256 \
  --blocks 20 \
  --supervised-sgf-dir supervised_go_data/sgf_9x9 \
  --supervised-steps 1000 \
  --mcts-rounds 300 \
  --mcts-inference-batch-size 64 \
  --self-play-games 25 \
  --training-steps 1000 \
  --evaluation-games 20 \
  --device cuda
```
```

## play_and_train_commands/train_both_models_9x9_t4.py

```python
from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--target-hours-per-model", type=float, default=24.0)
    parser.add_argument("--supervised-budget-fraction", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--self-play-games", type=int, default=100)
    parser.add_argument("--mcts-rounds", type=int, default=300)
    parser.add_argument("--mcts-inference-batch-size", type=int, default=64)
    parser.add_argument("--training-steps", type=int, default=800)
    parser.add_argument("--evaluation-games", type=int, default=8)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-buffer-size", type=int, default=250_000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--supervised-sgf-dir", type=Path, default=Path("supervised_go_data/sgf_9x9"))
    parser.add_argument("--supervised-steps", type=int, default=None)
    parser.add_argument("--supervised-max-examples", type=int, default=None)
    parser.add_argument("--supervised-batch-size", type=int, default=128)
    parser.add_argument("--skip-supervised-pretraining", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.supervised_budget_fraction < 1:
        raise SystemExit("--supervised-budget-fraction must be in [0, 1)")
    supervised_steps = _supervised_steps(args)
    per_model_budget_seconds = int(args.target_hours_per_model * 3600)
    supervised_budget_seconds = int(
        per_model_budget_seconds * args.supervised_budget_fraction
    )

    if (
        not args.skip_supervised_pretraining
        and not args.supervised_sgf_dir.exists()
        and not args.dry_run
    ):
        raise SystemExit(
            "Supervised pretraining data is required before self-play. "
            f"Put 9x9 SGF files in {args.supervised_sgf_dir} "
            "or pass --supervised-sgf-dir /path/to/sgfs."
        )

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
        "--mcts-inference-batch-size",
        str(args.mcts_inference_batch_size),
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
        "--self-play-records-path",
        str(run_dir / "self_play_records_resnet_policy_value.jsonl"),
        "--max-training-seconds",
        str(per_model_budget_seconds),
        "--weights-export-dir",
        str(weights_dir),
        "--device",
        args.device,
        "--seed",
        "1",
    ]
    if not args.skip_supervised_pretraining:
        resnet_command.extend(
            [
                "--supervised-sgf-dir",
                str(args.supervised_sgf_dir),
                "--supervised-steps",
                str(supervised_steps),
                "--supervised-max-seconds",
                str(supervised_budget_seconds),
                "--supervised-batch-size",
                str(args.supervised_batch_size),
            ]
        )
        if args.supervised_max_examples is not None:
            resnet_command.extend(["--supervised-max-examples", str(args.supervised_max_examples)])

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
        "--mcts-inference-batch-size",
        str(args.mcts_inference_batch_size),
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
        "--self-play-records-path",
        str(run_dir / "self_play_records_convnext_policy_value.jsonl"),
        "--max-training-seconds",
        str(per_model_budget_seconds),
        "--weights-export-dir",
        str(weights_dir),
        "--device",
        args.device,
        "--seed",
        "2",
    ]
    if not args.skip_supervised_pretraining:
        convnext_command.extend(
            [
                "--supervised-sgf-dir",
                str(args.supervised_sgf_dir),
                "--supervised-steps",
                str(supervised_steps),
                "--supervised-max-seconds",
                str(supervised_budget_seconds),
                "--supervised-batch-size",
                str(args.supervised_batch_size),
            ]
        )
        if args.supervised_max_examples is not None:
            convnext_command.extend(["--supervised-max-examples", str(args.supervised_max_examples)])

    commands = [
        ("resnet_policy_value", resnet_command),
        ("convnext_policy_value", convnext_command),
    ]
    print(f"Run directory: {run_dir}")
    for name, command in commands:
        print(f"{name}: {' '.join(command)}")
    _write_run_manifest(
        run_dir=run_dir,
        args=args,
        commands=commands,
        weights_dir=weights_dir,
        supervised_steps=supervised_steps,
        supervised_budget_seconds=supervised_budget_seconds,
    )

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


def _write_run_manifest(
    run_dir: Path,
    args: argparse.Namespace,
    commands: list[tuple[str, list[str]]],
    weights_dir: Path,
    supervised_steps: int,
    supervised_budget_seconds: int,
) -> None:
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    total_self_play_games = args.iterations * args.self_play_games * len(commands)
    self_play_training_steps = args.iterations * args.training_steps
    supervised_fraction_of_gradient_steps = (
        supervised_steps / (supervised_steps + self_play_training_steps)
        if not args.skip_supervised_pretraining
        else 0.0
    )
    target_hours_total = args.target_hours_per_model * len(commands)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_hours_total": target_hours_total,
        "target_hours_per_model": args.target_hours_per_model,
        "supervised_budget_fraction_target": args.supervised_budget_fraction,
        "supervised_budget_seconds_per_model": supervised_budget_seconds,
        "supervised_steps": supervised_steps if not args.skip_supervised_pretraining else 0,
        "supervised_fraction_of_gradient_steps": supervised_fraction_of_gradient_steps,
        "notes": (
            "One GPU run: ResNet first, ConvNeXt second. Each model receives "
            "its own wall-clock budget. Supervised SGF pretraining is a short "
            "warm-up; most budget is self-play data generation."
        ),
        "default_data_targets": {
            "models": len(commands),
            "iterations_per_model": args.iterations,
            "self_play_games_per_iteration": args.self_play_games,
            "total_self_play_games_across_models": total_self_play_games,
            "mcts_rounds": args.mcts_rounds,
            "mcts_inference_batch_size": args.mcts_inference_batch_size,
            "replay_buffer_size": args.replay_buffer_size,
        },
        "artifacts": {
            "weights_dir": str(weights_dir),
            "resnet_metrics": str(run_dir / "metrics_resnet_policy_value.jsonl"),
            "convnext_metrics": str(run_dir / "metrics_convnext_policy_value.jsonl"),
            "resnet_self_play_records": str(run_dir / "self_play_records_resnet_policy_value.jsonl"),
            "convnext_self_play_records": str(run_dir / "self_play_records_convnext_policy_value.jsonl"),
            "logs_dir": str(run_dir / "logs"),
        },
        "commands": {name: command for name, command in commands},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Run manifest: {manifest_path}")


def _supervised_steps(args: argparse.Namespace) -> int:
    if args.skip_supervised_pretraining:
        return 0
    if args.supervised_steps is not None:
        return args.supervised_steps
    self_play_training_steps = args.iterations * args.training_steps
    if args.supervised_budget_fraction == 0:
        return 0
    return max(
        1,
        round(
            args.supervised_budget_fraction
            / (1 - args.supervised_budget_fraction)
            * self_play_training_steps
        ),
    )


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
```

## tests/test_encoding.py

```python
from __future__ import annotations

import pytest

from go_engine.board import Board
from go_engine.game import GameState, Move
from go_engine.types import Player, Point
from zero_training_pipeline.encoding import (
    encode_board_history,
    encode_game_state,
    encode_policy,
    index_to_move,
    move_to_index,
)
from zero_training_pipeline.torch_training import transform_board_planes, transform_policy


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
    assert planes[0][0][0] == 1
    assert planes[1][1][1] == 1
    assert planes[2][0][0] == 1


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


def test_dihedral_augmentation_rotates_board_planes_and_policy() -> None:
    planes = (
        (
            (1, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
        ),
    )
    policy = encode_policy({Move.play(Point(1, 1)): 1.0}, board_size=3)

    rotated_planes = transform_board_planes(planes, symmetry=1)
    rotated_policy = transform_policy(policy, board_size=3, symmetry=1)

    assert rotated_planes[0][0][2] == 1
    assert rotated_policy[2] == 1.0
    assert rotated_policy[-1] == 0.0


def test_dihedral_augmentation_preserves_pass_policy() -> None:
    policy = encode_policy({Move.pass_turn(): 1.0}, board_size=3)

    transformed = transform_policy(policy, board_size=3, symmetry=4)

    assert transformed[-1] == 1.0
```

## tests/test_match.py

```python
from __future__ import annotations

import random

import pytest

from search_players.random_bot import RandomBot
from match_evaluation.match import play_game, play_match
from go_engine.game import GameState, Move
from go_engine.types import Player, Point


class IllegalBot:
    def select_move(self, game_state: GameState) -> Move:
        return Move.play(Point(1, 1))


def test_play_game_returns_a_completed_result() -> None:
    result = play_game(
        RandomBot(rng=random.Random(1)),
        RandomBot(rng=random.Random(2)),
        board_size=3,
        max_moves=100,
    )

    assert result.final_state.is_over()
    assert result.num_moves > 0
    assert result.winner in {Player.BLACK, Player.WHITE}


def test_play_match_aggregates_results() -> None:
    result = play_match(
        black_bot_factory=lambda index: RandomBot(rng=random.Random(index)),
        white_bot_factory=lambda index: RandomBot(rng=random.Random(100 + index)),
        num_games=3,
        board_size=3,
        max_moves=100,
    )

    assert len(result.games) == 3
    assert result.black_wins + result.white_wins == 3
    assert result.average_moves > 0


def test_play_game_rejects_illegal_bot_moves() -> None:
    with pytest.raises(ValueError):
        play_game(
            IllegalBot(),
            RandomBot(rng=random.Random(1)),
            board_size=3,
            max_moves=10,
        )
```

## tests/test_mcts_bot.py

```python
from __future__ import annotations

import random

import pytest

from search_players.mcts_bot import (
    Evaluation,
    MCTSBot,
    MCTSNode,
    _select_move_from_visit_counts,
)
from search_players.random_bot import RandomBot
from go_engine.game import GameState, Move
from go_engine.types import Player, Point


def test_mcts_selects_a_legal_move() -> None:
    game = GameState.new_game(board_size=5)
    bot = MCTSBot(num_rounds=8, max_rollout_moves=20, rng=random.Random(1))

    move = bot.select_move(game)

    assert game.is_valid_move(move)


def test_mcts_can_play_a_small_game_against_random() -> None:
    game = GameState.new_game(board_size=3)
    bots = {
        Player.BLACK: MCTSBot(
            num_rounds=3,
            max_rollout_moves=12,
            rng=random.Random(1),
        ),
        Player.WHITE: RandomBot(rng=random.Random(2)),
    }

    for _ in range(100):
        if game.is_over():
            break
        move = bots[game.next_player].select_move(game)
        assert game.is_valid_move(move)
        game = game.apply_move(move)

    assert game.is_over()


def test_mcts_rejects_move_selection_after_game_over() -> None:
    game = GameState.new_game(board_size=3)
    game = game.apply_move(Move.pass_turn())
    game = game.apply_move(Move.pass_turn())
    bot = MCTSBot(num_rounds=1, rng=random.Random(1))

    with pytest.raises(ValueError):
        bot.select_move(game)


def test_puct_selection_uses_prior_probability() -> None:
    game = GameState.new_game(board_size=3)
    parent = MCTSNode(game_state=game)
    parent.num_rollouts = 10

    low_prior_child = MCTSNode(
        game_state=game.apply_move(Move.play(Point(1, 1))),
        parent=parent,
        move=Move.play(Point(1, 1)),
        prior_probability=0.1,
    )
    high_prior_child = MCTSNode(
        game_state=game.apply_move(Move.play(Point(2, 2))),
        parent=parent,
        move=Move.play(Point(2, 2)),
        prior_probability=0.9,
    )
    for child in (low_prior_child, high_prior_child):
        child.num_rollouts = 2
        child.value_sum = 0.0
    parent.children = [low_prior_child, high_prior_child]

    assert parent.select_child(c_puct=1.5) is high_prior_child


def test_evaluation_normalizes_legal_priors_and_clamps_value() -> None:
    game = GameState.new_game(board_size=3)
    move_a = Move.play(Point(1, 1))
    move_b = Move.play(Point(1, 2))
    illegal_move = Move.play(Point(4, 4))

    evaluation = Evaluation.from_priors(
        game,
        {
            move_a: 2.0,
            move_b: 1.0,
            illegal_move: 100.0,
        },
        value=3.0,
    )

    assert evaluation.move_priors == {
        move_a: pytest.approx(2 / 3),
        move_b: pytest.approx(1 / 3),
    }
    assert evaluation.value == 1.0


def test_mcts_can_use_a_custom_evaluator_prior() -> None:
    class CenterPriorEvaluator:
        def evaluate(self, game_state: GameState) -> Evaluation:
            center = Move.play(Point(2, 2))
            return Evaluation.from_priors(game_state, {center: 1.0}, value=0.0)

    game = GameState.new_game(board_size=3)
    bot = MCTSBot(num_rounds=2, evaluator=CenterPriorEvaluator())

    assert bot.select_move(game) == Move.play(Point(2, 2))


def test_mcts_uses_batched_evaluator_when_configured() -> None:
    class CountingBatchEvaluator:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def evaluate(self, game_state: GameState) -> Evaluation:
            return self.evaluate_many((game_state,))[0]

        def evaluate_many(self, game_states: tuple[GameState, ...]) -> tuple[Evaluation, ...]:
            self.batch_sizes.append(len(game_states))
            return tuple(Evaluation.uniform(game_state, value=0.0) for game_state in game_states)

    evaluator = CountingBatchEvaluator()
    game = GameState.new_game(board_size=3)
    bot = MCTSBot(
        num_rounds=9,
        evaluator=evaluator,
        inference_batch_size=4,
        rng=random.Random(1),
    )

    move = bot.select_move(game)

    assert game.is_valid_move(move)
    assert max(evaluator.batch_sizes) > 1


def test_temperature_zero_selects_most_visited_move() -> None:
    move_a = Move.play(Point(1, 1))
    move_b = Move.play(Point(1, 2))

    selected = _select_move_from_visit_counts(
        {move_a: 1, move_b: 5},
        temperature=0.0,
        rng=random.Random(1),
    )

    assert selected == move_b


def test_dirichlet_noise_changes_root_priors() -> None:
    game = GameState.new_game(board_size=3)
    root = MCTSNode(game_state=game)
    root.expand(Evaluation.uniform(game, value=0.0).move_priors)
    before = [child.prior_probability for child in root.children]

    root.add_dirichlet_noise(alpha=0.3, epsilon=1.0, rng=random.Random(1))
    after = [child.prior_probability for child in root.children]

    assert after != before
    assert sum(after) == pytest.approx(1.0)
```

## tests/test_rules.py

```python
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
```

## tests/test_self_play.py

```python
from __future__ import annotations

import random

from search_players.mcts_bot import MCTSBot
from zero_training_pipeline.self_play import generate_self_play_game


def test_self_play_generates_training_examples() -> None:
    bot = MCTSBot(
        num_rounds=3,
        max_rollout_moves=8,
        rng=random.Random(1),
    )

    game = generate_self_play_game(bot, board_size=3, max_moves=30)

    assert game.final_state.is_over()
    assert len(game.examples) == len(game.moves)
    assert game.examples
    assert game.examples[0].winner is game.winner
    assert game.examples[0].value in {-1.0, 1.0}
    assert sum(game.examples[0].visit_distribution.values()) == 1.0


def test_self_play_records_board_history() -> None:
    bot = MCTSBot(
        num_rounds=2,
        max_rollout_moves=8,
        rng=random.Random(2),
    )

    game = generate_self_play_game(bot, board_size=3, max_moves=12, history_length=2)

    assert game.examples[0].board_history
    assert len(game.examples[-1].board_history) <= 2
```

## tests/test_supervised_pretraining.py

```python
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

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
```

## tests/test_zero_loop.py

```python
from __future__ import annotations

import random
import zipfile
import json

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


def test_zero_training_writes_self_play_records(tmp_path) -> None:
    records_path = tmp_path / "self_play_records.jsonl"
    config = ZeroTrainingConfig(
        board_size=3,
        iterations=1,
        self_play_games_per_iteration=2,
        mcts_rounds=1,
        max_rollout_moves=6,
        training_steps_per_iteration=1,
        batch_size=4,
        channels=8,
        num_res_blocks=1,
        checkpoint_dir=tmp_path / "checkpoints",
        self_play_records_path=records_path,
        seed=8,
    )

    run_zero_training(config)
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 2
    assert records[0]["architecture"] == "resnet_policy_value"
    assert records[0]["mcts_rounds"] == 1
    assert records[0]["moves"]
```
