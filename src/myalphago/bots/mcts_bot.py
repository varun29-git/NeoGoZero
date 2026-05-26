from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

from myalphago.go.game import GameState, Move
from myalphago.go.types import Player


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

        parent_visits = math.sqrt(max(self.num_rollouts, 1))

        def puct_score(child: MCTSNode) -> float:
            value_score = -child.q_value()
            prior_score = (
                c_puct
                * child.prior_probability
                * parent_visits
                / (1 + child.num_rollouts)
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

        for _ in range(self.num_rounds):
            node = root

            while node.is_expanded() and not node.game_state.is_over():
                node = node.select_child(self.c_puct)

            assert self.evaluator is not None
            evaluation = self.evaluator.evaluate(node.game_state)
            node.expand(evaluation.move_priors)
            if (
                add_dirichlet_noise
                and node is root
                and not added_root_noise
                and root.children
            ):
                root.add_dirichlet_noise(
                    alpha=self.dirichlet_alpha,
                    epsilon=self.dirichlet_epsilon,
                    rng=self.rng,
                )
                added_root_noise = True

            value = evaluation.value
            while node is not None:
                node.record_visit(value)
                value = -value
                node = node.parent

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
