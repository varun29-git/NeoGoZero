from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from myalphago.go.game import GameState, Move
from myalphago.go.types import Player


@dataclass
class MCTSNode:
    game_state: GameState
    parent: MCTSNode | None = None
    move: Move | None = None
    win_counts: dict[Player, int] = field(
        default_factory=lambda: {Player.BLACK: 0, Player.WHITE: 0}
    )
    num_rollouts: int = 0
    children: list[MCTSNode] = field(default_factory=list)
    unvisited_moves: list[Move] = field(init=False)

    def __post_init__(self) -> None:
        self.unvisited_moves = list(self.game_state.legal_moves())

    def can_add_child(self) -> bool:
        return bool(self.unvisited_moves)

    def add_random_child(self, rng: random.Random) -> MCTSNode:
        move_index = rng.randrange(len(self.unvisited_moves))
        move = self.unvisited_moves.pop(move_index)
        child = MCTSNode(
            game_state=self.game_state.apply_move(move),
            parent=self,
            move=move,
        )
        self.children.append(child)
        return child

    def record_win(self, winner: Player) -> None:
        self.win_counts[winner] += 1
        self.num_rollouts += 1

    def winning_fraction(self, player: Player) -> float:
        if self.num_rollouts == 0:
            return 0.0
        return self.win_counts[player] / self.num_rollouts

    def select_child(self, exploration_weight: float) -> MCTSNode:
        if not self.children:
            raise ValueError("cannot select a child from a leaf node")

        total_rollouts = max(self.num_rollouts, 1)
        log_total = math.log(total_rollouts)
        player = self.game_state.next_player

        def uct_score(child: MCTSNode) -> float:
            if child.num_rollouts == 0:
                return math.inf
            exploitation = child.winning_fraction(player)
            exploration = exploration_weight * math.sqrt(log_total / child.num_rollouts)
            return exploitation + exploration

        return max(self.children, key=uct_score)

    def most_visited_child(self) -> MCTSNode:
        if not self.children:
            raise ValueError("cannot choose a move from a leaf node")
        return max(self.children, key=lambda child: child.num_rollouts)


@dataclass
class MCTSBot:
    num_rounds: int = 100
    exploration_weight: float = math.sqrt(2)
    max_rollout_moves: int | None = None
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.num_rounds < 1:
            raise ValueError("num_rounds must be at least 1")
        if self.exploration_weight < 0:
            raise ValueError("exploration_weight cannot be negative")
        if self.max_rollout_moves is not None and self.max_rollout_moves < 1:
            raise ValueError("max_rollout_moves must be at least 1")

    def select_move(self, game_state: GameState) -> Move:
        legal_moves = game_state.legal_moves()
        if not legal_moves:
            raise ValueError("cannot select a move after the game is over")
        if len(legal_moves) == 1:
            return legal_moves[0]

        root = MCTSNode(game_state)

        for _ in range(self.num_rounds):
            node = root

            while not node.can_add_child() and not node.game_state.is_over():
                node = node.select_child(self.exploration_weight)

            if node.can_add_child():
                node = node.add_random_child(self.rng)

            winner = self._simulate_random_game(node.game_state)

            while node is not None:
                node.record_win(winner)
                node = node.parent

        best_child = root.most_visited_child()
        assert best_child.move is not None
        return best_child.move

    def _simulate_random_game(self, game_state: GameState) -> Player:
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

        return rollout_state.winner()
