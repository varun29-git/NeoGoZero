from __future__ import annotations

from dataclasses import dataclass

from neogozero_core.go.board import Board
from neogozero_core.go.types import Player, Point


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
        next_situation = (self.next_player.other, next_board.zobrist_key())
        state = self.previous_state
        while state is not None:
            if (state.next_player, state.board.zobrist_key()) == next_situation:
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
