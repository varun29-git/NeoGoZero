from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from neogozero_core.go.types import Player, Point


@dataclass(frozen=True)
class Board:
    size: int = 9
    grid: frozenset[tuple[Point, Player]] = frozenset()

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("board size must be at least 2")
        points = [point for point, _ in self.grid]
        if len(points) != len(set(points)):
            raise ValueError("a point cannot contain more than one stone")
        for point, _ in self.grid:
            if not self.is_on_grid(point):
                raise ValueError(f"point is outside the board: {point}")

    @classmethod
    def from_grid(
        cls, size: int, stones: Iterable[tuple[Point, Player]]
    ) -> Board:
        return cls(size=size, grid=frozenset(stones))

    def is_on_grid(self, point: Point) -> bool:
        return 1 <= point.row <= self.size and 1 <= point.col <= self.size

    def get(self, point: Point) -> Player | None:
        for stone_point, player in self.grid:
            if stone_point == point:
                return player
        return None

    def is_empty(self, point: Point) -> bool:
        return self.get(point) is None

    def place_stone(self, player: Player, point: Point) -> Board:
        if not self.is_on_grid(point):
            raise ValueError(f"point is outside the board: {point}")
        if not self.is_empty(point):
            raise ValueError(f"point is already occupied: {point}")

        stones = dict(self.grid)
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

    def zobrist_key(self) -> tuple[tuple[int, int, str], ...]:
        return tuple(
            sorted((point.row, point.col, player.value) for point, player in self.grid)
        )

    def __str__(self) -> str:
        rows = []
        for row in range(1, self.size + 1):
            cells = []
            for col in range(1, self.size + 1):
                player = self.get(Point(row, col))
                cells.append(player.marker if player else ".")
            rows.append(" ".join(cells))
        return "\n".join(rows)
