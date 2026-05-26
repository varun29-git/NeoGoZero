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
