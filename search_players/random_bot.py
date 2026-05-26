from __future__ import annotations

import random
from dataclasses import dataclass, field

from go_engine.game import GameState, Move


@dataclass
class RandomBot:
    rng: random.Random = field(default_factory=random.Random)

    def select_move(self, game_state: GameState) -> Move:
        return self.rng.choice(game_state.legal_moves())
