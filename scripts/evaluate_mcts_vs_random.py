from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neogozero_core.bots.mcts_bot import MCTSBot
from neogozero_core.bots.random_bot import RandomBot
from neogozero_core.evaluation.match import play_match
from neogozero_core.go.types import Player


def main() -> None:
    num_games = 3
    board_size = 3

    result = play_match(
        black_bot_factory=lambda index: MCTSBot(
            num_rounds=4,
            max_rollout_moves=12,
            rng=random.Random(100 + index),
        ),
        white_bot_factory=lambda index: RandomBot(rng=random.Random(200 + index)),
        num_games=num_games,
        board_size=board_size,
        max_moves=board_size * board_size * 3,
    )

    print(f"Games: {len(result.games)}")
    print(f"Black MCTS wins: {result.black_wins}")
    print(f"White random wins: {result.white_wins}")
    print(f"Black win rate: {result.win_rate(Player.BLACK):.1%}")
    print(f"Average moves: {result.average_moves:.1f}")
    print(f"Average margin: {result.average_margin:.1f}")


if __name__ == "__main__":
    main()
