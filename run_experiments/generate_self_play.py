from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search_players.mcts_bot import MCTSBot
from zero_training_pipeline.self_play import generate_self_play_game


def main() -> None:
    board_size = 3
    bot = MCTSBot(
        num_rounds=6,
        max_rollout_moves=12,
        rng=random.Random(7),
    )
    game = generate_self_play_game(
        bot,
        board_size=board_size,
        max_moves=board_size * board_size * 3,
    )

    print(f"Board size: {board_size}x{board_size}")
    print(f"Moves: {len(game.moves)}")
    print(f"Training examples: {len(game.examples)}")
    print(f"Winner: {game.winner.value}")
    print(f"First example value: {game.examples[0].value:.1f}")


if __name__ == "__main__":
    main()
