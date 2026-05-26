from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neogozero_core.bots.mcts_bot import MCTSBot
from neogozero_core.bots.random_bot import RandomBot
from neogozero_core.go.game import GameState
from neogozero_core.go.types import Player


def main() -> None:
    game = GameState.new_game(board_size=5)
    bots = {
        Player.BLACK: MCTSBot(
            num_rounds=6,
            max_rollout_moves=25,
            rng=random.Random(1),
        ),
        Player.WHITE: RandomBot(rng=random.Random(2)),
    }

    turn = 0
    max_turns = 100
    while not game.is_over() and turn < max_turns:
        move = bots[game.next_player].select_move(game)
        game = game.apply_move(move)
        turn += 1

    score = game.score()
    print(game.board)
    print(f"Turns: {turn}")
    print(f"Black MCTS: {score.black:.1f}")
    print(f"White random: {score.white:.1f}")
    print(f"Winner: {game.winner().value}")


if __name__ == "__main__":
    main()
