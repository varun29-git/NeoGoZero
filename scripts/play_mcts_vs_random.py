from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myalphago.bots.mcts_bot import MCTSBot
from myalphago.bots.random_bot import RandomBot
from myalphago.go.game import GameState
from myalphago.go.types import Player


def main() -> None:
    game = GameState.new_game(board_size=9)
    bots = {
        Player.BLACK: MCTSBot(
            num_rounds=40,
            max_rollout_moves=120,
            rng=random.Random(1),
        ),
        Player.WHITE: RandomBot(rng=random.Random(2)),
    }

    turn = 0
    max_turns = 300
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
