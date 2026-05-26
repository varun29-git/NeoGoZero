from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myalphago.bots.random_bot import RandomBot
from myalphago.go.game import GameState
from myalphago.go.types import Player


def main() -> None:
    game = GameState.new_game(board_size=9)
    bots = {
        Player.BLACK: RandomBot(),
        Player.WHITE: RandomBot(),
    }

    while not game.is_over():
        move = bots[game.next_player].select_move(game)
        game = game.apply_move(move)

    score = game.score()
    print(game.board)
    print(f"Black: {score.black:.1f}")
    print(f"White: {score.white:.1f}")
    print(f"Winner: {game.winner().value}")


if __name__ == "__main__":
    main()
