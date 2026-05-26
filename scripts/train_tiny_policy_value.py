from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from myalphago.bots.mcts_bot import MCTSBot
from myalphago.models.policy_value import PolicyValueNet, TorchPolicyValueEvaluator
from myalphago.training.self_play import generate_self_play_game
from myalphago.training.torch_training import train_step


def main() -> None:
    board_size = 3
    search_bot = MCTSBot(
        num_rounds=4,
        max_rollout_moves=10,
        rng=random.Random(11),
    )
    self_play_game = generate_self_play_game(
        search_bot,
        board_size=board_size,
        max_moves=board_size * board_size * 3,
    )

    model = PolicyValueNet(board_size=board_size, channels=16, num_res_blocks=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for step in range(1, 4):
        loss = train_step(
            model,
            optimizer,
            self_play_game.examples,
            board_size=board_size,
        )
        print(f"Step {step}: loss={loss:.4f}")

    evaluator = TorchPolicyValueEvaluator(model)
    neural_bot = MCTSBot(num_rounds=4, evaluator=evaluator, rng=random.Random(12))
    move = neural_bot.select_move(self_play_game.final_state.previous_state)
    print(f"Neural-PUCT sample move: {move}")


if __name__ == "__main__":
    main()
