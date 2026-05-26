from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from myalphago.bots.mcts_bot import Evaluation
from myalphago.go.game import GameState
from myalphago.training.encoding import encode_game_state, move_to_index


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        board_size: int = 9,
        input_planes: int = 3,
        channels: int = 64,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.policy_size = board_size * board_size + 1

        self.body = nn.Sequential(
            nn.Conv2d(input_planes, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, self.policy_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, board_planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(board_planes)
        policy_logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return policy_logits, values


@dataclass
class TorchPolicyValueEvaluator:
    model: PolicyValueNet
    device: torch.device | str = "cpu"

    def evaluate(self, game_state: GameState) -> Evaluation:
        self.model.eval()
        board_planes = torch.tensor(
            encode_game_state(game_state),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            policy_logits, values = self.model(board_planes)
            probabilities = torch.softmax(policy_logits, dim=1)[0]

        move_priors = {
            move: float(probabilities[move_to_index(move, game_state.board.size)].item())
            for move in game_state.legal_moves()
        }
        return Evaluation.from_priors(
            game_state,
            move_priors=move_priors,
            value=float(values[0].item()),
        )
