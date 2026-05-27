from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import nn

from search_players.mcts_bot import Evaluation
from go_engine.game import GameState
from zero_training_pipeline.encoding import encode_game_state, move_to_index


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.permute(0, 2, 3, 1)
        outputs = self.layer_norm(outputs)
        return outputs.permute(0, 3, 1, 2)


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob must be in [0, 1)")
        self.drop_prob = drop_prob

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return inputs

        keep_prob = 1 - self.drop_prob
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        mask = inputs.new_empty(shape).bernoulli_(keep_prob)
        return inputs * mask / keep_prob


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        layer_scale_init: float = 1e-6,
        stochastic_depth_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        hidden_channels = channels * mlp_ratio
        self.depthwise_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = LayerNorm2d(channels)
        self.pointwise_expand = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.pointwise_project = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.layer_scale = nn.Parameter(
            torch.ones(channels, 1, 1) * layer_scale_init
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.depthwise_conv(inputs)
        outputs = self.norm(outputs)
        outputs = self.pointwise_expand(outputs)
        outputs = self.activation(outputs)
        outputs = self.pointwise_project(outputs)
        outputs = self.layer_scale * outputs
        outputs = self.stochastic_depth(outputs)
        return residual + outputs


class ConvNeXtPolicyValueNet(nn.Module):
    def __init__(
        self,
        board_size: int = 9,
        input_planes: int = 17,
        channels: int = 256,
        num_blocks: int = 20,
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        layer_scale_init: float = 1e-6,
        stochastic_depth_prob: float = 0.1,
    ) -> None:
        super().__init__()
        if input_planes < 1 or input_planes % 2 == 0:
            raise ValueError("input_planes must be a positive odd number")
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")

        self.board_size = board_size
        self.policy_size = board_size * board_size + 1
        self.input_planes = input_planes
        self.channels = channels
        self.num_blocks = num_blocks

        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, channels, kernel_size=3, padding=1),
            LayerNorm2d(channels),
        )
        block_drop_probs = torch.linspace(0, stochastic_depth_prob, num_blocks).tolist()
        self.convnext_tower = nn.Sequential(
            *(
                ConvNeXtBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    layer_scale_init=layer_scale_init,
                    stochastic_depth_prob=block_drop_probs[index],
                )
                for index in range(num_blocks)
            )
        )
        self.final_norm = LayerNorm2d(channels)
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, self.policy_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, channels),
            nn.GELU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, board_planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stem(board_planes)
        features = self.convnext_tower(features)
        features = self.final_norm(features)
        policy_logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return policy_logits, values


@dataclass
class ConvNeXtPolicyValueEvaluator:
    model: ConvNeXtPolicyValueNet
    device: torch.device | str = "cpu"

    def evaluate(self, game_state: GameState) -> Evaluation:
        return self.evaluate_many((game_state,))[0]

    def evaluate_many(self, game_states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        if not game_states:
            return ()

        self.model.eval()
        history_length = (self.model.input_planes - 1) // 2
        board_planes = torch.tensor(
            [
                encode_game_state(game_state, history_length=history_length)
                for game_state in game_states
            ],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            policy_logits, values = self.model(board_planes)
            probabilities = torch.softmax(policy_logits, dim=1)

        evaluations = []
        for index, game_state in enumerate(game_states):
            move_priors = {
                move: float(probabilities[index, move_to_index(move, game_state.board.size)].item())
                for move in game_state.legal_moves()
            }
            evaluations.append(
                Evaluation.from_priors(
                    game_state,
                    move_priors=move_priors,
                    value=float(values[index].item()),
                )
            )
        return tuple(evaluations)
