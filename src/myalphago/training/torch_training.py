from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from myalphago.models.policy_value import PolicyValueNet
from myalphago.training.encoding import encode_board_snapshot, encode_policy
from myalphago.training.self_play import TrainingExample


def examples_to_tensors(
    examples: Sequence[TrainingExample],
    board_size: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boards = [
        encode_board_snapshot(example.board, example.player, board_size)
        for example in examples
    ]
    policies = [
        encode_policy(example.visit_distribution, board_size)
        for example in examples
    ]
    values = [example.value for example in examples]

    return (
        torch.tensor(boards, dtype=torch.float32, device=device),
        torch.tensor(policies, dtype=torch.float32, device=device),
        torch.tensor(values, dtype=torch.float32, device=device),
    )


def policy_value_loss(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
) -> torch.Tensor:
    policy_loss = -(target_policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1)
    value_loss = F.mse_loss(values, target_values, reduction="none")
    return (policy_loss + value_loss).mean()


def train_step(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[TrainingExample],
    board_size: int,
    device: torch.device | str = "cpu",
) -> float:
    model.train()
    boards, target_policies, target_values = examples_to_tensors(
        examples,
        board_size=board_size,
        device=device,
    )

    optimizer.zero_grad()
    policy_logits, values = model(boards)
    loss = policy_value_loss(policy_logits, values, target_policies, target_values)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())
