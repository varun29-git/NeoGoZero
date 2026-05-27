from __future__ import annotations

from collections.abc import Sequence
import random
from typing import Protocol

import torch
import torch.nn.functional as F

from zero_training_pipeline.encoding import BoardPlanes, PolicyVector
from zero_training_pipeline.encoding import encode_board_snapshot, encode_policy
from zero_training_pipeline.self_play import TrainingExample


class PolicyValueModel(Protocol):
    def train(self, mode: bool = True) -> PolicyValueModel:
        ...

    def __call__(self, board_planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...


def examples_to_tensors(
    examples: Sequence[TrainingExample],
    board_size: int,
    history_length: int = 1,
    device: torch.device | str = "cpu",
    augment: bool = False,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boards = []
    policies = []
    augmentation_rng = rng or random
    for example in examples:
        board_planes = encode_board_snapshot(
            example.board,
            example.player,
            board_size,
            history_length=history_length,
            board_history=example.board_history or (example.board,),
        )
        policy = encode_policy(example.visit_distribution, board_size)
        if augment:
            symmetry = augmentation_rng.randrange(8)
            board_planes = transform_board_planes(board_planes, symmetry)
            policy = transform_policy(policy, board_size, symmetry)
        boards.append(board_planes)
        policies.append(policy)
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
    model: PolicyValueModel,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[TrainingExample],
    board_size: int,
    history_length: int = 1,
    device: torch.device | str = "cpu",
    augment: bool = True,
    rng: random.Random | None = None,
) -> float:
    model.train()
    boards, target_policies, target_values = examples_to_tensors(
        examples,
        board_size=board_size,
        history_length=history_length,
        device=device,
        augment=augment,
        rng=rng,
    )

    optimizer.zero_grad()
    policy_logits, values = model(boards)
    loss = policy_value_loss(policy_logits, values, target_policies, target_values)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def transform_board_planes(
    board_planes: BoardPlanes,
    symmetry: int,
) -> BoardPlanes:
    if not 0 <= symmetry < 8:
        raise ValueError("symmetry must be in [0, 7]")
    size = len(board_planes[0])
    transformed_planes = []
    for plane in board_planes:
        transformed = [[0 for _ in range(size)] for _ in range(size)]
        for row in range(size):
            for col in range(size):
                new_row, new_col = _transform_coords(row, col, size, symmetry)
                transformed[new_row][new_col] = plane[row][col]
        transformed_planes.append(tuple(tuple(row_values) for row_values in transformed))
    return tuple(transformed_planes)


def transform_policy(
    policy: PolicyVector,
    board_size: int,
    symmetry: int,
) -> PolicyVector:
    if len(policy) != board_size * board_size + 1:
        raise ValueError("policy length does not match board size")

    transformed = [0.0 for _ in policy]
    for row in range(board_size):
        for col in range(board_size):
            old_index = row * board_size + col
            new_row, new_col = _transform_coords(row, col, board_size, symmetry)
            new_index = new_row * board_size + new_col
            transformed[new_index] = policy[old_index]
    transformed[-1] = policy[-1]
    return tuple(transformed)


def _transform_coords(
    row: int,
    col: int,
    size: int,
    symmetry: int,
) -> tuple[int, int]:
    if symmetry == 0:
        return row, col
    if symmetry == 1:
        return col, size - 1 - row
    if symmetry == 2:
        return size - 1 - row, size - 1 - col
    if symmetry == 3:
        return size - 1 - col, row
    if symmetry == 4:
        return row, size - 1 - col
    if symmetry == 5:
        return size - 1 - row, col
    if symmetry == 6:
        return col, row
    if symmetry == 7:
        return size - 1 - col, size - 1 - row
    raise ValueError("symmetry must be in [0, 7]")
