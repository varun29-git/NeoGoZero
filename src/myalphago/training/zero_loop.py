from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from myalphago.bots.mcts_bot import MCTSBot
from myalphago.evaluation.match import play_game
from myalphago.go.types import Player
from myalphago.models.policy_value import PolicyValueNet, TorchPolicyValueEvaluator
from myalphago.training.self_play import TrainingExample, generate_self_play_game
from myalphago.training.torch_training import train_step


@dataclass(frozen=True)
class ZeroTrainingConfig:
    board_size: int = 3
    iterations: int = 1
    self_play_games_per_iteration: int = 1
    mcts_rounds: int = 2
    max_rollout_moves: int | None = 12
    training_steps_per_iteration: int = 2
    batch_size: int = 8
    learning_rate: float = 0.01
    channels: int = 16
    num_res_blocks: int = 2
    history_length: int = 1
    replay_buffer_size: int = 1_000
    evaluation_games: int = 0
    promotion_threshold: float = 0.55
    self_play_temperature: float = 1.0
    temperature_drop_move: int = 30
    dirichlet_alpha: float = 0.03
    dirichlet_epsilon: float = 0.25
    checkpoint_dir: Path = Path("checkpoints")
    resume_checkpoint: Path | None = None
    metrics_path: Path | None = None
    seed: int = 1
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.board_size < 2:
            raise ValueError("board_size must be at least 2")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.self_play_games_per_iteration < 1:
            raise ValueError("self_play_games_per_iteration must be at least 1")
        if self.mcts_rounds < 1:
            raise ValueError("mcts_rounds must be at least 1")
        if self.max_rollout_moves is not None and self.max_rollout_moves < 1:
            raise ValueError("max_rollout_moves must be at least 1")
        if self.training_steps_per_iteration < 1:
            raise ValueError("training_steps_per_iteration must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.channels < 1:
            raise ValueError("channels must be at least 1")
        if self.num_res_blocks < 1:
            raise ValueError("num_res_blocks must be at least 1")
        if self.history_length < 1:
            raise ValueError("history_length must be at least 1")
        if self.replay_buffer_size < 1:
            raise ValueError("replay_buffer_size must be at least 1")
        if self.evaluation_games < 0:
            raise ValueError("evaluation_games cannot be negative")
        if not 0 <= self.promotion_threshold <= 1:
            raise ValueError("promotion_threshold must be in [0, 1]")
        if self.self_play_temperature < 0:
            raise ValueError("self_play_temperature cannot be negative")
        if self.temperature_drop_move < 0:
            raise ValueError("temperature_drop_move cannot be negative")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError("dirichlet_epsilon must be in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["checkpoint_dir"] = str(self.checkpoint_dir)
        data["resume_checkpoint"] = (
            str(self.resume_checkpoint) if self.resume_checkpoint is not None else None
        )
        data["metrics_path"] = str(self.metrics_path) if self.metrics_path is not None else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ZeroTrainingConfig:
        converted = dict(data)
        converted["checkpoint_dir"] = Path(str(converted["checkpoint_dir"]))
        if converted.get("resume_checkpoint") is not None:
            converted["resume_checkpoint"] = Path(str(converted["resume_checkpoint"]))
        if converted.get("metrics_path") is not None:
            converted["metrics_path"] = Path(str(converted["metrics_path"]))
        return cls(**converted)


@dataclass
class ReplayBuffer:
    capacity: int
    examples: list[TrainingExample] = field(default_factory=list)

    def add_examples(self, examples: list[TrainingExample] | tuple[TrainingExample, ...]) -> None:
        self.examples.extend(examples)
        if len(self.examples) > self.capacity:
            self.examples = self.examples[-self.capacity :]

    def sample(self, batch_size: int, rng: random.Random) -> list[TrainingExample]:
        if not self.examples:
            raise ValueError("cannot sample from an empty replay buffer")
        if batch_size <= len(self.examples):
            return rng.sample(self.examples, batch_size)
        return [rng.choice(self.examples) for _ in range(batch_size)]

    def __len__(self) -> int:
        return len(self.examples)


@dataclass(frozen=True)
class TrainingIterationResult:
    iteration: int
    generated_examples: int
    mean_loss: float
    candidate_win_rate: float
    promoted: bool
    checkpoint_path: Path


@dataclass(frozen=True)
class TrainingRunResult:
    iterations: tuple[TrainingIterationResult, ...]
    final_checkpoint_path: Path


def run_zero_training(config: ZeroTrainingConfig) -> TrainingRunResult:
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_iteration = 0
    if config.resume_checkpoint is not None:
        model, optimizer, loaded_config, replay_buffer, start_iteration = load_checkpoint(
            config.resume_checkpoint
        )
        _ensure_resume_compatible(config, loaded_config)
        replay_buffer.capacity = config.replay_buffer_size
    else:
        model = _new_model(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        replay_buffer = ReplayBuffer(capacity=config.replay_buffer_size)

    iteration_results: list[TrainingIterationResult] = []

    for offset in range(1, config.iterations + 1):
        iteration = start_iteration + offset
        champion_model = _clone_model(model, config)
        generated_examples = _run_self_play(config, model, replay_buffer, rng)
        losses = _train_from_replay(config, model, optimizer, replay_buffer, rng)

        candidate_win_rate = _evaluate_candidate(
            config=config,
            candidate_model=model,
            champion_model=champion_model,
            rng=rng,
        )
        promoted = candidate_win_rate >= config.promotion_threshold
        if not promoted:
            model.load_state_dict(champion_model.state_dict())

        checkpoint_path = save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            iteration=iteration,
            model=model,
            optimizer=optimizer,
            config=config,
            replay_buffer=replay_buffer,
            promoted=promoted,
            candidate_win_rate=candidate_win_rate,
        )
        iteration_results.append(
            TrainingIterationResult(
                iteration=iteration,
                generated_examples=generated_examples,
                mean_loss=sum(losses) / len(losses),
                candidate_win_rate=candidate_win_rate,
                promoted=promoted,
                checkpoint_path=checkpoint_path,
            )
        )
        _write_metrics(config, iteration_results[-1], len(replay_buffer))

    return TrainingRunResult(
        iterations=tuple(iteration_results),
        final_checkpoint_path=iteration_results[-1].checkpoint_path,
    )


def save_checkpoint(
    checkpoint_dir: Path,
    iteration: int,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    config: ZeroTrainingConfig,
    replay_buffer: ReplayBuffer,
    promoted: bool,
    candidate_win_rate: float,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"iteration_{iteration:04d}.pt"
    torch.save(
        {
            "iteration": iteration,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config.to_dict(),
            "replay_examples": replay_buffer.examples,
            "promoted": promoted,
            "candidate_win_rate": candidate_win_rate,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
) -> tuple[PolicyValueNet, torch.optim.Optimizer, ZeroTrainingConfig, ReplayBuffer, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = ZeroTrainingConfig.from_dict(checkpoint["config"])
    model = _new_model(config)
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    replay_buffer = ReplayBuffer(
        capacity=config.replay_buffer_size,
        examples=list(checkpoint["replay_examples"]),
    )
    return model, optimizer, config, replay_buffer, int(checkpoint["iteration"])


def _run_self_play(
    config: ZeroTrainingConfig,
    model: PolicyValueNet,
    replay_buffer: ReplayBuffer,
    rng: random.Random,
) -> int:
    generated_examples = 0
    evaluator = TorchPolicyValueEvaluator(model, device=config.device)

    for _ in range(config.self_play_games_per_iteration):
        bot = MCTSBot(
            num_rounds=config.mcts_rounds,
            max_rollout_moves=config.max_rollout_moves,
            evaluator=evaluator,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            rng=random.Random(rng.randrange(1_000_000_000)),
        )
        game = generate_self_play_game(
            bot,
            board_size=config.board_size,
            max_moves=config.board_size * config.board_size * 3,
            history_length=config.history_length,
            temperature=config.self_play_temperature,
            temperature_drop_move=config.temperature_drop_move,
            add_dirichlet_noise=True,
        )
        replay_buffer.add_examples(game.examples)
        generated_examples += len(game.examples)

    return generated_examples


def _train_from_replay(
    config: ZeroTrainingConfig,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    rng: random.Random,
) -> list[float]:
    losses = []
    for _ in range(config.training_steps_per_iteration):
        batch = replay_buffer.sample(config.batch_size, rng)
        losses.append(
            train_step(
                model=model,
                optimizer=optimizer,
                examples=batch,
                board_size=config.board_size,
                history_length=config.history_length,
                device=config.device,
            )
        )
    return losses


def _evaluate_candidate(
    config: ZeroTrainingConfig,
    candidate_model: PolicyValueNet,
    champion_model: PolicyValueNet,
    rng: random.Random,
) -> float:
    if config.evaluation_games == 0:
        return 1.0

    candidate_wins = 0
    total_games = config.evaluation_games * 2
    candidate_evaluator = TorchPolicyValueEvaluator(candidate_model, device=config.device)
    champion_evaluator = TorchPolicyValueEvaluator(champion_model, device=config.device)

    for game_index in range(config.evaluation_games):
        candidate_black = MCTSBot(
            num_rounds=config.mcts_rounds,
            evaluator=candidate_evaluator,
            rng=random.Random(rng.randrange(1_000_000_000)),
        )
        champion_white = MCTSBot(
            num_rounds=config.mcts_rounds,
            evaluator=champion_evaluator,
            rng=random.Random(rng.randrange(1_000_000_000)),
        )
        result = play_game(
            black_bot=candidate_black,
            white_bot=champion_white,
            board_size=config.board_size,
            max_moves=config.board_size * config.board_size * 3,
        )
        if result.winner is Player.BLACK:
            candidate_wins += 1

        champion_black = MCTSBot(
            num_rounds=config.mcts_rounds,
            evaluator=champion_evaluator,
            rng=random.Random(rng.randrange(1_000_000_000)),
        )
        candidate_white = MCTSBot(
            num_rounds=config.mcts_rounds,
            evaluator=candidate_evaluator,
            rng=random.Random(rng.randrange(1_000_000_000)),
        )
        result = play_game(
            black_bot=champion_black,
            white_bot=candidate_white,
            board_size=config.board_size,
            max_moves=config.board_size * config.board_size * 3,
        )
        if result.winner is Player.WHITE:
            candidate_wins += 1

    return candidate_wins / total_games


def _new_model(config: ZeroTrainingConfig) -> PolicyValueNet:
    return PolicyValueNet(
        board_size=config.board_size,
        input_planes=2 * config.history_length + 1,
        channels=config.channels,
        num_res_blocks=config.num_res_blocks,
    ).to(config.device)


def _clone_model(model: PolicyValueNet, config: ZeroTrainingConfig) -> PolicyValueNet:
    clone = _new_model(config)
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    clone.eval()
    return clone


def _write_metrics(
    config: ZeroTrainingConfig,
    result: TrainingIterationResult,
    replay_buffer_size: int,
) -> None:
    metrics_path = config.metrics_path or Path(config.checkpoint_dir) / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "iteration": result.iteration,
                    "generated_examples": result.generated_examples,
                    "mean_loss": result.mean_loss,
                    "candidate_win_rate": result.candidate_win_rate,
                    "promoted": result.promoted,
                    "checkpoint_path": str(result.checkpoint_path),
                    "replay_buffer_size": replay_buffer_size,
                }
            )
            + "\n"
        )


def _ensure_resume_compatible(
    config: ZeroTrainingConfig,
    loaded_config: ZeroTrainingConfig,
) -> None:
    fields = (
        "board_size",
        "channels",
        "num_res_blocks",
        "history_length",
        "device",
    )
    for field_name in fields:
        if getattr(config, field_name) != getattr(loaded_config, field_name):
            raise ValueError(
                "resume checkpoint is incompatible: "
                f"{field_name}={getattr(loaded_config, field_name)!r} in checkpoint, "
                f"{getattr(config, field_name)!r} requested"
            )
