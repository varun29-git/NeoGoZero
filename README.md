# NeoGoZero

NeoGoZero is a research-oriented AlphaGo Zero / AlphaZero-style Go engine built
from first principles for controlled 9x9 experiments. The project contains a
complete rules engine, PUCT search, neural policy-value evaluators, self-play
data generation, replay training, supervised warm-start support, checkpointing,
metrics, and two interchangeable backbone families:

- an AlphaGo Zero-style residual policy-value network
- a spatial ConvNeXt policy-value network variant

The codebase is intentionally modular rather than packaged as a black-box bot.
The main goal is to make each component inspectable, testable, and replaceable
before scaling training.

## Training Status

Full long-horizon model training is currently deferred until adequate compute
and throughput resources are available.

The implementation is ready as a code and experiment scaffold, but initial
single-T4 training attempts showed that the current Python self-play/MCTS path
is the limiting factor. The GPU works for neural evaluation and training, but
the search loop still spends substantial time in Python tree traversal, Go state
transition logic, legality checks, and immutable state construction. Training
will resume after the self-play benchmark and MCTS throughput path are improved
or stronger compute is available.

This repository should therefore be read as:

- complete algorithmic implementation
- validated testbed for rules, encoding, PUCT, training, and checkpointing
- ready-to-run smoke training stack
- deferred full-scale training artifact

It should not yet be presented as a fully trained strong Go model.

## What Makes This Different From A Minimal AlphaZero Clone

Many small AlphaZero reproductions stop at a monolithic script with UCT-like
search, a simple CNN, and unstructured self-play. NeoGoZero is deliberately more
research-grade in the code structure and algorithmic boundaries:

1. Search is PUCT, not UCT.
2. Neural evaluators are behind an explicit `Evaluator` / `BatchEvaluator`
   protocol.
3. MCTS supports batched leaf evaluation instead of forcing one GPU call per
   leaf.
4. Virtual visits are applied during batched search selection to reduce repeated
   exploration of the same path.
5. Dirichlet exploration noise is injected at the root for self-play.
6. The training target is the MCTS visit distribution, not just the played move.
7. The value target is final game outcome from the acting player's perspective.
8. The input representation supports AlphaZero-style temporal planes.
9. The project contains both a canonical residual tower and a ConvNeXt tower.
10. Self-play records, metrics, checkpoints, and exported weight bundles are
    written as analysis artifacts.
11. SGF supervised pretraining exists as a short warm-start phase before
    self-play.
12. Training examples use dihedral symmetry augmentation.
13. Checkpoint resume compatibility is explicitly checked against structural
    config fields.
14. The Go engine implements captures, suicide prevention, area scoring, komi,
    and positional ko detection.

The result is not a toy script. It is a layered AlphaZero system whose current
limitation is training throughput, not missing core algorithmic pieces.

## Repository Layout

```text
go_engine/
  board.py                         Immutable Go board, cached lookup map, position hash
  game.py                          Game state, legal moves, ko, scoring, winner logic
  types.py                         Player and Point primitives

search_players/
  random_bot.py                    Simple baseline player
  mcts_bot.py                      PUCT MCTS, evaluators, batching, virtual visits

match_evaluation/
  match.py                         Bot-vs-bot game and match utilities

zero_training_pipeline/
  encoding.py                      Board/history encoding and policy indexing
  self_play.py                     Self-play example generation
  supervised_pretraining.py        SGF parsing and supervised warm-start
  torch_training.py                Tensor conversion, loss, augmentation, train step
  weight_exports.py                Checkpoint-to-weights export and zip bundling
  zero_loop.py                     ResNet Zero training loop

policy_value_networks/
  resnet_policy_value/
    policy_value.py                Residual policy-value tower and evaluator
  convnext_policy_value/
    convnext_policy_value.py       ConvNeXt policy-value tower and evaluator
    convnext_zero_loop.py          ConvNeXt Zero training loop
    train_convnext_zero.py         ConvNeXt training CLI

play_and_train_commands/
  play_random_game.py              Rules smoke test
  play_mcts_vs_random.py           MCTS smoke game
  evaluate_mcts_vs_random.py       Match evaluation command
  generate_self_play.py            Tiny self-play data command
  train_tiny_policy_value.py       Neural smoke training
  train_zero.py                    ResNet training CLI
  train_both_models_9x9_t4.py      Sequential ResNet then ConvNeXt launcher

tests/                             Rules, MCTS, encoding, self-play, training tests
JOURNAL.md                         Development journal
LLM_CODE_BUNDLE_RESNET.md          Full code bundle for ResNet review
LLM_CODE_BUNDLE_CONVNEXT.md        Full code bundle for ConvNeXt review
```

## Core Algorithm

NeoGoZero follows the AlphaZero improvement loop:

```text
current network -> PUCT self-play -> visit-policy targets
                -> replay training -> checkpoint -> repeat
```

Each position produces a training example:

```text
(board history, player to move, MCTS visit distribution, final winner)
```

The policy head is trained against the normalized visit counts. The value head
is trained against `+1` if the final winner matches the example's player and
`-1` otherwise.

The combined loss is:

```text
loss = cross_entropy(MCTS_policy, network_policy_logits)
     + mean_squared_error(outcome_value, network_value)
```

The training loop uses AdamW for both model families.

## Go Engine

The engine is intentionally small but rule-complete enough for 9x9 self-play.

Implemented:

- arbitrary square board sizes
- black/white stones
- pass moves
- captures
- suicide rejection
- legal move generation
- two-pass game termination
- Tromp-Taylor-style area scoring with komi
- positional ko detection
- explicit tie error instead of silently assigning ties to White

The board remains immutable for correctness and easy tree search reasoning, but
it now caches:

- `_stones_by_point` for O(1) point lookup
- `_position_hash` for faster positional comparisons

This is a meaningful improvement over a naive immutable `frozenset` board where
every `get(point)` call scans all stones. The current board is still not a
high-performance engine board; future work should move toward mutable boards
with undo, flat arrays, and incremental liberty tracking.

## PUCT MCTS

`search_players/mcts_bot.py` implements predictor + UCT search in the AlphaZero
style.

Selection score:

```text
score(child) = -Q(child)
             + c_puct * P(child) * sqrt(N(parent)) / (1 + N(child))
```

Important details:

- `Q` is negated across edges because value is always interpreted from the
  current player's perspective.
- `prior_probability` comes from the evaluator policy head.
- `Evaluation.from_priors` masks illegal moves and renormalizes priors.
- root Dirichlet noise is optional and used for self-play exploration.
- visit-count temperature controls early-game sampling vs late-game argmax.
- `BatchEvaluator.evaluate_many` allows multiple leaves to be sent through the
  neural network in a single call.
- virtual visits are applied before batched leaf evaluation, then released when
  backing up values.

The code still includes `RandomRolloutEvaluator`, but it exists as a smoke-test
and isolation tool. The intended AlphaZero path is neural policy-value
evaluation, not rollout-heavy classical MCTS.

## State Encoding

The default neural input is 17 planes:

```text
8 historical board states * 2 color planes + 1 side-to-move plane
```

For each history step:

- one plane marks current-player stones
- one plane marks opponent stones

The final plane is filled with:

- `1` if Black is to move
- `0` if White is to move

Policy vectors have length:

```text
board_size * board_size + 1
```

The final index is pass.

This matches the important AlphaZero idea that a spatial convolutional model
needs explicit temporal context to reason about repetitions and move history.

## ResNet Policy-Value Network

`policy_value_networks/resnet_policy_value/policy_value.py` implements the
canonical AlphaGo Zero-style residual tower:

- 3x3 convolutional stem
- batch normalization
- ReLU
- configurable stack of residual blocks
- policy head with 1x1 convolution and linear logits
- value head with 1x1 convolution, hidden linear layer, and `tanh`

The default full-size research configuration is parameterized as:

```text
board_size=9
input_planes=17
channels=128 or 256
num_res_blocks=10 or 20
```

Small defaults are kept for tests and laptop smoke runs.

## ConvNeXt Policy-Value Network

`policy_value_networks/convnext_policy_value/convnext_policy_value.py` implements
a spatial ConvNeXt translation for Go.

Unlike image-classification ConvNeXt variants, this model avoids downsampling,
patch merging, pooling, and class-token style reductions. Go needs a dense
policy over board intersections, so the tower preserves `H x W` resolution
throughout.

Each ConvNeXt block contains:

- depthwise spatial convolution
- 2D LayerNorm implemented through channel-last permutation
- pointwise expansion
- GELU
- pointwise projection
- layer scale
- stochastic depth
- residual addition

The policy and value heads mirror the ResNet output contract:

```text
forward(board_planes) -> (policy_logits, values)
```

Because both architectures share the same training interface, they can be
compared under the same self-play, replay, checkpoint, and evaluation code.

## Supervised Warm-Start

The project supports short supervised pretraining from SGF files before
self-play. This is not intended to replace AlphaZero-style self-play. It is a
warm-start phase to move the network away from random priors.

SGF support includes:

- recursive discovery of `.sgf` files
- board-size filtering
- move parsing
- pass handling through `[]` and `tt`
- result-derived value targets when available
- legality filtering through the local Go engine

The unattended T4 launcher caps supervised pretraining as a fraction of each
model's wall-clock budget. Full training is currently deferred, but this path is
implemented and tested.

## Replay Training

Replay data is stored in a sliding FIFO buffer:

```text
ReplayBuffer(capacity=N)
```

Each iteration:

1. generates self-play games
2. appends examples to replay
3. samples mini-batches
4. applies random dihedral augmentation
5. optimizes policy and value heads
6. writes metrics
7. saves a checkpoint

Dihedral augmentation uses the eight square-board symmetries and transforms both
the board planes and the policy vector. Pass probability is preserved.

## Checkpoints And Artifacts

Checkpoints include:

- iteration number
- model state dict
- optimizer state dict
- serialized training config
- replay examples
- promotion metadata
- candidate win rate

Exported weight bundles include:

- model weights
- manifest JSON
- zipped download bundle

Self-play records are JSONL, one line per completed game. They include:

- architecture
- iteration
- game index
- board size
- MCTS rounds
- inference batch size
- winner
- number of moves
- number of examples
- score
- move list

These files are intended for later analysis and paper-style experiment tables.

## Training Launcher

The sequential T4 launcher runs:

```text
ResNet first, then ConvNeXt
```

Command:

```bash
python3 play_and_train_commands/train_both_models_9x9_t4.py
```

The launcher writes:

```text
training_runs/t4_9x9/<run_id>/
  run_manifest.json
  logs/
  checkpoints_resnet_policy_value/
  checkpoints_convnext_policy_value/
  metrics_resnet_policy_value.jsonl
  metrics_convnext_policy_value.jsonl
  self_play_records_resnet_policy_value.jsonl
  self_play_records_convnext_policy_value.jsonl
  trained_model_weights/
```

The original long-run intent was 24 hours per model on a single T4. That run is
deferred until either:

- MCTS/self-play throughput is improved, or
- stronger compute is available, or
- a smaller benchmarked configuration is explicitly accepted.

## Known Bottleneck

The current bottleneck is self-play throughput, not CUDA availability.

On a single T4, PyTorch can see and use the GPU. However, 9x9 self-play with
large PUCT budgets spends most time in Python:

- tree traversal
- legal move generation
- immutable `GameState` creation
- board reconstruction
- group/liberty checks
- ko history traversal

The GPU is underutilized because the CPU search path cannot feed enough batched
neural evaluations.

This is why full model training is deferred. Running a 24-hour experiment before
fixing throughput would produce weak data and waste compute.

## Recommended Next Engineering Work

Before serious training:

1. Add per-game progress logging and real ETA independent of child-process
   stdout.
2. Add a benchmark CLI for games/hour across MCTS budgets 16, 32, 64, 96, 128.
3. Replace the immutable board transition hot path with a faster representation.
4. Add mutable board + undo for MCTS.
5. Add incremental liberty/group accounting.
6. Add true incremental Zobrist hashing.
7. Add optional tree reuse between moves.
8. Add multiprocessing self-play workers feeding a shared replay writer.
9. Add mixed precision for neural training/evaluation.
10. Separate production checkpoints from replay-heavy resume checkpoints.

After these changes, the 24h ResNet and 24h ConvNeXt comparison becomes much
more meaningful.

## Running Tests

```bash
python3 -m pytest
```

The test suite covers:

- Go rules and scoring
- match execution
- MCTS behavior
- encoding and policy indexing
- policy-value model training
- self-play generation
- supervised SGF parsing
- checkpointing and resume paths
- ConvNeXt training/export smoke tests

## Smoke Commands

Random game:

```bash
python3 play_and_train_commands/play_random_game.py
```

MCTS vs random:

```bash
python3 play_and_train_commands/play_mcts_vs_random.py
```

Evaluation match:

```bash
python3 play_and_train_commands/evaluate_mcts_vs_random.py
```

Generate one small self-play game:

```bash
python3 play_and_train_commands/generate_self_play.py
```

Tiny neural smoke training:

```bash
python3 play_and_train_commands/train_tiny_policy_value.py
```

ResNet training CLI:

```bash
python3 play_and_train_commands/train_zero.py
```

ConvNeXt training CLI:

```bash
python3 policy_value_networks/convnext_policy_value/train_convnext_zero.py
```

## Example Research Config

This is an example configuration, not a currently recommended full training
run:

```bash
python3 play_and_train_commands/train_zero.py \
  --board-size 9 \
  --history-length 8 \
  --channels 128 \
  --res-blocks 10 \
  --supervised-sgf-dir supervised_go_data/sgf_9x9 \
  --supervised-steps 1000 \
  --mcts-rounds 64 \
  --mcts-inference-batch-size 64 \
  --self-play-games 25 \
  --training-steps 300 \
  --evaluation-games 0 \
  --device cuda
```

Benchmark throughput before scaling any run.

## Current Position

NeoGoZero has reached the point where the important AlphaZero components exist
and are wired together. The next milestone is not adding more theory; it is
making the implementation mechanically fast enough to generate meaningful
self-play data.

Training is deferred until resources and throughput are available.
