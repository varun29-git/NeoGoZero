# NeoGoZero Journal

## 2026-05-26

### Project Start

- Started the project as a sequential AlphaGo-style implementation.
- Chose to begin with a small, testable Go engine before adding search or neural networks.
- Targeted 9x9 Go first so the system can be built and debugged quickly.

### Milestone 1: Go Rules Engine

- Created the first Python project structure, later clarified into explicit top-level folders.
- Implemented the core Go types:
  - `Player`
  - `Point`
  - `Board`
  - `Move`
  - `GameState`
- Implemented board behavior:
  - Stone placement
  - Group detection
  - Liberty detection
  - Captures
  - Empty point listing
  - Text rendering of the board
- Implemented game behavior:
  - Legal move checking
  - Suicide prevention
  - Ko prevention
  - Passing
  - Game end after two consecutive passes
  - Area scoring with komi
  - Winner selection
- Added a `RandomBot` that chooses uniformly from legal moves.
- Added `play_and_train_commands/play_random_game.py` so two random bots can complete a full game.
- Added tests covering captures, group captures, suicide, capture-not-suicide, ko, passing, scoring, and random bot completion.
- Verified the rules milestone with `python3 -m pytest`.

### Project Rename

- Renamed the repo folder from `MyAlphaGo` to `NeoGoZero`.
- Updated the README title and project metadata to use `NeoGoZero`.
- Split the code into explicit top-level packages: `go_engine`, `search_players`, `match_evaluation`, `zero_training_pipeline`, and `policy_value_networks`.
- Renamed the command folder to `play_and_train_commands` so the runnable files describe their purpose instead of sitting under a generic scripts folder.
- Moved the repo out of `/Users/varundaiya/1bit_llm_advanced` so it lives as its own standalone project at `/Users/varundaiya/NeoGoZero`.

### Milestone 2: Basic PUCT Bot

- Added `MCTSNode` and `MCTSBot`.
- Implemented the four classical tree-search steps:
  - Selection with PUCT
  - Expansion from move priors
  - Evaluation
  - Signed value backpropagation
- Used uniform move priors with random rollout values as the temporary evaluator.
- Added rollout limits so early experiments stay bounded.
- Added `play_and_train_commands/play_mcts_vs_random.py` for a quick MCTS-vs-random smoke match.
- Added tests for MCTS legal move selection, small-game play, PUCT prior selection, custom evaluator behavior, and game-over handling.

### Milestone 3: PUCT Evaluator Interface

- Refactored the search implementation to be more AlphaGo Zero-shaped.
- Added an `Evaluator` protocol that returns:
  - Move priors
  - A scalar value from the next player's perspective
- Added `Evaluation` normalization and value clamping.
- Added `RandomRolloutEvaluator` as the temporary value source until a neural value head exists.
- Changed MCTS backpropagation from win-count tracking to signed scalar value backup.
- Added `SearchResult` so MCTS can expose visit counts and visit distributions for policy training.

### Milestone 4: Match Evaluation Harness

- Added a reusable match harness in `match_evaluation`.
- Implemented:
  - `play_game`
  - `play_match`
  - `GameResult`
  - `MatchResult`
- Added aggregate stats:
  - Black wins
  - White wins
  - Win rate
  - Average moves
  - Average margin
- Added `play_and_train_commands/evaluate_mcts_vs_random.py` for quick multi-game bot evaluation.
- Added tests for completed games, match aggregation, and illegal bot move rejection.

### Milestone 5: Self-Play Data

- Added self-play generation in `zero_training_pipeline.self_play`.
- Each self-play position stores:
  - Board snapshot
  - Player to move
  - MCTS visit distribution
  - Final winner
  - Training value target
- Added `play_and_train_commands/generate_self_play.py` for a tiny self-play smoke run.
- Added tests that verify self-play produces completed games and training examples.

### Milestone 6: Neural-Network Encoding Contract

- Added board and policy encoding helpers in `zero_training_pipeline.encoding`.
- Implemented:
  - Current-player board planes
  - Opponent board planes
  - Next-player color plane
  - Move-to-policy-index mapping
  - Policy-index-to-move mapping
  - Visit-distribution-to-policy-vector encoding
- Added tests for board encoding, policy encoding, pass moves, and index round trips.

### Milestone 7: ResNet Policy-Value Network

- Added `PolicyValueNet`, an AlphaGo Zero-style PyTorch model with:
  - Initial convolution, batch norm, and ReLU stem
  - Configurable residual tower
  - Shared convolutional body
  - Policy head over all board points plus pass
  - Value head in `[-1, 1]`
- Defaults mimic the real AlphaGo Zero architecture shape with 256 channels and 20 residual blocks, while commands and tests can request smaller versions for fast local smoke runs.
- Added `TorchPolicyValueEvaluator` so PUCT can consume neural priors and values.
- Added tensor conversion from self-play examples.
- Added policy-value loss and a single `train_step` helper.
- Added `play_and_train_commands/train_tiny_policy_value.py` for a tiny self-play-to-training smoke path.
- Added tests for model output shapes, tensor conversion, training loss, and neural evaluator priors.

### Milestone 8: Repeatable Zero Training Loop

- Added replay buffer support.
- Added a configurable Zero training loop that runs:
  - Self-play
  - Replay-buffer insertion
  - Policy-value training steps
  - Optional candidate-vs-champion evaluation
  - Model promotion or rollback
  - Checkpoint saving
- Added checkpoint loading for model, optimizer, config, and replay examples.
- Added `play_and_train_commands/train_zero.py` as the main repeatable training entrypoint.
- Added tests for replay-buffer behavior and checkpoint round trips.

### Milestone 9: Training-Ready Self-Play Features

- Added temperature-based move sampling from MCTS visit counts.
- Added root Dirichlet noise for self-play exploration.
- Added configurable history-plane encoding.
- Set the standalone ResNet model default to 17 input planes, matching 8 history positions plus the side-to-move plane.
- Added `--history-length`, `--self-play-temperature`, `--temperature-drop-move`, `--dirichlet-alpha`, and `--dirichlet-epsilon` flags to `play_and_train_commands/train_zero.py`.
- Added checkpoint resume support through `--resume-checkpoint`.
- Added JSONL metrics writing for each training iteration.
- Added `--device` to ResNet and ConvNeXt training commands so 9x9 runs can target CUDA/MPS-capable accelerators.
- Added optional supervised SGF pretraining before self-play fine-tuning for both ResNet and ConvNeXt.
- Added batched neural leaf evaluation in PUCT through `--mcts-inference-batch-size`.
- Added random dihedral symmetry augmentation during training batches.
- Added cached board point lookup and cached board position hashes for faster rules and ko checks while keeping the public board API stable.
- Renamed active board snapshot calls away from the misleading `zobrist_key` name, removed dead encoder code, generalized `train_step` typing for both model families, and made tied scores explicit errors instead of silent White wins.
- Added automatic final weight exports and zipped download bundles for both ResNet and ConvNeXt training commands.
- Added `play_and_train_commands/train_both_models_9x9_t4.py` so a single T4 VM can train ResNet and ConvNeXt sequentially without supervision.
- Tuned the unattended T4 launcher for a 24-hour-per-model run with ResNet first, ConvNeXt second, supervised pretraining capped at 20% of each model's wall-clock budget, and larger self-play data collection.
- Added self-play JSONL records and a run manifest so later analysis can inspect games, scores, move counts, metrics, logs, checkpoints, and exported weights.
- Added ETA reporting to the unattended T4 launcher.
- Added tests for temperature selection, Dirichlet noise, history planes, self-play history capture, metrics writing, and checkpoint resume.

### Next Direction

- Profile batched MCTS on the T4 and tune the inference batch size.
- Add stronger training/evaluation dashboards.

### Milestone 10: ConvNeXt Variant

- Researched ConvNeXt's block structure and added a separate implementation in `policy_value_networks/convnext_policy_value/`.
- Added a ConvNeXt-style policy-value model with:
  - Depthwise 7x7 convolution
  - Channel LayerNorm
  - 4x inverted bottleneck
  - GELU
  - Layer scale
  - Optional stochastic depth
- Kept Go board spatial resolution intact so policy logits still map to board points plus pass.
- Added a ConvNeXt-specific training loop, checkpoint format, CLI, README, and tests.
