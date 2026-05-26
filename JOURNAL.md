# NeoGoZero Journal

## 2026-05-26

### Project Start

- Started the project as a sequential AlphaGo-style implementation.
- Chose to begin with a small, testable Go engine before adding search or neural networks.
- Targeted 9x9 Go first so the system can be built and debugged quickly.

### Milestone 1: Go Rules Engine

- Created the Python project structure with `src/`, `tests/`, and `scripts/`.
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
- Added `scripts/play_random_game.py` so two random bots can complete a full game.
- Added tests covering captures, group captures, suicide, capture-not-suicide, ko, passing, scoring, and random bot completion.
- Verified the rules milestone with `python3 -m pytest`.

### Project Rename

- Renamed the repo folder from `MyAlphaGo` to `NeoGoZero`.
- Updated the README title and project metadata to use `NeoGoZero`.
- Kept the Python import package as `myalphago` for stability during early development.
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
- Added `scripts/play_mcts_vs_random.py` for a quick MCTS-vs-random smoke match.
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

- Added a reusable match harness in `myalphago.evaluation`.
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
- Added `scripts/evaluate_mcts_vs_random.py` for quick multi-game bot evaluation.
- Added tests for completed games, match aggregation, and illegal bot move rejection.

### Milestone 5: Self-Play Data

- Added self-play generation in `myalphago.training.self_play`.
- Each self-play position stores:
  - Board snapshot
  - Player to move
  - MCTS visit distribution
  - Final winner
  - Training value target
- Added `scripts/generate_self_play.py` for a tiny self-play smoke run.
- Added tests that verify self-play produces completed games and training examples.

### Milestone 6: Neural-Network Encoding Contract

- Added board and policy encoding helpers in `myalphago.training.encoding`.
- Implemented:
  - Current-player board planes
  - Opponent board planes
  - Next-player color plane
  - Move-to-policy-index mapping
  - Policy-index-to-move mapping
  - Visit-distribution-to-policy-vector encoding
- Added tests for board encoding, policy encoding, pass moves, and index round trips.

### Milestone 7: Policy-Value Network

- Added `PolicyValueNet`, a compact PyTorch model with:
  - Shared convolutional body
  - Policy head over all board points plus pass
  - Value head in `[-1, 1]`
- Added `TorchPolicyValueEvaluator` so PUCT can consume neural priors and values.
- Added tensor conversion from self-play examples.
- Added policy-value loss and a single `train_step` helper.
- Added `scripts/train_tiny_policy_value.py` for a tiny self-play-to-training smoke path.
- Added tests for model output shapes, tensor conversion, training loss, and neural evaluator priors.

### Next Direction

- Turn the smoke training path into a repeatable training loop with checkpoints.
- Add model evaluation against previous checkpoints.
- Improve MCTS speed so larger 5x5 and 9x9 experiments are practical.
