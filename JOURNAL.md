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

### Current Direction

- Next milestone is a basic Monte Carlo Tree Search bot.
- This MCTS bot will not use neural networks yet.
- The goal is to reach `MCTSBot vs RandomBot` before introducing policy or value networks.

### Milestone 2: Basic MCTS Bot

- Added `MCTSNode` and `MCTSBot`.
- Implemented the four classical MCTS steps:
  - Selection with UCT
  - Expansion from unvisited legal moves
  - Random rollout
  - Backpropagation of the winning player
- Added rollout limits so early experiments stay bounded.
- Added `scripts/play_mcts_vs_random.py` for a quick MCTS-vs-random smoke match.
- Added tests for MCTS legal move selection, small-game play, and game-over handling.
- Verified the expanded suite with `python3 -m pytest`.
- Ran the MCTS-vs-random smoke script successfully.

### Next Direction

- Improve performance and observability before neural networks.
- Next likely step: add a match/evaluation harness so bots can play many games and report win rate.
