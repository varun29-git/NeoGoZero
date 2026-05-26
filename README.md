# NeoGoZero

A sequential AlphaGo Zero-style implementation.

Current milestone:

1. 9x9 Go rules engine
2. Random bot smoke play
3. Basic MCTS bot

Run tests:

```bash
python3 -m pytest
```

Run a random 9x9 game:

```bash
python3 scripts/play_random_game.py
```

Run a quick MCTS vs random smoke game:

```bash
python3 scripts/play_mcts_vs_random.py
```
