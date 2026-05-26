# NeoGoZero

A sequential AlphaGo Zero-style implementation.

Current milestone:

1. 9x9 Go rules engine
2. Random bot smoke play
3. Basic MCTS bot
4. PUCT evaluator interface
5. Match evaluation harness
6. Self-play data generation
7. Neural-network encoding contract
8. ResNet-style policy-value neural network

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

Run a small evaluation match:

```bash
python3 scripts/evaluate_mcts_vs_random.py
```

Generate one tiny self-play training game:

```bash
python3 scripts/generate_self_play.py
```

Run a tiny policy-value training smoke test:

```bash
python3 scripts/train_tiny_policy_value.py
```

Run the repeatable Zero training loop:

```bash
python3 scripts/train_zero.py
```

Useful training flags:

```bash
python3 scripts/train_zero.py \
  --board-size 9 \
  --history-length 8 \
  --channels 256 \
  --res-blocks 20 \
  --mcts-rounds 800 \
  --self-play-games 25 \
  --training-steps 1000 \
  --evaluation-games 20
```

The default command is intentionally tiny so it can run as a smoke test on a laptop.
