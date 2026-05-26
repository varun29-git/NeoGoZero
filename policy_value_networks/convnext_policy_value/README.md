# ConvNeXt NeoGoZero Variant

This folder contains a separate policy-value implementation that replaces the
ResNet tower with a ConvNeXt-style tower.

## Research Notes

ConvNeXt modernizes ConvNets by borrowing design choices that made Vision
Transformers strong while staying fully convolutional. The key block design we
use here is:

1. depthwise 7x7 convolution
2. LayerNorm over channels
3. inverted bottleneck expansion, usually 4x
4. GELU activation
5. projection back to the original channel count
6. learnable layer scale
7. residual connection with optional stochastic depth

The original image ConvNeXt uses patch/downsampling stages. For Go, we keep a
single spatial resolution because the policy head must preserve one logit per
board point plus pass. This is the same reason the ResNet version keeps the
board grid intact.

## Files

- `convnext_policy_value.py`: ConvNeXt policy-value network and evaluator.
- `convnext_zero_loop.py`: repeatable training loop with checkpoints and metrics.
- `train_convnext_zero.py`: CLI entrypoint.
- `test_convnext_policy_value.py`: smoke and checkpoint tests.

## Smoke Run

```bash
python3 policy_value_networks/convnext_policy_value/train_convnext_zero.py
```

The command exports final weights and a zipped download bundle under
`trained_model_weights/convnext_policy_value/` when training finishes. In Colab,
add `--auto-download-weights` to trigger a browser download.

## Larger 9x9 Shape

```bash
python3 policy_value_networks/convnext_policy_value/train_convnext_zero.py \
  --board-size 9 \
  --history-length 8 \
  --channels 256 \
  --blocks 20 \
  --supervised-sgf-dir supervised_go_data/sgf_9x9 \
  --supervised-steps 1000 \
  --mcts-rounds 300 \
  --self-play-games 25 \
  --training-steps 1000 \
  --evaluation-games 20 \
  --device cuda
```
