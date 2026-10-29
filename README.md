# AttentionLab

Implementations of attention mechanisms in PyTorch, built step by step from standard softmax attention up to Kimi Delta Attention (KDA), with tests that verify each layer and benchmarks for comparing them.

This is a learning project: each layer is written from scratch, starting with simple reference implementations that prioritize clarity over speed.

## What's implemented

| Layer | Paper | Status |
|---|---|---|
| Multi-head softmax attention (optional causal mask) | [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) | Tested |
| Linear attention | [Katharopoulos et al., 2020](https://arxiv.org/abs/2006.16236) | Tested |
| DeltaNet | [Schlag et al., 2021](https://arxiv.org/abs/2102.11174); [Yang et al., 2024](https://arxiv.org/abs/2406.06484) | Tested |
| Gated DeltaNet | [Yang et al., 2024](https://arxiv.org/abs/2412.06464) | Tested |
| Kimi Delta Attention (KDA) | [Kimi Team, 2025](https://arxiv.org/abs/2510.26692) | Tested |

Each layer in the chain adds one idea to the previous one, so each can be tested against its predecessor and benchmarked to show what that idea contributes.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). A CUDA GPU is recommended for training but not needed for the tests.

```bash
git clone https://github.com/SamuelWLee27/AttentionLab.git
cd AttentionLab/attnlab
uv sync
```

`uv sync` creates a virtual environment and installs the exact dependency versions pinned in `uv.lock`.

## Running tests

```bash
uv run pytest
```

Layers with a PyTorch equivalent are checked value-for-value against it (for example, softmax attention against `torch.nn.functional.scaled_dot_product_attention`).

## Linting and formatting

```bash
uv run ruff check .
uv run ruff format .
```

## Project structure

```
attnlab/
├── src/attnlab/
│   ├── attention/   # one file per attention variant
│   ├── models/      # models that accept any attention layer
│   └── data/        # datasets and synthetic tasks
├── tests/           # mirrors src/attnlab
├── configs/         # experiment settings
├── scripts/         # entry points for training and benchmarks
└── results/         # run outputs (not tracked by git)
```

See the [package README](attnlab/README.md) for details.

## References

- Vaswani et al. (2017). *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Katharopoulos et al. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)
- Schlag, Irie, Schmidhuber (2021). *Linear Transformers Are Secretly Fast Weight Programmers.* [arXiv:2102.11174](https://arxiv.org/abs/2102.11174)
- Yang et al. (2024). *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)
- Yang, Kautz, Hatamizadeh (2024). *Gated Delta Networks: Improving Mamba2 with Delta Rule.* [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)

## License

MIT. See [LICENSE](LICENSE).
