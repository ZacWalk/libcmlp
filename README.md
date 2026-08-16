# nn

[![Linux](https://github.com/ZacWalk/nn/actions/workflows/linux.yml/badge.svg)](https://github.com/ZacWalk/nn/actions/workflows/linux.yml)
[![Windows](https://github.com/ZacWalk/nn/actions/workflows/windows.yml/badge.svg)](https://github.com/ZacWalk/nn/actions/workflows/windows.yml)

A small, dependency-free Fashion-MNIST classifier in C++20 — a multilayer perceptron trained
with mini-batch SGD and momentum, with AVX2 intrinsics in the hot loops. No frameworks, no
third-party libraries.

**[docs/design.md](docs/design.md) is the full documentation** — architecture, memory layout,
algorithms, configuration, testing, and results. This file is only a quickstart.
[docs/experiments.md](docs/experiments.md) records the accuracy tuning search behind the
current defaults.

## Quickstart

The Fashion-MNIST CSVs are stored in git-lfs. If `data/` holds small pointer files rather
than the real 155 MB of data, pull them first:

```
git lfs pull
```

Windows — needs Visual Studio Build Tools; the driver imports the developer environment
itself:

```powershell
.\dd.ps1 run     # configure, build, then train and evaluate
.\dd.ps1 test    # build, then run the smoke suite
.\dd.ps1 build   # build only
```

Linux and WSL — needs `cmake`, `ninja-build`, and GCC or Clang:

```bash
./dd.sh run
./dd.sh test
./dd.sh build
```

Both drivers wrap CMake + Ninja + ctest and build into `build/<preset>/`. Run them from the
repository root so the relative dataset paths resolve.

## At a glance

| | |
|---|---|
| Topology | `784 -> 100 -> 50 -> 10` (84 060 weights) |
| Activations | sigmoid hidden, softmax output |
| Loss | categorical cross-entropy |
| Optimizer | mini-batch SGD with momentum and geometric LR decay |
| Evaluation accuracy | ~8 990 / 10 000, single-threaded |
| Wall clock | ~38 s with MSVC, ~28 s with GCC |

Hyperparameters are overridable at runtime via `NN_HIDDEN1`, `NN_HIDDEN2`, `NN_LR`,
`NN_LR_DECAY`, `NN_MOMENTUM`, `NN_BATCH_SIZE`, `NN_EPOCHS`, `NN_SEED`, `NN_TRAIN_CSV`, and
`NN_TEST_CSV` — no recompile needed. See [docs/design.md](docs/design.md) for the details and
[docs/experiments.md](docs/experiments.md) for the approaches that were measured and rejected.

For roughly a third of the runtime at ~8 900 / 10 000:

```powershell
$env:NN_EPOCHS='10'; $env:NN_LR='0.10'; $env:NN_LR_DECAY='0.70'
.\dd.ps1 run
```

```bash
NN_EPOCHS=10 NN_LR=0.10 NN_LR_DECAY=0.70 ./dd.sh run
```

## Contributing

Agent and contributor workflow rules live in [AGENTS.md](AGENTS.md).
