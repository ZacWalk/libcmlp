# nn

A small, dependency-free Fashion-MNIST classifier in C++20 — a multilayer perceptron trained
with mini-batch SGD and momentum, with AVX2 intrinsics in the hot loops. No frameworks, no
third-party libraries.

**[docs/design.md](docs/design.md) is the full documentation** — architecture, memory layout,
algorithms, configuration, testing, and results. This file is only a quickstart.
[docs/experiments.md](docs/experiments.md) records the accuracy tuning search behind the
current defaults.

## Quickstart

```powershell
.\dd.ps1 run     # build Release|x64, then train and evaluate
.\dd.ps1 test    # build, then run the smoke suite
.\dd.ps1 build   # build only
```

Requires Visual Studio Build Tools (MSBuild) and an AVX2-capable CPU for the Release x64
build. Run from the repository root so the relative dataset paths resolve.

## At a glance

| | |
|---|---|
| Topology | `784 -> 100 -> 50 -> 10` (84 060 weights) |
| Activations | sigmoid hidden, softmax output |
| Loss | categorical cross-entropy |
| Optimizer | mini-batch SGD with momentum and geometric LR decay |
| Evaluation accuracy | ~8 990 / 10 000 in ~36 s single-threaded |

Hyperparameters are overridable at runtime via `NN_HIDDEN1`, `NN_HIDDEN2`, `NN_LR`,
`NN_LR_DECAY`, `NN_MOMENTUM`, `NN_BATCH_SIZE`, `NN_EPOCHS`, and `NN_SEED` — no recompile
needed. See [docs/design.md](docs/design.md) for the details and
[docs/experiments.md](docs/experiments.md) for the approaches that were measured and rejected.

For a ~13 s run at ~8 900 / 10 000:

```powershell
$env:NN_EPOCHS='10'; $env:NN_LR='0.10'; $env:NN_LR_DECAY='0.70'
.\dd.ps1 run
```

## Contributing

Agent and contributor workflow rules live in [AGENTS.md](AGENTS.md).
