# libcmlp

[![Linux](https://github.com/ZacWalk/libcmlp/actions/workflows/linux.yml/badge.svg)](https://github.com/ZacWalk/libcmlp/actions/workflows/linux.yml)
[![Windows](https://github.com/ZacWalk/libcmlp/actions/workflows/windows.yml/badge.svg)](https://github.com/ZacWalk/libcmlp/actions/workflows/windows.yml)

A dependency-free C11 multilayer-perceptron library, with a Fashion-MNIST example CLI.
It supports classification and unbounded regression/Q-values, SGD with momentum or
Adam, gradient clipping, batching, target-network copying, and binary checkpoints.
AVX2/FMA accelerates the hot loops; scalar builds work without AVX2.

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

Windows needs PowerShell 7.4+, CMake 3.24+, Ninja, and Visual Studio Build Tools. Vendored
upstream dd resolves the developer environment:

```powershell
pwsh -File dd.ps1 build release  # C library + mlp-cli example
pwsh -File dd.ps1 run     # train and evaluate with mlp-cli
pwsh -File dd.ps1 test    # build and test Release and Debug
```

Linux and WSL need PowerShell 7.4+, CMake 3.24+, `ninja-build`, and GCC/G++.
The supplied dd presets target native x64 Windows/Linux and select GCC on Linux.
Upstream dd checks for `g++` even though this project compiles only C; direct
CMake configuration can use Clang or other architectures instead:

```bash
pwsh -File dd.ps1 build release
pwsh -File dd.ps1 run
pwsh -File dd.ps1 test
```

The former bespoke PowerShell/Bash drivers are retired. The same upstream dd
entry point works on both platforms. Run from the repository root so relative
dataset paths resolve. `cmlp` is a library target and cannot be run; `mlp-cli` is
the runnable example. Missing git-lfs data does not block the offline CI suite.
Bare `build` builds both Release and Debug. Binaries and archives are under
`build/x64-windows/{release,debug}/` or `build/x64-linux/{release,debug}/`.

```powershell
pwsh -File dd.ps1 test --label '^ci$'  # generated data, no git-lfs required
pwsh -File dd.ps1 scalar --yes        # isolated scalar build and CI tests
pwsh -File dd.ps1 asan --yes          # Linux ASan + fatal UBSan lane
```

When developing, set `TEMP`, `TMP`, and `TMPDIR` to the repository's `tmp` directory
if dd's transient logs/JUnit files must stay inside the repository. CI does this
automatically.

## At a glance

| | |
|---|---|
| Topology | `784 -> 100 -> 50 -> 10` (84 060 weights and separate biases) |
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
pwsh -File dd.ps1 run
```

```bash
NN_EPOCHS=10 NN_LR=0.10 NN_LR_DECAY=0.70 pwsh -File dd.ps1 run
```

## Library API

Include [cmlp.h](include/cmlp.h) and link the CMake target `cmlp`. The header is
usable from C and C++. Start from `cmlp_default_config()`, supply the layer widths,
then call `cmlp_create`; report non-success status codes and release the model with
`cmlp_destroy`.

For DQN, select ReLU hidden units, linear output, Adam, and `gradient_clip = 10`.
`cmlp_forward` / `cmlp_backward` operate on batches and accept caller-provided
output derivatives; `cmlp_step` averages accumulated gradients. `cmlp_copy_from`
updates a compatible target network. `cmlp_save` / `cmlp_load` handle versioned
checkpoints. See [the API and ownership guide](docs/design.md#2-api-and-ownership)
for details and an example.

The test-count contract remains **11 CTests with Fashion-MNIST, 6 CI tests without
it**. Numerical API, persistence, DQN and CSV/trainer checks run within the
generated-data cases. Invalid numeric environment values now fail explicitly,
rather than silently selecting defaults.

## Contributing

Agent and contributor workflow rules live in [AGENTS.md](AGENTS.md).
