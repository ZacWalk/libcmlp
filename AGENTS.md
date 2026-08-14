# AGENTS.md

Guidance for AI coding agents working in this repository.

**Read [docs/design.md](docs/design.md) first.** It is the single source of truth for
architecture, data layout, algorithms, configuration, and conventions. This file only covers
how to operate in the repo.

## What this is

A dependency-free C++20 Fashion-MNIST multilayer perceptron, built as a Visual Studio
solution (`nn.sln`). Standard library plus AVX2 intrinsics — nothing else.

| Path | Contents |
|---|---|
| `src/` | All C++ sources — see the module map in [docs/design.md](docs/design.md) |
| `data/` | Fashion-MNIST training and test CSVs |
| `bin/nn.exe` | Release x64 build output |
| `py/torch-test.py` | PyTorch reference baseline, not a mirror of the C++ model |
| `docs/design.md` | Architecture, algorithms, conventions — read first |
| `docs/experiments.md` | Accuracy tuning record: what was tried, adopted, and rejected |

## Commands

Always use the driver script; it resolves MSBuild and sets the working directory correctly.

```powershell
.\dd.ps1 build   # MSBuild Release|x64 -> bin\nn.exe
.\dd.ps1 run     # build, then train and evaluate
.\dd.ps1 test    # build, then run the smoke suite
```

Run `.\dd.ps1 test` after any change to `src/`. It is the only test suite.

## Rules

- **Run from the repository root.** Dataset paths in `main.cpp` are relative (`.\data\...`).
- **No new dependencies.** Zero third-party libraries is the point of the project. If a
  change seems to need one, it is the wrong change.
- **Defaults live in `src/common.h`.** Do not hard-code a hyperparameter anywhere else; add
  the constant there and read it through the matching `NN_*` environment variable.
- **Add new `.cpp`/`.h` files to `nn.vcxproj`** (both the `ClCompile` and `ClInclude`
  item groups) or they will be silently ignored by the build.
- **Weights are row-major** (`[output_neuron][input_activation]`) and biases are a pinned
  `1.0` activation column, not a separate vector. Changing either breaks every kernel.
- **Keep the console output format stable.** `dd.ps1 test` parses the
  `[EVALUATION] ... [ACCURACY n out of m]` line.
- **Preserve seeded reproducibility.** Any new randomness must be driven from
  `nn_config::seed` / `trainer_config::seed`, never from a fresh `std::random_device`.
- **Guard SIMD with `#if defined(__AVX2__)`** and keep the scalar fallback correct — Debug
  and Win32 builds do not enable AVX2.
- **Documentation delegates.** Substantive design content belongs in `docs/design.md`; other
  markdown files link to it rather than restating it.

## Performance expectations

The default configuration should finish in roughly 36 seconds and evaluate around
8 990 / 10 000. A large regression in either is a bug, not noise. Benchmark by interleaving
runs against an unmodified build rather than comparing against a remembered number.

Before proposing an accuracy improvement, read [docs/experiments.md](docs/experiments.md).
ReLU, weight decay, and shift/flip augmentation have all been measured on this model and none
of them helped. Validate any claimed gain across several `NN_SEED` values — the single-run
spread is roughly ±65 samples, so differences under ~100 are noise.

## Verifying a refactor

A green build proves nothing about whether a change actually landed. After extracting or
replacing code, grep for the old symbols and the new header's include sites to confirm the
old path is gone and the new one is reached.
