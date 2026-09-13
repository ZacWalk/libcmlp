# AGENTS.md

Guidance for AI coding agents working in this repository.

**Read [docs/design.md](docs/design.md) first.** It is the single source of truth
for architecture, layouts, algorithms, configuration and conventions.

## What this is

A dependency-free C11 MLP library plus a Fashion-MNIST example CLI. C standard
library, platform timing/entropy support, and optional AVX2/FMA intrinsics only.
No C++ runtime, third-party numerical libraries, or platform-h dependency.

| Path | Contents |
|---|---|
| `include/cmlp.h` | Public C API; usable from C++ |
| `src/nn.c`, `src/activation.h`, `src/random.h` | Library implementation |
| `src/dataset.c`, `src/pipeline.c`, `src/trainer.c`, `src/main.c` | Example classifier |
| `src/common.h` | Default constants |
| `CMakeLists.txt`, `CMakePresets.json`, `dd.psd1` | Build and dd target declarations |
| `dd.ps1`, `.dd/` | Immutable vendored upstream dd runtime |
| `docs/dd-upstream.json` | SHA-256 fingerprints of vendored files |
| `test/` | C API/pipeline tests and CMake smoke harness |
| `.github/workflows/` | Separate Windows and Linux workflows |
| `data/` | Fashion-MNIST CSVs, git-lfs |
| `py/torch-test.py` | Independent PyTorch reference; not the C implementation |
| `docs/experiments.md` | Historical classifier tuning record |

## Commands

Use PowerShell 7.4+ and the upstream driver from the repository root on either
platform. The old bespoke `dd.sh` is retired.

```powershell
pwsh -File dd.ps1 build
pwsh -File dd.ps1 build debug
pwsh -File dd.ps1 run
pwsh -File dd.ps1 test
pwsh -File dd.ps1 test --label '^ci$'
pwsh -File dd.ps1 scalar --yes
```

`cmlp` is a buildable/testable library, not a runnable program; `mlp-cli` is the
example. CMake target names and `dd.psd1` `cmake-target` declarations must agree.
Bare build/test covers Release and Debug; `build release` selects only Release.
Linux also supports `pwsh -File dd.ps1 asan --yes`. Set `TEMP`, `TMP`, and `TMPDIR`
to the repository's `tmp` directory to keep dd logs/JUnit artifacts local.
Do not hand-edit vendored dd. Vendor updates must also regenerate every affected
fingerprint; preserve vendor bytes through `.gitattributes`.

Run the relevant tests before and after code changes. The suite has **11 tests**
when real Fashion-MNIST CSVs are present, **6** with `-L ci` or without git-lfs.
API tests run inside `generated.runs`, pipeline tests inside `generated.loads`.
A six-test pass alone is not evidence of real Fashion-MNIST accuracy.

## Rules

- **Stay C11 and portable.** Both MSVC and GCC/Clang must compile the library.
  Use `NN_FORCEINLINE` rather than spelling compiler attributes at call sites.
- **No new numerical dependencies.** Do not import a framework to implement a
  feature already expressible with the existing kernels.
- **Weights are row-major** `[output][input]`; biases are separate trainable
  vectors. P4 deliberately replaces the old pinned `1.0` activation column.
  Forward, backward, initialization and optimizers must agree on this layout.
- **Defaults live in `src/common.h`.** Classifier overrides use `NN_*` variables.
- **Preserve ownership and error contracts.** Check sizes/overflow/allocation,
  return explicit statuses from the library, and report failures at CLI boundaries.
  Never print from the numerical library or silently turn errors into success.
- **Keep the console metric format stable.** The smoke harness parses
  `[EVALUATION] ... [ACCURACY n out of m]`.
- **Preserve seeded reproducibility.** Route randomness through the shared RNG
  and configuration seed. Seed zero is the explicitly nondeterministic mode.
- **Guard SIMD with `#if defined(__AVX2__)`.** Test `NN_ENABLE_AVX2=OFF` as well as
  vectorized builds, including odd widths/vector tails.
- **Do not enable fast-math on validation code.** Nonfinite-input rejection must
  remain valid under optimization.
- **CI never sees the real dataset.** Dataset-dependent checks cannot carry the
  `ci` label; maintain both workflow files and corresponding README badges.
- **Temporary artifacts go in `tmp/` or a build-tree test scratch directory.**
- **Documentation delegates.** Architecture belongs in `docs/design.md`; other
  documents link to it.

## Performance and accuracy

Full 30-epoch accuracy must remain within seed noise of the pre-P4 ~8,960/10,000
four-seed mean. Reference times are machine-dependent; the measured P4 controls
and results are in [docs/design.md](docs/design.md).

Benchmark by interleaving runs against an unmodified executable, never against
remembered numbers or a different OS/compiler. Clear unrelated `NN_*` settings.
Check multiple seeds (12345, 1, 7, 99); differences under about 100 examples are
noise. Do not silently accept a large runtime regression to pass accuracy.

Before proposing classifier accuracy changes, read [docs/experiments.md](docs/experiments.md).
ReLU, weight decay and shift/flip augmentation have already been measured.
ReLU/Adam are included for general models and DQN, not as new classifier defaults.

After replacing code, search for obsolete symbols and include sites to confirm
the old path is gone. A green build alone does not prove that the new code is used.

## dd modes

dd has two modes, and the command line selects between them.

**CLI mode** is the default and the single behavior owner. Each verb runs once,
prints one schema 1 result envelope and exits:

```pwsh
pwsh -NoProfile -File ./dd.ps1 test --json
pwsh -NoProfile -File ./dd.ps1 build debug
```

**MCP mode** starts with `dd mcp`. The process becomes a stdio JSON-RPC server and
stays alive until stdin closes, adapting typed MCP requests onto CLI mode — each
tool call runs as a child `dd` invocation and returns that command's envelope.

MCP mode owns stdout for protocol messages, so it prints no result envelope,
rejects `--json`, and sends diagnostics to stderr. Its workspace boundary is the
project root. Register the client configuration with `dd ide --mcp`; add
`--allow-execution` only when project-code execution is intended.
