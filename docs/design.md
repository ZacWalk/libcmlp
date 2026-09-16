# libcmlp - Design

A dependency-free C11 multilayer-perceptron library, with a Fashion-MNIST example
CLI. No BLAS, frameworks, C++ runtime, or platform-h dependency. The public header
also works from C++ through `extern "C"`.

This is the single source of truth for architecture, algorithms, configuration,
and conventions. [experiments.md](experiments.md) records the original classifier
tuning, including negative results that should not be repeated.

## 1. Scope and modules

The library supports both classification and regression/Q-value approximation:

- Sigmoid, ReLU, tanh, or linear hidden activations.
- Linear (unbounded), sigmoid, ReLU, tanh, or softmax output.
- Mini-batch SGD with classical momentum, or Adam with bias correction.
- Optional global gradient-norm clipping.
- Batched forward/backward with caller-supplied output derivatives.
- Softmax/categorical-cross-entropy training and evaluation helpers.
- Parameter copying for DQN target networks and versioned binary checkpoints.
- AVX2/FMA kernels with a scalar fallback.

GPU execution, convolutions, and concurrent use of a single model are out of scope.
Independent model instances have independent state. Switching an application from
its local model to this API is a separate consumer migration; P4 does not remove
crypto-app's working model.

| File | Responsibility |
|---|---|
| [include/cmlp.h](../include/cmlp.h) | Public C API, configuration, status and ownership contracts |
| [src/nn.c](../src/nn.c) | Model storage, math, optimizers, copying and persistence |
| [src/activation.h](../src/activation.h) | Activations and derivatives |
| [src/random.h](../src/random.h) | Shared seeded random-number generation |
| [src/common.h](../src/common.h) | Scalar alias and default constants |
| [src/dataset.c](../src/dataset.c) | Flat sample and one-hot-label storage |
| [src/pipeline.c](../src/pipeline.c) | Buffered CSV parsing, normalization and pair validation |
| [src/trainer.c](../src/trainer.c) | Shuffling, mini-batches, schedule and metric reporting |
| [src/main.c](../src/main.c) | Example CLI configuration, wiring and timing |

The `cmlp` library knows nothing about CSVs or Fashion-MNIST. Dataset, pipeline,
trainer and main are example code in `mlp-cli`, not dependencies of the public API.

## 2. API and ownership

Start with `cmlp_default_config()`, provide a topology, then call `cmlp_create`.
Creation copies the topology; the caller's configuration and widths need only
survive the call. Destroy each successfully created or loaded model with
`cmlp_destroy`. A failed create/load leaves the caller's output pointer unchanged.
Do not overwrite an existing owned model without first arranging its destruction.

All fallible operations return a `cmlp_status`; use `cmlp_status_string` to report
failures. The library does not print to stdout, exit the process, or silently
substitute a successful result for invalid input. Buffer lengths are implied by
the model dimensions and sample count: callers must supply that much storage.
Do not share or overlap input/output storage with model-owned memory.

```c
#include <stdio.h>
#include "cmlp.h"

int main(void)
{
    const int widths[] = {4, 16, 3};
    const float input[] = {0.1f, 0.2f, 0.3f, 0.4f};
    float output[3];
    cmlp_model *model = NULL;
    cmlp_config config = cmlp_default_config();
    cmlp_status status;
    config.layers = widths;
    config.layer_count = 3;
    config.seed = 12345;
    config.hidden_activation = CMLP_RELU;
    config.output_activation = CMLP_LINEAR;
    config.optimizer = CMLP_ADAM;
    config.learning_rate = 0.001f;
    config.gradient_clip = 10.0f;
    status = cmlp_create(&config, &model);
    if (status == CMLP_OK)
        status = cmlp_forward(model, input, 1, output);
    if (status != CMLP_OK)
        fprintf(stderr, "%s\n", cmlp_status_string(status));
    cmlp_destroy(model);
    return status == CMLP_OK ? 0 : 1;
}
```

For a CMake consumer, link `cmlp`; its public include directory supplies `cmlp.h`.
There is no need to enable CXX. C++ consumers can link the same archive.

### Batching and DQN

`cmlp_forward(model, inputs, samples, outputs)` accepts contiguous row-major
inputs (`samples * input_size`) and optionally copies contiguous outputs
(`samples * output_size`). It retains all activations for that batch.

`cmlp_backward(model, derivatives, samples)` uses the **most recent forward
batch** and accepts `dLoss/dOutput`, not `dLoss/dLogit`. This includes the full
softmax Jacobian when softmax output is selected. Pass unaveraged derivatives;
`cmlp_step` averages over all accumulated sample rows. Multiple batches can
accumulate before a step. `cmlp_zero_grad` explicitly discards pending gradients.
Do not evaluate or run another forward between a forward and its backward.
Backward requires exactly the retained sample count and consumes that batch
once. A failed forward invalidates it. A numeric gradient-accumulation overflow
discards all pending gradients without changing parameters; a failed optimizer
step leaves parameters, gradients and optimizer state unchanged.

A DQN update can therefore:

1. Forward next states through a separate target model.
2. Form `reward + discount * max(next_q)` for nonterminal transitions, or just
   `reward` for terminal transitions.
3. Forward current states through the online model.
4. Fill derivatives with zeros except for each sampled action's TD derivative.
5. Backward the batch, then step once.
6. Periodically `cmlp_copy_from(target, online)`.

Copy requires identical topology and activations. It copies parameters and
resets the destination's gradient/optimizer state; it is not a training-resume
operation. The destination retains its optimizer configuration. The numerical
API tests exercise this sequence without any crypto-app or market-data dependency.

### Persistence

`cmlp_save` and `cmlp_load` use a versioned, little-endian binary32 format with a
CRC32 checksum, rather than a raw struct dump. Checkpoints retain optimizer
configuration, moments, update count and pending gradients, but not the retained
forward batch. A load validates the checkpoint and constructs a new model before handing
ownership to the caller. Unknown versions, malformed/truncated data and I/O errors
are reported explicitly. See the format implementation and round-trip/corruption
tests before changing it. This is a new libcmlp format: it does not claim
compatibility with crypto-app's existing model files.
Save overwrites the named file; a write failure may leave a partial file.
Applications needing atomic replacement should save to a separate path and
perform their own checked replacement operation.

## 3. Model layout and mathematics

The default classifier is `784 -> 100 -> 50 -> 10`, with 84,060 parameters:
weights **and** biases. Hidden layers use sigmoid; the output is stable softmax.

### Separate bias vectors

P4 intentionally replaces the original pinned `1.0` activation column.
Each connection has a row-major matrix `[output_neuron][input_neuron]` and a
separate trainable bias vector `[output_neuron]`. Activations contain only real
neurons. The forward equation is `activation(dot(weight_row, input) + bias)`.

Bias gradients are the sum of neuron deltas. Biases participate in the same
momentum/Adam update and global norm as weights, but **not** in the input dot
products, hidden-delta propagation or fan-in calculation.

Parameters, accumulated gradients and optimizer buffers use matching offsets.
Row-major weights keep the dot product, gradient accumulation and delta
propagation unit-stride. Backpropagation accumulates downstream weight rows
scaled by their deltas, then applies the hidden activation derivative.

### Initialization and randomness

Use He initialization for ReLU and Xavier/Glorot for the other activations.
Seeded initialization and shuffling use a shared C PRNG, not global `rand()`.
Nonzero seeds give repeatability within a build/toolchain; seed zero requests a
fresh seed. Porting from C++ distributions and shuffling does not promise the old
bitwise sequence, nor identical predictions across compilers or SIMD modes.
Accuracy is compared over multiple seeds rather than a single lucky result.

### Loss and optimization

The classifier helpers require softmax output and probability targets.
Targets must sum to one within `1e-5`. Stable
softmax subtracts the maximum logit before exponentiation. Cross-entropy has
output-logit derivative `probability - target`; no separate softmax Jacobian is
needed in that combined helper.

For SGD, with mean batch gradient `g`:

```
velocity = momentum * velocity - learning_rate * g
parameter += velocity
```

For Adam, the first/second moments use configurable `beta1`/`beta2`; both are
bias-corrected using the update count before division by `sqrt(v_hat) + epsilon`.
Clipping, when enabled, rescales the mean gradient by
`min(1, gradient_clip / global_norm)` before optimization. It applies to the
entire parameter vector, including biases, not separately per layer.
Each successful step clears its accumulated gradients.

The AVX2/FMA implementations are guarded by `__AVX2__`. Scalar loops must cover
non-x86 builds, `NN_ENABLE_AVX2=OFF`, and vector tails of non-multiple widths.
Floating-point validation must remain meaningful: do not enable compiler options
that assume NaNs/infinities cannot occur in the public API's validation code.
Single-sample training has a specialized path with separate affine and activation
loops; general batches retain their own path. Conservative double-precision
gradient magnitude bounds avoid repeated per-element overflow checks only when
overflow is provably impossible; otherwise the checked kernels are used. The
cross-entropy helper reuses the forward softmax normalization instead of computing
the exponentials again.

## 4. Data pipeline and training example

The example reads `label,pixel0,...`, with a header by default. Its C loader also
supports headerless data. Files are read in 1 MiB blocks. Row storage
grows as needed, so long rows are not truncated; CRLF and missing final newlines
are accepted. Capacity is estimated from file size and the first data row.

Integer parsing is locale-independent and checks token boundaries and overflow.
Malformed labels, missing/extra columns and malformed pixels are logged and
skipped, preserving the former loader policy. Blank rows are ignored; embedded
NUL bytes are rejected rather than silently truncating a record. A dataset
with no valid rows is an error. Features are multiplied by `1 / x_max` (255).
Sample and one-hot-label arrays are flat with independent strides.

Loads are transactional: failure preserves the caller's prior dataset/pair and
releases partially loaded buffers. Training/evaluation dimensions must agree.

The trainer shuffles indices, never the large feature array. For each sample it
accumulates the classifier loss/gradient; each full batch steps once. A trailing
partial batch is stepped with its actual sample count. The learning rate is
multiplied by the configured decay after each epoch.
Epoch and evaluation losses accumulate in double precision and are divided
before conversion to the public float metric. Large finite per-sample losses
therefore cannot overflow a representable mean; nonfinite final metrics fail.

The output contract remains:

```
[EPOCH    1] [LOSS ...] [ACCURACY      n out of m]
[EVALUATION] [LOSS ...] [ACCURACY      n out of m]
Time taken: ... seconds
```

`test/smoke_test.cmake` parses the evaluation line. Failures propagate to a nonzero
CLI exit status, including invalid trainer settings (which the original trainer
could report without failing the process).

## 5. Configuration

Defaults live in [src/common.h](../src/common.h).

| Environment variable | Default | Meaning |
|---|---|---|
| `NN_HIDDEN1` | `100` | First hidden-layer width |
| `NN_HIDDEN2` | `50` | Second hidden-layer width |
| `NN_LR` | `0.08` | Initial learning rate |
| `NN_LR_DECAY` | `0.90` | Per-epoch learning-rate multiplier |
| `NN_MOMENTUM` | `0.9` | Momentum in `[0, 1)` |
| `NN_BATCH_SIZE` | `16` | Samples per update |
| `NN_EPOCHS` | `30` | Training epochs |
| `NN_SEED` | `0` | Unsigned 32-bit seed; zero selects a fresh seed |
| `NN_TRAIN_CSV` | `data/fashion-mnist_train.csv` | Training CSV |
| `NN_TEST_CSV` | `data/fashion-mnist_test.csv` | Evaluation CSV |

Unset numeric variables use defaults. **Intentional boundary change:** malformed,
out-of-range or nonfinite numeric values now fail explicitly rather than silently
falling back or narrowing. Epochs, dimensions, batch size, learning rate and decay
must be positive. Unset/empty path variables keep the default paths.
`mlp-cli --help` does not require a dataset.

Adam/activation/clipping selection is an API feature, not additional classifier
hyperparameter flags. The Fashion-MNIST defaults are deliberately unchanged.

## 6. Build and testing

The build uses CMake/Ninja and vendored upstream dd v0.2.0, with library target
`cmlp` and example target `mlp-cli`. The former bespoke drivers are retired.
PowerShell 7.4+ runs dd on native x64 Windows and Linux, with CMake 3.24+.
Supplied Linux presets select GCC, and upstream prerequisite detection requires
`g++` as well, even though this project enables only C. Use a separate direct
CMake configuration for Clang or non-x64 architectures. See [README.md](../README.md) for
commands and [dd-upstream.json](dd-upstream.json) for immutable vendor fingerprints.
Bare `build` and `test` cover both configurations; `build release` selects only
Release. `test --label '^ci$'` runs just the generated suite. Project commands
`scalar --yes` and (Linux) `asan --yes` validate isolated scalar and sanitizer
trees; their `--dry-run` forms skip builds/tests but may still write dd log files.

`NN_ENABLE_AVX2=OFF` selects scalar kernels. Separate Release/Debug and
Windows/Linux build trees prevent one toolchain overwriting another.
Run from the repository root so default dataset paths resolve.
When embedded with `add_subdirectory`/FetchContent, libcmlp adds only the library
by default. `CMLP_BUILD_EXAMPLE=ON` explicitly enables the CLI; repository CTest
and dd-adoption checks are top-level-only and never register in an application.
Standalone tests require the example; disable `BUILD_TESTING` for library-only
standalone builds.

The CTest suite retains **11 tests with real data, 6 without it / with `-L ci`**:

- `generated.dataset`: generates deterministic offline CI data.
- Five `generated.*` smoke cases: runs, loads, accuracy, reproducibility,
  invalid topology.
- Five analogous `dataset.*` cases, registered only for real Fashion-MNIST CSVs.

Additional numerical API tests run inside `generated.runs`; CSV/trainer tests run
inside `generated.loads`. `generated.runs` also configures, builds and runs an
embedding consumer whose CTest inventory must contain only its own test.
This preserves the established count and label contract
without omitting new coverage. The API suite covers both classifier and DQN math,
batching, activation derivatives, copying, checkpoint validation and errors.
The pipeline suite covers storage growth/rollback, malformed/long/headerless CSVs,
CRLF, pair validation and partial batches.

CI checks out without git-lfs and must exercise the six-test `ci` suite. Real-data
tests must never carry the `ci` label. The generated accuracy floor is 150/200
after 20 epochs; real-data smoke uses 7,500/10,000 after one epoch. These loose
floors detect broken learning, **not** full-run accuracy parity.

The separate `windows.yml` / `linux.yml` workflows retain their README badge
names and run on pushes, pull requests and manual dispatch. Each lane validates
dependency declarations and runs `dd doctor`; main lanes run both configurations,
scalar lanes disable AVX2, and Linux's sanitizer lane retains fatal UBSan.
Failure artifacts contain logs and test reports only, not datasets or models.
The existing adoption checks cover these workflow/badge contracts, actual
PowerShell script parsing and rejection of invalid build configurations,
without adding a seventh CI test or a separate test framework.

## 7. Regression gate

Before any P4 source changes, Windows/MSVC Release passed 11/11 tests.
The original full default 30-epoch baseline, measured September 16, 2026:

| Seed | Correct / 10,000 | Wall seconds |
|---|---:|---:|
| 12345 | 8,994 | 34.10 |
| 1 | 8,949 | 33.01 |
| 7 | 8,953 | 34.05 |
| 99 | 8,944 | 36.55 |
| Mean | 8,960 | 34.43 |

The original executable and raw logs are retained locally under
`tmp/p4-baseline/` to permit interleaved controls. The historical reference is
about 38 s MSVC / 28 s GCC, but do not compare different machines/toolchains as a
performance regression test. Accuracy differences under about 100 examples are
within the observed seed noise; validate across several seeds.

The original source revision was also built in an isolated temporary tree under
WSL Ubuntu (GCC 13.3), passing its six generated-data tests. Its full-data runs:

| Seed | Correct / 10,000 | Wall seconds |
|---|---:|---:|
| 12345 | 8,965 | 33.11 |
| 1 | 8,965 | 30.86 |
| 7 | 8,974 | 29.69 |
| 99 | 8,976 | 31.76 |
| Mean | 8,970 | 31.35 |

### Final P4 verification

After the port and performance fixes, all four seed comparisons passed an
absolute accuracy-difference gate of 100 examples. Mean elapsed time was required
to remain within 15% of the interleaved control, on each platform independently.
No other build/test work ran during the final measurements.

| Platform | Seed | Original correct | C11 correct | Original seconds | C11 seconds |
|---|---:|---:|---:|---:|---:|
| MSVC | 12345 | 8,994 | 8,951 | 33.14 | 27.63 |
| MSVC | 1 | 8,949 | 8,995 | 30.74 | 34.24 |
| MSVC | 7 | 8,953 | 8,951 | 40.09 | 36.00 |
| MSVC | 99 | 8,944 | 8,953 | 35.37 | 28.82 |
| GCC | 12345 | 8,965 | 8,955 | 30.24 | 28.55 |
| GCC | 1 | 8,965 | 8,997 | 29.75 | 29.23 |
| GCC | 7 | 8,974 | 8,974 | 30.45 | 30.83 |
| GCC | 99 | 8,976 | 8,948 | 32.78 | 29.79 |

Mean accuracy: **8,960 -> 8,962.5** on Windows; **8,970 -> 8,968.5** on Linux.
Mean time: **34.83 -> 31.67 s** on Windows; **30.80 -> 29.60 s** on Linux.
These establish parity, not an accuracy improvement or a statistically established
speedup. An initial ~34% GCC runtime regression was found and fixed before this
final gate, without fast-math flags or changes to the process floating-point mode.

Final validation:

- `dd test`: **11/11** in Release and Debug on both Windows/MSVC and Linux/GCC.
- `dd scalar --yes`: **6/6** on each platform.
- Linux `dd asan --yes`: **6/6**, with ASan and fatal UBSan.
- Numerical suite: **43,388 checks**, including overflow-bound fallbacks,
  finite-difference gradients, optimizer references, DQN, copying and persistence.
- Separately compiled C++20 consumer links and exercises the C11 API.
- **38/38** dd fingerprints survive a fresh checkout with line-ending conversion;
  a no-lfs checkout configures exactly **6** tests, also with the `ci` filter.
- Separate Windows/Linux workflows are updated and their commands pass locally.
  Remote CI has not been triggered by this local implementation.

The tuning conclusions remain unchanged: geometric LR decay matters, whereas
wider layers, weight decay and shift/flip augmentation did not improve this dense
classifier reliably. ReLU is included for general models/DQN, not claimed as a
Fashion-MNIST accuracy improvement. See [experiments.md](experiments.md).
