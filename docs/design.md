# nn — Design

A dependency-free Fashion-MNIST classifier in C++20. No BLAS, no framework, no third-party
headers: just the standard library plus AVX2 intrinsics in the three hot loops.

This document is the single source of truth for the project's design. `README.md`,
`AGENTS.md`, and `.github/copilot-instructions.md` deliberately stay thin and point here.

[experiments.md](experiments.md) is the companion record of the accuracy tuning search —
what was measured, what was adopted, and what was rejected.

---

## 1. Goals and non-goals

| | |
|---|---|
| **Goal** | Readable reference implementation of a multilayer perceptron trained by mini-batch SGD |
| **Goal** | Fast enough to iterate on: a full 30-epoch run over 60 000 samples in ~36 s on one core |
| **Goal** | Zero dependencies; the whole thing builds from `nn.sln` with MSBuild |
| **Non-goal** | Convolutions, GPU offload, model serialization, multi-threading |
| **Non-goal** | Beating a CNN. A dense net tops out around 89–90 % on Fashion-MNIST |

---

## 2. Module map

```
main.cpp          runtime configuration -> wiring -> timing
  |
  +-- pipeline    dataset_pipeline: CSV -> validated dataset pair
  |     +-- dataset     flat sample storage (X) and one-hot labels (Y)
  |
  +-- nn          the model: topology, weights, forward, backward, update
  |     +-- activation  sigmoid + its derivative
  |     +-- common      xfloat alias and the tuned default constants
  |
  +-- trainer     epoch loop, shuffling, batching, metric aggregation
```

The dependency graph is a DAG with no cycles. `nn` knows nothing about `dataset` or
`trainer`; it only sees raw `const xfloat*` spans. `trainer` is the only component that
knows both, which is what keeps the model reusable against any data source.

### Source layout

| File | Responsibility |
|---|---|
| [src/common.h](src/common.h) | `xfloat` scalar alias and the tuned default hyperparameters |
| [src/activation.h](src/activation.h) | `sigmoid` and `sig_derivative` |
| [src/dataset.h](src/dataset.h), [src/dataset.cpp](src/dataset.cpp) | Contiguous storage for samples and one-hot labels |
| [src/pipeline.h](src/pipeline.h), [src/pipeline.cpp](src/pipeline.cpp) | CSV parsing, normalization, dataset validation |
| [src/nn.h](src/nn.h), [src/nn.cpp](src/nn.cpp) | The MLP and its SIMD kernels |
| [src/trainer.h](src/trainer.h), [src/trainer.cpp](src/trainer.cpp) | Mini-batch SGD loop and reporting |
| [src/main.cpp](src/main.cpp) | Entry point and runtime configuration |

---

## 3. Data pipeline

`dataset_pipeline::load` reads both CSVs and refuses to continue unless the training and
evaluation sets agree on dimensionality.

- **Format**: one row per sample, `label,pixel0,pixel1,...,pixel783`, optional header row.
- **Parsing**: `std::from_chars` over `std::string_view` tokens — no allocation, no `stoi`,
  no locale. Malformed rows are reported and skipped rather than aborting the load.
- **Normalization**: every pixel is multiplied by `1 / x_max` (255) to land in `[0, 1]`.
- **Buffering**: a 1 MiB `pubsetbuf` buffer is installed *before* `open`, because installing
  it afterwards is unspecified behaviour and MSVC ignores it. The buffer is declared before
  the stream so it outlives it.
- **Capacity**: `estimate_samples` divides the file size by the length of the first *data*
  row (not the much longer header) to pre-reserve, which avoids the reallocation churn of
  growing a 47 M-float vector.

`dataset` stores features and labels as two flat `std::vector<xfloat>`, indexed by stride.
This keeps a sample's 784 floats contiguous, which is what makes the forward pass's dot
products streamable. `append_sample` returns a writable span for the parser to fill;
`discard_last_sample` unwinds it when a row turns out to be malformed.

---

## 4. Model

### Topology

Default `784 -> 100 -> 50 -> 10`, 84 060 trainable weights.

- Hidden layers: **sigmoid**.
- Output layer: **softmax**, shifted by the maximum logit for numerical stability.
- Loss: **categorical cross-entropy**.

### Bias handling

There is no separate bias vector. Every layer except the output carries one extra activation
slot pinned to `1.0`, and the weight matrix of the following connection is one column wider.
A bias is therefore just another weight, trained by exactly the same code path — no special
case in the forward pass, the gradient accumulation, or the update.

### Memory layout

All per-layer state lives in **one flat vector per kind**, with each layer holding an offset
into it:

| Buffer | Contents |
|---|---|
| `activations` | forward values, including the pinned bias slots |
| `deltas` | backpropagated error per non-input neuron |
| `weights` | row-major `[output_neuron][input_activation]` per connection |
| `gradient_accumulators` | batch-summed gradients, same shape as `weights` |
| `velocity` | momentum state, same shape as `weights` |

One allocation per kind instead of one per layer means no pointer chasing between layers and
a much friendlier prefetch pattern. `weights`, `gradient_accumulators`, and `velocity` share
the *same* offset table, so the update loop walks three parallel arrays in lockstep.

**Row-major is the load-bearing decision.** A neuron's weights are contiguous, so the forward
dot product, the gradient accumulation, and the delta propagation are all unit-stride.

### Initialization

Xavier/Glorot uniform: `U(-limit, +limit)` with `limit = sqrt(6 / (fan_in + fan_out))`.
Seeded from `nn_config::seed`, or from `std::random_device` when the seed is `0`. The seed is
what makes `dd test` able to assert reproducibility.

---

## 5. Training

### The step

`trainer::fit` owns the loop; `nn` owns the math. Per epoch:

1. Shuffle an index vector (never the data itself — 47 M floats stay put).
2. For each sample: `accumulate_gradients` — forward, loss, backward, gradient accumulation.
3. Every `batch_size` samples: `apply_batch`.

`apply_batch` folds `1 / batch_size` into the learning rate rather than dividing the
gradients, saving a pass over all 84 060 accumulators. It applies classical momentum and
**zeroes the accumulators in the same pass**, so there is no separate "begin batch" clear —
one streaming read-modify-write over each buffer instead of two.

The trailing partial batch is applied with its true count, so it is scaled correctly rather
than being dropped or over-weighted.

### Learning-rate schedule

The learning rate is multiplied by `learning_rate_decay` after every epoch — geometric decay,
the cheapest schedule that works. It is worth more than any other single hyperparameter here:
a high initial rate (`0.08`) covers ground early, and the decay lets the run settle instead of
bouncing around the minimum. Measured across four seeds it is worth **+2.0 percentage points**
over a flat rate (see §8).

The default `0.90` over 30 epochs shrinks the rate to ~4 % of its initial value. When changing
the epoch count, keep that end-to-end factor roughly constant: `decay^epochs ≈ 0.04`.

### Backpropagation

The output delta is `softmax - one_hot`, which is the exact gradient of cross-entropy with
respect to the pre-softmax logits — the softmax Jacobian and the log both cancel. That is why
the output layer needs no derivative term.

Hidden deltas are computed **row-wise**, not column-wise:

```
zero the delta vector
for each downstream neuron j:
    delta[0..width) += weight_row_j[0..width) * next_delta[j]
then multiply elementwise by sig_derivative(activation)
```

The obvious formulation walks the weight matrix down a column (`w[j * input_width + i]`),
which strides by `input_width` floats per step: cache-hostile and impossible to vectorize.
Accumulating along rows instead makes the inner loop unit-stride and lets it reuse the same
`scaled_accumulate` FMA kernel as the gradient pass. The bias column is skipped because a
constant input has no error to receive.

### SIMD kernels

Three functions in the anonymous namespace of [src/nn.cpp](src/nn.cpp) carry essentially all
the runtime. Each is a plain scalar loop guarded by `#if defined(__AVX2__)`, so the Debug and
Win32 configurations still compile and produce identical results.

| Kernel | Used by | Notes |
|---|---|---|
| `dot_product` | forward pass | Two independent FMA accumulators to hide the ~4-cycle latency, 16 floats per iteration, then a 8-wide tail |
| `scaled_accumulate` | gradient accumulation, delta propagation | Fused `dst += scale * src` |
| `apply_momentum_update` | `apply_batch` | Velocity, weight step, and gradient clear in one pass; `_mm256_fnmadd_ps` for `momentum*v - lr*g` |

`horizontal_sum_avx` reduces a `__m256` once per neuron, which is negligible next to the
784-element dot product it terminates.

---

## 6. Configuration

Defaults live in exactly one place, [src/common.h](src/common.h), and every one of them is
overridable at runtime through an environment variable. Nothing needs recompiling to tune.

| Variable | Default | Meaning |
|---|---|---|
| `NN_HIDDEN1` | `100` | First hidden layer width |
| `NN_HIDDEN2` | `50` | Second hidden layer width |
| `NN_LR` | `0.08` | Initial learning rate |
| `NN_LR_DECAY` | `0.90` | Learning rate multiplier applied after each epoch |
| `NN_MOMENTUM` | `0.9` | Momentum coefficient |
| `NN_BATCH_SIZE` | `16` | Samples per weight update |
| `NN_EPOCHS` | `30` | Passes over the training set |
| `NN_SEED` | `0` | Seeds weight init and shuffling; `0` means nondeterministic |

Malformed values fall back to the default rather than failing. Values that are *parseable but
invalid* (a zero-width layer, a negative epoch count, momentum outside `[0, 1)`) are rejected
at the boundary — `nn::compile` throws and `trainer::fit` refuses to run.

---

## 7. Build and run

Everything goes through [dd.ps1](dd.ps1):

```powershell
.\dd.ps1 build   # MSBuild Release|x64 -> bin\nn.exe
.\dd.ps1 run     # build, then train and evaluate
.\dd.ps1 test    # build, then run the smoke suite
```

`run` and `test` execute from the repository root so the relative `.\data\...` paths resolve.

The Release x64 configuration sets `/arch:AVX2`, `/fp:fast`, whole-program optimization, and
`stdcpp20`. AVX2 is only enabled for Release x64 — the other configurations take the scalar
paths.

### Testing

There is no test framework, by design. `dd.ps1 test` drives the real binary and asserts on
its output:

| Case | Catches |
|---|---|
| Exits cleanly | Crashes, unhandled exceptions |
| Datasets load | Broken CSV parsing, missing data files |
| ≥ 7500 / 10000 after one epoch | A broken forward or backward pass — a subtly wrong gradient collapses this immediately |
| Seeded runs are byte-identical | Unseeded randomness leaking into the run |
| Zero-width hidden layer rejected | Missing validation at the configuration boundary |

The accuracy floor is deliberately loose. It is a regression tripwire, not a quality bar; a
correct network clears ~85 % after a single epoch.

---

## 8. Results

Reference run, `784 -> 100 -> 50 -> 10`, seed 12345, defaults, single core:

| Metric | Value |
|---|---|
| Training accuracy (epoch 30) | 56 526 / 60 000 |
| Evaluation accuracy | 8 994 / 10 000 (89.9 %) |
| Evaluation loss | 0.301 |
| Wall clock | ~36 s |

### Tuning history

The defaults above came out of a documented search — see
[experiments.md](experiments.md) for the full record, raw measurements, and method.
Mean evaluation accuracy over four seeds (12345, 1, 7, 99) at the default topology:

| Configuration | Mean | Wall clock |
|---|---|---|
| 10 epochs, flat `lr = 0.04` (original) | 8 760 | ~12 s |
| 10 epochs, `lr = 0.10`, decay `0.70` | 8 898 | ~13 s |
| 30 epochs, `lr = 0.08`, decay `0.90` (**default**) | 8 960 | ~36 s |

The decay schedule is the whole story: at *identical* runtime it is worth +1.4 points, and
allowing three times the epochs buys another +0.6.

If runtime matters more than the last half point, the fast profile is a one-liner:

```powershell
$env:NN_EPOCHS='10'; $env:NN_LR='0.10'; $env:NN_LR_DECAY='0.70'
.\dd.ps1 run
```

### Things that did not work

Summarized here so they are not re-attempted; measurements and reasoning are in
[experiments.md](experiments.md). Each was fully implemented, measured against a control, and
reverted.

| Change | Result | Why |
|---|---|---|
| **ReLU hidden units** (with He init) | 8 958 vs 8 994 | No better than sigmoid at this depth, and it needs a ~4× lower learning rate — at `0.04` it diverges into dead units. Rectifiers pay off with depth this network does not have. |
| **L2 weight decay** | flat to `1e-3`, 8 740 at `1e-2` | Does not touch the generalization gap. The gap is representational, not a weight-magnitude problem. |
| **Shift augmentation** (±1–2 px) | 8 893 / 8 677 | *Actively harmful.* An MLP has no translation invariance, so every input pixel is a distinct feature. Shifting destroys the alignment the network depends on and forces it to relearn each offset. This is a CNN technique. |
| **Horizontal flip augmentation** | 8 969 | Mildly harmful. Several Fashion-MNIST classes (shoes, bags) have a consistent orientation that the flip discards. |
| **Wider layers** (256/128, 300/100) | 8 989 / 8 980 | Within noise of the 100/50 default for 3× the runtime. Capacity is not the binding constraint. |
| **Batch size 8 or 32** (LR rescaled) | 8 981 / 8 991 | No effect. 16 is not a delicate choice. |

The pattern is consistent: past ~89 %, this architecture is the limit, not the optimizer.
Training accuracy reaches 94 % while evaluation sits at 90 %, and no regularizer closes that
gap because a dense network simply has no way to encode spatial structure. Going meaningfully
past 90 % requires convolutions — which is a different project.

For context on what is achievable:

- Simple MLP: 87–89 %
- Well-tuned dense model: ~90 %
- CNN: low 90s and above

[py/torch-test.py](py/torch-test.py) is a PyTorch baseline kept for cross-checking; it uses
ReLU and dropout and is not intended to mirror the C++ model exactly.

### Reproducing a sweep

Set `NN_SEED` and override the variables from §6. Clear *every* `NN_*` variable between runs —
a leaked setting silently contaminates the next configuration, which is easy to miss because
the run still succeeds. Validate any claimed gain on at least three seeds; the single-run
spread on this model is roughly ±65 samples.

---

## 9. Conventions

- `snake_case` for types, functions, and variables; capitals for compile-time constants.
- `xfloat` everywhere a scalar is stored, so the precision is switchable in one line.
- Tabs in `.cpp` files, four spaces in `.h` files — matching what is already there.
- Validate at boundaries (`compile`, `load`, `reset`) and trust internal calls afterwards.
- Hot loops take `const xfloat*` and a count. No iterators, no `std::function`, no virtuals
  anywhere in the inner loops.
- Comments explain *why*, not *what*. Most functions have none, and that is correct.
