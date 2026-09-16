# Accuracy tuning experiment

A record of the hyperparameter and architecture search run against this network, kept so the
negative results are not re-attempted and the positive one can be re-verified.

This is a historical record of the pre-P4 C++ classifier. Old executable names,
commands and random sequences below refer to that measured implementation; the
current C11 API and P4 regression measurements are in [design.md](design.md).

**Outcome:** evaluation accuracy went from **8 806 to 8 994 out of 10 000** (88.1 % → 89.9 %)
on the reference seed, and from a **8 760 to 8 960** four-seed mean. Exactly one change
survived: a geometric learning-rate schedule. Everything else was measured and reverted.

For the resulting design see [design.md](design.md); this document is the working record
behind §8 of it.

---

## 1. Starting point

| | |
|---|---|
| Topology | `784 -> 100 -> 50 -> 10`, sigmoid hidden, softmax output |
| Optimizer | mini-batch SGD, momentum `0.9`, batch `16` |
| Schedule | flat `lr = 0.04`, 10 epochs |
| Result | 8 806 / 10 000 eval, 53 712 / 60 000 train, ~12 s |

The opening diagnosis came from the gap between those two numbers: **89.5 % train vs 88.1 %
eval**. A 1.4-point gap is small, which says the model was *underfitting* — it had not yet
extracted what it could from the data, let alone started memorizing it. That ruled out
regularization as a first move and pointed at capacity and schedule instead.

## 2. Method

All runs used a fixed seed so that weight initialization and shuffle order were identical
across configurations, isolating the variable under test. Runs were driven by a throwaway
harness that set `NN_*` environment variables and scraped the `[EVALUATION]` line:

```powershell
foreach ($c in $Configs) {
    # Clear every NN_* variable, otherwise settings leak from the previous configuration.
    Get-ChildItem Env: | Where-Object { $_.Name -like 'NN_*' } | ForEach-Object {
        [Environment]::SetEnvironmentVariable($_.Name, $null)
    }
    foreach ($e in $c.Env.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($e.Key, [string]$e.Value)
    }
    [Environment]::SetEnvironmentVariable('NN_SEED', [string]$Seed)
    $out = & .\build\windows-release\nn.exe 2>&1 | ForEach-Object { $_.ToString() }
    $out | Where-Object { $_ -match '\[EVALUATION\]' }
}
```

Two rules that mattered:

- **Every round carries a control.** The unmodified configuration is re-run inside the same
  round rather than compared against a remembered number.
- **Only accept a result that survives multiple seeds.** Single-run spread on this model is
  roughly ±65 samples (see §9), so anything under ~100 is noise.

## 3. Round 1 — capacity and epochs

Testing the underfitting hypothesis. Seed 12345.

| Configuration | Eval | Train | Time |
|---|---|---|---|
| 100/50, e10, lr 0.04 *(control)* | 8 806 | 53 712 | 11.9 s |
| 100/50, e30, lr 0.04 | 8 905 | 55 830 | 36.0 s |
| 256/128, e30, lr 0.04 | 8 967 | 56 317 | 104.4 s |
| 256/128, e30, lr 0.10 | 8 892 | 55 029 | 94.1 s |

Underfitting confirmed — both more epochs and more width helped. Also an early signal that
the learning rate was the sensitive knob: raising it to 0.10 *cost* 75 samples.

## 4. Round 2 — ReLU

The standard advice is that sigmoid saturates and ReLU trains better. Implemented as a
selectable activation with He initialization (`sqrt(6 / fan_in)`) to match. Seed 12345, e10.

| Configuration | Eval | Train | Time |
|---|---|---|---|
| sigmoid, lr 0.04 *(control)* | 8 806 | 53 712 | 12.5 s |
| ReLU, lr 0.04 | 7 303 | 42 309 | 11.9 s |
| ReLU, lr 0.02 | 8 703 | 52 948 | 14.8 s |
| ReLU, lr 0.01 | **8 808** | 53 762 | 15.7 s |
| ReLU, lr 0.005 | 8 788 | 53 941 | 14.2 s |

At its best ReLU tied sigmoid (8 808 vs 8 806) and needed a 4× lower learning rate to get
there; at the sigmoid rate it collapsed to 7 303 as units died. **No gain.** Rectifiers earn
their reputation in deep networks where sigmoid's vanishing gradient compounds over many
layers — with two hidden layers there is not enough depth for that to bite.

## 5. Round 3 — learning-rate decay

Added a per-epoch multiplier on the learning rate. This is the round that worked. Seed 12345,
e30.

| Configuration | Eval | Train |
|---|---|---|
| sigmoid, lr 0.04, no decay *(control)* | 8 905 | 55 830 |
| sigmoid, lr 0.06, decay 0.95 | 8 948 | 56 501 |
| sigmoid, lr 0.08, decay 0.90 | **8 994** | 56 526 |
| ReLU, lr 0.01, no decay | 8 935 | 55 489 |
| ReLU, lr 0.02, decay 0.95 | 8 852 | 56 024 |

+89 over the round control and +188 over the original baseline. The mechanism is the obvious
one: a high initial rate covers ground quickly, and shrinking it lets the run settle into the
minimum instead of bouncing around it. Note ReLU again failed to keep up once sigmoid had a
proper schedule.

### Round 4 — refining the schedule

| Configuration | Eval | Train | Time |
|---|---|---|---|
| e30, lr 0.08, decay 0.90 *(best so far)* | **8 994** | 56 526 | 34.8 s |
| e30, lr 0.10, decay 0.90 | 8 973 | 56 445 | 40.5 s |
| e30, lr 0.08, decay 0.85 | 8 979 | 55 866 | 38.1 s |
| e50, lr 0.08, decay 0.93 | 8 972 | 57 480 | 62.1 s |
| 256/128, e30, lr 0.08, decay 0.90 | 8 989 | 56 919 | 99.2 s |

A broad plateau — the schedule is not delicate. Note the e50 run: train climbed to 57 480
(95.8 %) while eval *fell* to 8 972. The model had crossed from underfitting into
overfitting, which set up the next two rounds.

## 6. Round 5 — augmentation (and a contaminated result)

With overfitting now visible, augmentation was the obvious regularizer. Implemented random
translation and horizontal mirroring in the trainer.

The first sweep produced this:

| Configuration | Eval | Train |
|---|---|---|
| control | 8 994 | 56 526 |
| flip only | 8 969 | 55 499 |
| shift 1 | 8 841 | 53 179 |
| shift 2 + flip | 8 677 | 51 885 |
| shift 1 + flip | 8 841 | 53 179 |

**`shift 1` and `shift 1 + flip` are byte-identical.** Two configurations that consume the
random number generator differently cannot produce the same result, so the harness was wrong,
not the model. The cause: it cleared a hardcoded list of variable names that predated
`NN_AUGMENT_SHIFT` and `NN_AUGMENT_FLIP`, so the flip flag set in row 2 stayed set for every
later row. Rows 3–5 all ran with flip enabled, and the "shift only" case was never measured.

This is the failure mode worth remembering: **a leaked setting does not cause an error.**
Every run succeeded and every number looked plausible. Only the impossible coincidence of two
identical rows exposed it. The fix was to clear all `NN_*` variables by enumeration rather
than by list.

Re-run with a clean environment:

| Configuration | Eval | Train | Time |
|---|---|---|---|
| control, e30 | **8 994** | 56 526 | 34.7 s |
| shift 1 only, e30 | 8 893 | 53 537 | 34.5 s |
| shift 1, e60, decay 0.95 | 8 908 | 54 280 | 50.6 s |
| control, e60, decay 0.95 | 8 931 | 58 032 | 56.1 s |

Augmentation is **actively harmful**, and the longer schedule did not rescue it. The reason is
structural: a dense layer has no translation invariance, so input pixel 391 is simply a
different feature from pixel 392. Shifting the image does not teach the network that a sleeve
is a sleeve wherever it appears — it destroys the pixel alignment the network entirely depends
on and forces it to spend capacity relearning every offset. Fashion-MNIST is also
pre-centered, so the augmentation adds variation the test set does not contain. Flipping fails
for a different reason: shoes and bags face a consistent direction, and mirroring discards a
genuine class cue.

Shift augmentation is a convolutional technique. Reverted.

## 7. Round 6 — L2 weight decay

The other standard answer to overfitting. Two extra FMAs in the momentum update. Seed 12345,
e30, lr 0.08, decay 0.90.

| Weight decay | Eval | Train |
|---|---|---|
| 0 *(control)* | **8 994** | 56 526 |
| 1e-5 | 8 990 | 56 426 |
| 1e-4 | 8 994 | 56 427 |
| 1e-3 | 8 985 | 55 900 |
| 1e-2 | 8 740 | 52 715 |

Completely flat until it starts hurting. The generalization gap is not caused by oversized
weights, so shrinking them changes nothing. Reverted.

## 8. Round 7 — remaining knobs

| Configuration | Eval | Train | Time |
|---|---|---|---|
| batch 16, lr 0.08 *(control)* | **8 994** | 56 526 | 36.3 s |
| batch 8, lr 0.04 | 8 981 | 56 344 | 35.9 s |
| batch 32, lr 0.15 | 8 991 | 56 312 | 34.5 s |
| ReLU, lr 0.02, decay 0.90 | 8 958 | 56 057 | 34.1 s |
| 300/100, lr 0.08 | 8 980 | 56 852 | 115.6 s |

All within noise of the control. Batch size is not a delicate choice once the learning rate is
scaled with it, and extra width buys nothing for 3× the runtime.

### A fast schedule

Since the schedule rather than the epoch count was doing the work, the same idea was retuned
for a 10-epoch budget, holding the end-to-end shrink factor (`decay^epochs ≈ 0.04`) roughly
constant.

| Configuration | Eval | Train | Time |
|---|---|---|---|
| e10, lr 0.04, no decay *(original)* | 8 806 | 53 712 | 12.8 s |
| e10, lr 0.08, decay 0.72 | 8 913 | 54 357 | 13.7 s |
| e10, lr 0.10, decay 0.70 | **8 919** | 54 402 | 14.4 s |
| e10, lr 0.06, decay 0.80 | 8 903 | 54 340 | 14.4 s |
| e15, lr 0.08, decay 0.80 | 8 936 | 55 056 | 20.1 s |

+113 at the original runtime.

## 9. Round 8 — seed robustness

Everything above is single-seed. The two candidate schedules were re-run against the original
baseline on four seeds before being adopted.

| Configuration | 12345 | 1 | 7 | 99 | Mean |
|---|---|---|---|---|---|
| e10, lr 0.04, no decay *(original)* | 8 806 | 8 675 | 8 775 | 8 785 | **8 760** |
| e10, lr 0.10, decay 0.70 | 8 919 | 8 871 | 8 917 | 8 885 | **8 898** |
| e30, lr 0.08, decay 0.90 | 8 994 | 8 949 | 8 953 | 8 944 | **8 960** |

Both improvements hold on every seed. This round also quantifies the noise floor: the original
configuration spans 8 675–8 806 across seeds, a 131-sample range, which is why the ±40 sample
differences in rounds 6 and 7 were treated as no result.

## 10. Conclusions

**Adopted:** geometric learning-rate decay, and defaults retuned to `lr = 0.08`,
`decay = 0.90`, 30 epochs. Worth +2.0 points on the four-seed mean; +1.4 of that is available
at no runtime cost at all.

**Reverted:** ReLU with He initialization, L2 weight decay, shift augmentation, flip
augmentation, wider topologies, alternative batch sizes. Each was fully implemented and
measured; none earned its keep, so none stayed in the code.

The recurring theme is that the schedule was the only thing left on the table, and once it was
fixed the model hit an architectural wall. Training accuracy sits at 94 % against 90 %
evaluation, and no regularizer closed that gap because it is not an optimization problem — a
fully connected network has no mechanism for encoding spatial structure, so it cannot learn
that neighbouring pixels are related or that a feature means the same thing in two places.
Every technique that failed here (rectifiers, augmentation) is one that pays off specifically
in deep or convolutional models.

~90 % is the ceiling for this architecture. Passing it means convolutions, which is a
different project.
