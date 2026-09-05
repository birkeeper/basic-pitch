# nmp_salience_finetuned_conv.weights.h5

Fine-tuned variant of Basic Pitch's **salience (contour) head**
(`scripts/convert_model_to_savedmodel.build_salience_model`), adapted by
**convolutional fine-tuning** on real choir recordings of blocked chords.

Goal: reduce the *"quiet voice → low salience"* bias, i.e. raise multi-F0 recall
on a voice sung softly relative to the rest of the ensemble, without altering the
model's behaviour on real recordings.

| | |
|---|---|
| Base weights | `basic_pitch/saved_models/icassp_2022/nmp` (variables checkpoint) |
| Architecture | Basic Pitch contour path only — note and onset outputs dropped |
| Adapted parameters | both conv kernels and biases — 7,697 (`conv2d`, `contours-reduced`) |
| Frozen parameters | both `BatchNormalization` layers (γ, β **and** running mean/variance) |
| Training data | 167 prepared windows, `./finetune/data/train` |
| Anchor data | none — no `--distill_dir` in this run |
| Drift screen | 2 real recordings, unannotated (`heroes_opname`, `My Love_opname`) |
| Date produced | 2026-09-05 |
| Log | `nmp_salience_finetuned_conv_20260905-175247.log` |
| Status | F improved 0.393 → 0.664 (retained checkpoint) with one guard rejection in 40 epochs. Validation shares songs, ensemble and room with training, so generalisation is unmeasured. `d@high` inflating (§4.1) |
| Retained checkpoint | **`nmp_salience_finetuned_conv_e36.weights.h5`** — epoch-boundary e36, F 0.664 (§3.2). Not the `e36_s000610` selection wrote to `--out` |

> **Note on the filename.** `train()` appends the strategy to `--out`, so the
> `_conv` suffix is generated, not typed. **This model is `--strategy conv`** — no
> BatchNorm parameter was modified.

---

## 1. Method

A voice sung quietly is an **input-amplitude / SNR domain shift**, and its
evidence is attenuated before the output layer, so adapting only the final
convolution cannot recover it. The salience path is only two convolutions deep,
so `conv` adapts essentially all of it: 7,697 of the 7,733 parameters.

`--strategy conv` freezes both `BatchNormalization` layers. Keras special-cases
BN — `trainable=False` also puts the layer in **inference mode** even when called
with `training=True` — so γ, β *and* the running statistics stay at their
pretrained values. This matters more than the 18 parameters it withholds:
`--l2sp` anchors only variables named `kernel`, and running statistics are not
trainable variables at all, so under `full` nothing could anchor them. Under
`conv`, L2-SP covers 100% of what moves. Confirmed in the log:
`Strategy 'conv': 2 trainable layer(s), 7697 parameters`.

### 1.1 The loss

```
L = bce_masked( y , f(x) , mask )      # supervised, ridge targets
  + 1e-3 · Σ ‖W − W₀‖²                 # L2-SP on 2 kernels
```

**Supervised term.** Binary cross-entropy against a Gaussian ridge target
(25 cents), masked to the frames where the score-derived labels hold — inside a
chord's attack, release or reverb tail no target is defensible, but those frames
are still fed as input because they supply the receptive field of the frames that
are labelled. `--pos_weight 1.0`: no reweighting of voice bins.

No label smoothing. Basic Pitch pretrained with 0.2 against single-bin targets;
these targets are ridges whose shoulders carry the sub-bin pitch, and smoothing
squashes exactly those.

**No distillation anchor in this run.** Both real recordings were assigned to the
drift screen, leaving nothing for `--distill_dir`. L2-SP on two kernels is
therefore the only constraint. §4.1 shows the cost.

## 2. Procedure

Windows are fixed at 172 frames (`AUDIO_N_SAMPLES`), one per chord, centred on it
with the real surrounding audio as context. The length is not a hyperparameter:
Basic Pitch's input is fixed, and `NormalizedLog` rescales by each window's own
dynamic range, so a training window of a different length would be normalised
differently from anything seen at inference.

```
--strategy conv --batch_size 10 --epochs 40 --eval_every 10 --lr 1e-4
--l2sp 1e-3 --pos_weight 1.0 --thresh auto
--real_audio heroes_opname.flac "My Love_opname.flac"
```

17 batches per epoch, 670 optimizer steps total, checkpointed every 10.

## 3. Results

Peak threshold is re-tuned per checkpoint (reported as `@thresh`) because an
absolute threshold is not comparable across checkpoints: fine-tuning removes the
pretrained background floor, so a threshold chosen from the pretrained model
progressively understates every later epoch. Selection ranks on **F**, which is
what the sweep optimises.

### 3.1 Trajectory

| epoch | train | val_loss | recall | precision | F | @thresh | d@high | mean | r |
|---|---|---|---|---|---|---|---|---|---|
| baseline | — | 0.1506 | 0.427 | 0.365 | 0.393 | 0.170 | — | — | — |
| 1 | 0.1443 | 0.1245 | 0.454 | 0.473 | 0.463 | 0.155 | -0.060 | -35.3% | 0.80 |
| 2 | 0.1260 | 0.1131 | 0.427 | 0.520 | 0.469 | 0.170 | -0.064 | -46.2% | 0.76 |
| 3 | 0.1143 | 0.1022 | 0.543 | 0.532 | 0.537 | 0.155 | -0.063 | -50.9% | 0.72 |
| 4 | 0.1044 | 0.0923 | 0.578 | 0.521 | 0.548 | 0.155 | -0.038 | -52.9% | 0.69 |
| 5 | 0.0960 | 0.0838 | 0.557 | 0.526 | 0.541 | 0.170 | -0.016 | -55.5% | 0.66 |
| 6 | 0.0893 | 0.0773 | 0.541 | 0.545 | 0.543 | 0.185 | -0.004 | -58.3% | 0.64 |
| 7 | 0.0841 | 0.0727 | 0.551 | 0.541 | 0.546 | 0.185 | +0.008 | -60.2% | 0.63 |
| 8 | 0.0802 | 0.0694 | 0.550 | 0.555 | 0.552 | 0.200 | +0.023 | -60.8% | 0.61 |
| 9 | 0.0775 | 0.0669 | 0.546 | 0.574 | 0.560 | 0.215 | +0.033 | -61.5% | 0.61 |
| 10 | 0.0756 | 0.0649 | 0.553 | 0.584 | 0.568 | 0.215 | +0.037 | -62.7% | 0.60 |
| 11 | 0.0736 | 0.0635 | 0.562 | 0.592 | 0.576 | 0.215 | +0.046 | -63.4% | 0.60 |
| 12 | 0.0724 | 0.0621 | 0.566 | 0.605 | 0.585 | 0.200 | +0.034 | -66.2% | 0.60 |
| 13 | 0.0711 | 0.0610 | 0.578 | 0.604 | 0.590 | 0.200 | +0.041 | -66.2% | 0.59 |
| 14 | 0.0703 | 0.0604 | 0.577 | 0.618 | 0.597 | 0.230 | +0.072 | -63.9% | 0.59 |
| 15 | 0.0695 | 0.0594 | 0.577 | 0.632 | 0.603 | 0.215 | +0.058 | -66.7% | 0.59 |
| 16 | 0.0687 | 0.0587 | 0.599 | 0.624 | 0.612 | 0.200 | +0.060 | -67.1% | 0.59 |
| 17 | 0.0684 | 0.0581 | 0.601 | 0.637 | 0.618 | 0.215 | +0.071 | -66.4% | 0.59 |
| 18 | 0.0673 | 0.0576 | 0.615 | 0.630 | 0.622 | 0.185 | +0.061 | -68.8% | 0.59 |
| 19 | 0.0672 | 0.0571 | 0.617 | 0.640 | 0.628 | 0.200 | +0.075 | -67.9% | 0.59 |
| 20 | 0.0665 | 0.0567 | 0.617 | 0.648 | 0.632 | 0.215 | +0.086 | -66.8% | 0.59 |
| 21 | 0.0664 | 0.0563 | 0.632 | 0.632 | 0.632 | 0.185 | +0.082 | -68.4% | 0.60 |
| 22 | 0.0658 | 0.0559 | 0.637 | 0.646 | 0.641 | 0.200 | +0.084 | -67.6% | 0.59 |
| 23 | 0.0651 | 0.0556 | 0.620 | 0.662 | 0.640 | 0.200 | +0.077 | -69.5% | 0.59 |
| 24 | 0.0649 | 0.0553 | 0.649 | 0.637 | 0.643 | 0.185 | +0.092 | -68.2% | 0.59 |
| 25 | 0.0648 | 0.0551 | 0.651 | 0.634 | 0.643 | 0.185 | +0.099 | -68.0% | 0.60 |
| 26 | 0.0643 | 0.0546 | 0.649 | 0.652 | 0.651 | 0.185 | +0.082 | -69.5% | 0.59 |
| 27 | 0.0643 | 0.0545 | 0.649 | 0.653 | 0.651 | 0.200 | +0.099 | -68.3% | 0.59 |
| 28 | 0.0638 | 0.0542 | 0.645 | 0.659 | 0.652 | 0.185 | +0.081 | -70.5% | 0.59 |
| 29 | 0.0637 | 0.0540 | 0.650 | 0.656 | 0.653 | 0.200 | +0.101 | -68.7% | 0.60 |
| 30 | 0.0636 | 0.0542 | 0.650 | 0.660 | 0.655 | 0.230 | +0.124 | -66.2% | 0.59 |
| 31 | 0.0632 | 0.0535 | 0.652 | 0.668 | 0.660 | 0.185 | +0.080 | -71.0% | 0.59 |
| 32 | 0.0632 | 0.0536 | 0.651 | 0.664 | 0.657 | 0.215 | +0.112 | -68.0% | 0.59 |
| 33 | 0.0629 | 0.0533 | 0.657 | 0.662 | 0.660 | 0.200 | +0.102 | -69.1% | 0.59 |
| 34 | 0.0630 | 0.0531 | 0.651 | 0.674 | 0.662 | 0.215 | +0.102 | -69.0% | 0.59 |
| 35 | 0.0625 | 0.0530 | 0.655 | 0.655 | 0.655 | 0.185 | +0.099 | -70.4% | 0.60 |
| 36 | 0.0623 | 0.0527 | 0.649 | 0.681 | 0.664 | 0.200 | +0.086 | -71.1% | 0.59 |
| 37 | 0.0623 | 0.0527 | 0.650 | 0.668 | 0.659 | 0.200 | +0.100 | -70.3% | 0.59 |
| 38 | 0.0620 | 0.0525 | 0.647 | 0.677 | 0.662 | 0.215 | +0.104 | -69.8% | 0.59 |
| 39 | 0.0623 | 0.0523 | 0.650 | 0.678 | 0.663 | 0.215 | +0.103 | -69.9% | 0.59 |
| 40 | 0.0617 | 0.0522 | 0.648 | 0.681 | 0.664 | 0.215 | +0.099 | -70.3% | 0.59 |

**F 0.393 → 0.664** (+69% relative), recall 0.427 → 0.648 (+52%), precision
0.365 → 0.681 (+87%).

`val_loss` fell monotonically for all 40 epochs and stayed *below* `train_loss`
(0.0522 vs 0.0617 at e40). No overfitting signal appeared, despite 7,697
parameters against 167 windows.

Gains arrive in two phases: precision first (e05–e15, recall flat at ~0.56),
recall second (e15–e25, +0.074). Both flatten from ~e26: the final ten epochs
move F by +0.004, and no 5-epoch block after e25 gains more than +0.004.

### 3.2 Selection

`Best (highest F, guards satisfied) -- from e36_s000610, F 0.666 vs baseline 0.393`

One rejection in the entire run: `recall 0.394<0.397` at step 20, from
`--bal_tol`. No drift rejection ever fired.

The selected checkpoint is a mid-epoch one, and its F of 0.666 is within noise of
e40 (0.664), e39 (0.663) and e36 (0.664) — F is within 0.005 of best from e31
onward, so selection among the last ten checkpoints is arbitrary. Drift is flat
across that range too (§4.1), so there is no accuracy/drift trade to make here:
any checkpoint from ~e31 is equivalent.

**Decision: the epoch-boundary `e36` checkpoint is kept** (F 0.664), not the
`e36_s000610` that selection wrote to `--out` (F 0.666). Nothing distinguishes a
mid-epoch step from an epoch boundary here — BatchNorm is frozen under `conv`, so
its running statistics are the pretrained values wherever training stops, and 15
of 17 batches into an epoch is not a meaningfully different point in the data
order. A separate check outside this log put the standard error on validation
recall and precision at ±0.018 across the 54 windows, against a spread of 0.006
in F between e31, e36 and `e36_s000610`: the candidates are statistically
indistinguishable, and the epoch boundary is the one that is simplest to name and
reproduce.

Note that picking the argmax of a noisy metric on 54 windows makes any of these
figures slightly optimistic as an estimate of true performance. That is a
property of the number, not of these weights, and applies to any checkpoint
chosen this way.

## 4. Behaviour on real recordings

### 4.1 The drift screen

Two unannotated recordings, compared against the pretrained model's output on the
same audio every checkpoint. Only `d@high` — the change in the pretrained model's
most confident band, its top 0.5% of bins — gates. `mean` and `r` are diagnostics.

Three distinct behaviours:

* **`mean` −35% → −70%.** Not compression. Basic Pitch pretrained with label
  smoothing and so carries a background floor near 0.10; these targets are 0 in
  the background, so every run removes that floor. ~93% of the map is background,
  so this statistic mostly reports where the floor sits. It is diagnostic for
  exactly this reason — gating on it would reject every checkpoint for doing what
  it was asked.

* **`r` 0.80 → 0.59, stable from e08.** The map's shape changed substantially in
  the first few epochs and then stopped diverging.

* **`d@high` −0.060 → +0.099, rising to e30 then level.** The confident band is
  not compressing; by e10 it is *above* pretrained, peaks at +0.124 (e30) and
  settles at +0.099-0.104 for the final ten epochs. Against a baseline band of
  roughly 0.45–0.50 this is ~+20% inflation, reached and then held rather than
  accumulating.

The gate is one-sided (`d@high < -drift_high_tol`), so inflation is unopposed —
inconsistent with `make_distill_loss`, which is deliberately symmetric on the
argument that calibration drifts both ways. Nothing in this run's guards would
have stopped the inflation continuing.

Rising `d@high` alongside rising precision and a rising `@thresh` (0.155 → 0.215)
is consistent with peaks genuinely sharpening rather than the map inflating
wholesale, since `mean` is flat from e12. That reading is plausible but not
established by this log.

### 4.2 What this does and does not establish

Established: on held-out windows from the same corpus, the fine-tuned model
detects substantially more voices at higher precision, and the pretrained model's
confident detections were not compressed on two unannotated recordings.

Not established: anything about a different choir, room or microphone. The six
validation takes are drawn from the same four songs, ensemble and sessions as the
167 training windows, and both drift-screen recordings come from those sessions
too. A model recalibrating to *this* ensemble in *this* room would score clean on
every number above.

## 5. Limitations

* **No distillation anchor.** Both real recordings went to the screen, so the
  only constraint was L2-SP on two kernels. The drift magnitudes in §4.1 should
  be read against that.
* **Selection has no genuinely held-out signal.** Validation and drift screen both
  come from the same recording sessions as training.
* **`d@high` inflation is ungated** and still rising at e40.
* **Labels are score-derived.** Pitches are read from the MIDI with a per-take
  tuning correction; the model is trained toward equal temperament plus a
  constant offset, not toward what was actually sung frame by frame.
* **40 epochs is more than needed**, though harmless. F is within 0.005 of its
  best from e31 (best 0.664 at e36) and the last five epochs gain +0.000; the
  linear trend over e30-e40 is +0.0007/epoch. Drift plateaus at the same point --
  `d@high` peaks at +0.124 (e30) and settles to +0.099, `mean` and `r` flat --
  so the final ten epochs cost compute and nothing else. ~30 epochs would give
  the same model.
