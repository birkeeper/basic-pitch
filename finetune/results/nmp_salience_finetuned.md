# nmp_salience_finetuned_conv.weights.h5

Fine-tuned variant of Basic Pitch's **salience (contour) head**
(`scripts/convert_model_to_savedmodel.build_salience_model`), adapted by
**convolutional fine-tuning** on real choir recordings of blocked chords.

Goal: reduce the *"quiet voice → low salience"* bias, i.e. raise multi-F0 recall
on a voice sung softly relative to the rest of the ensemble, without altering the
model's behaviour on real recordings.

This document covers **run02** and **run03** (`finetune/results/run02`,
`finetune/results/run03`) and reports only what those two runs measured.

| | |
|---|---|
| Base weights | `basic_pitch/saved_models/icassp_2022/nmp` (variables checkpoint) |
| Architecture | Basic Pitch contour path only — note and onset outputs dropped |
| Adapted parameters | both conv kernels and biases — 7,697 (`conv2d`, `contours-reduced`) |
| Frozen parameters | both `BatchNormalization` layers (γ, β **and** running mean/variance) |
| Training data | 167 prepared windows, `./finetune/data/train` |
| Anchor data | 4 unannotated recordings, 514 windows / 17.0 min, `./finetune/data/distill` |
| Drift screen | 2 unannotated recordings (`heroes_opname`, `My Love_opname`), disjoint from the anchor |
| Retained run | **run03**, 2026-09-13, log `nmp_salience_finetuned_conv_20260913-191733.log` |
| Retained checkpoint | **`run03/nmp_salience_finetuned_conv.weights.h5`** = e40, F 0.696 (§3.2) |
| Status | F 0.487 → 0.696 with real-audio drift held at −1.0%. Zero guard rejections in 40 epochs. Validation shares songs, ensemble and room with training; held-out evaluation exists for run02 only (§4.2) |

> **Note on the filename.** `train()` appends the strategy to `--out`, so the
> `_conv` suffix is generated, not typed. **These models are `--strategy conv`** —
> no BatchNorm parameter was modified.

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
`conv`, L2-SP covers 100% of what moves. Confirmed in both logs:
`Strategy 'conv': 2 trainable layer(s), 7697 parameters`.

### 1.1 The loss

```
L = bkld_masked( y , f(x) , mask )      # supervised, ridge targets, focal-weighted
  + λ · bce( f_teacher(z) , f(z) )      # distillation anchor on real audio
  + 1e-3 · Σ ‖W − W₀‖²                  # L2-SP on 2 kernels
```

**Supervised term.** Binary cross-entropy against a Gaussian ridge target
(25 cents), masked to the frames where the score-derived labels hold — inside a
chord's attack, release or reverb tail no target is defensible, but those frames
are still fed as input because they supply the receptive field of the frames that
are labelled.

No label smoothing. Basic Pitch pretrained with 0.2 against single-bin targets;
these targets are ridges whose shoulders carry the sub-bin pitch, and smoothing
squashes exactly those.

`--pos_weight 3.0` in both runs, upweighting bins where `y_true > 0.5`.

**Focal term (run03 only), `--focal_gamma 1`.** Scales each branch of the
cross-entropy by how wrong the prediction currently is — `(1-p)^g` on the positive
branch, `p^g` on the negative — so gradient concentrates on bins the model gets
wrong rather than being spread evenly over all annotated bins. The term is
renormalised by the mean modulation (stop-gradient'd), which prevents it
collapsing toward zero but does not preserve its absolute magnitude: at e08 the
supervised term reads 0.1651 in run02 and 0.4557 in run03, so focal at g=1 raises
its scale roughly 2.8x. `--distill_lambda` has to be set against that (§2).

**Distillation anchor.** The target is whatever the *pretrained* model predicts
for unannotated real audio, so the term simply says "do not change here". No
annotation needed. Weight-space distance is a poor proxy for output-space
distance, so this constrains what L2-SP cannot, on the input distribution that
actually matters.

`--distill_gamma` weights the anchor by the teacher's own confidence: run02 used
0.5, run03 used 0. These runs do not isolate that variable — run03 changed
`focal_gamma`, `distill_gamma` and `distill_lambda` together — so no conclusion
about `distill_gamma` is available from them.

## 2. Procedure

Windows are fixed at 172 frames (`AUDIO_N_SAMPLES`), one per chord, centred on it
with the real surrounding audio as context. The length is not a hyperparameter:
Basic Pitch's input is fixed, and `NormalizedLog` rescales by each window's own
dynamic range, so a training window of a different length would be normalised
differently from anything seen at inference.

```
run02:  --strategy conv --batch_size 10 --epochs 40 --lr 1e-4 --l2sp 1e-3
        --pos_weight 3 --distill_dir ./finetune/data/distill
        --distill_gamma 0.5              # --distill_lambda defaults to 1.0
        --real_audio heroes_opname.flac "My Love_opname.flac"

run03:  (as above) --pos_weight 3 --focal_gamma 1
        --distill_gamma 0 --distill_lambda 2
```

17 batches per epoch, 680 optimizer steps total, checkpointed every epoch.

**The supervised/anchor balance.** The two terms are summed directly
(`loss = loss_s + distill_lambda * loss_r`), so their printed values are the
balance. run02 sits at 0.1651 vs 1.0 × 0.3632 = 0.3632 at e08, ratio **0.45**.
run03 sits at 0.4557 vs 2 × 0.3620 = 0.7240, ratio **0.63**. λ=2 was chosen to
keep the ratio comparable once focal raised the supervised term's scale.

**Anchor and screen are disjoint.** The anchor set is `Ren Lenny`,
`Sweet child of mine`, `That don't impress me much`, `multicolor`; the drift
screen is `heroes_opname`, `My Love_opname`. Both logs confirm the split in their
`Distilling on 514 real windows` listing.

## 3. Results

Peak threshold is re-tuned per checkpoint (reported as `@thresh`) because an
absolute threshold is not comparable across checkpoints. Selection ranks on **F**,
which is what the sweep optimises.

> `THRESH_GRID` was extended from a 0.44 ceiling to 0.785 between these runs.
> run02's operating points (0.245–0.320) sit well inside the old grid and are
> unaffected; run03 reaches 0.440 at e30, at the old ceiling, so its sweep would
> have been marginally constrained under the old grid.

### 3.1 Trajectories

`val_loss` is **not comparable between the runs** — the focal renormalisation
changes the loss scale (baseline `val_loss` 0.2227 in run02, 0.6742 in run03, on
identical weights and identical data). Recall, precision, F and the drift columns
are comparable; both runs share the baseline **recall 0.527 / precision 0.453 /
F 0.487 @0.170**.

**run02** — anchor, no focal:

| epoch | train | distill | recall | precision | F | @thresh | d@high | mean | r |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.2278 | 0.3448 | 0.574 | 0.543 | 0.558 | 0.245 | +0.091 | −13.1% | 0.87 |
| 2 | 0.2071 | 0.3505 | 0.624 | 0.585 | 0.604 | 0.245 | +0.067 | −17.1% | 0.81 |
| 3 | 0.1943 | 0.3515 | 0.647 | 0.595 | 0.620 | 0.245 | +0.076 | −18.0% | 0.77 |
| 5 | 0.1773 | 0.3585 | 0.634 | 0.634 | 0.634 | 0.290 | +0.115 | −17.6% | 0.70 |
| 8 | 0.1651 | 0.3632 | 0.651 | 0.651 | 0.651 | 0.305 | +0.136 | −17.2% | 0.67 |
| 10 | 0.1612 | 0.3646 | 0.654 | 0.667 | 0.660 | 0.305 | +0.130 | −20.0% | 0.67 |
| 15 | 0.1554 | 0.3642 | 0.660 | 0.675 | 0.667 | 0.305 | +0.132 | −21.7% | 0.67 |
| 20 | 0.1504 | 0.3672 | 0.665 | 0.688 | 0.676 | 0.305 | +0.131 | −23.5% | 0.67 |
| 25 | 0.1500 | 0.3653 | 0.689 | 0.668 | 0.678 | 0.290 | +0.133 | −23.9% | 0.66 |
| 30 | 0.1478 | 0.3658 | 0.690 | 0.662 | 0.675 | 0.305 | +0.154 | −19.7% | 0.66 |
| 35 | 0.1470 | 0.3646 | 0.689 | 0.673 | 0.681 | 0.290 | +0.140 | −22.3% | 0.66 |
| **39** | 0.1466 | 0.3642 | **0.693** | **0.674** | **0.683** | 0.305 | +0.155 | −20.2% | 0.66 |
| 40 | 0.1447 | 0.3648 | 0.694 | 0.667 | 0.680 | 0.275 | +0.134 | −23.1% | 0.65 |

**run03** — anchor + focal g=1, λ=2:

| epoch | train | distill | recall | precision | F | @thresh | d@high | mean | r |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.5671 | 0.3519 | 0.544 | 0.529 | 0.536 | 0.425 | +0.223 | +39.0% | 0.76 |
| 2 | 0.4968 | 0.3541 | 0.606 | 0.524 | 0.562 | 0.350 | +0.194 | +20.4% | 0.77 |
| 3 | 0.4773 | 0.3559 | 0.577 | 0.585 | 0.581 | 0.380 | +0.197 | +15.2% | 0.75 |
| 5 | 0.4672 | 0.3580 | 0.637 | 0.626 | 0.631 | 0.410 | +0.192 | +15.7% | 0.67 |
| 8 | 0.4557 | 0.3620 | 0.684 | 0.614 | 0.647 | 0.410 | +0.206 | +13.9% | 0.62 |
| 10 | 0.4568 | 0.3628 | 0.679 | 0.639 | 0.658 | 0.410 | +0.192 | +9.5% | 0.61 |
| 15 | 0.4531 | 0.3635 | 0.683 | 0.655 | 0.669 | 0.410 | +0.193 | +6.6% | 0.61 |
| 20 | 0.4545 | 0.3641 | 0.686 | 0.674 | 0.680 | 0.410 | +0.185 | +2.9% | 0.61 |
| 25 | 0.4524 | 0.3630 | 0.692 | 0.679 | 0.685 | 0.410 | +0.184 | +2.6% | 0.61 |
| 30 | 0.4535 | 0.3634 | 0.686 | 0.691 | 0.688 | 0.440 | +0.196 | +5.0% | 0.61 |
| 35 | 0.4517 | 0.3634 | 0.695 | 0.694 | 0.695 | 0.410 | +0.173 | −0.1% | 0.61 |
| 39 | 0.4528 | 0.3633 | 0.702 | 0.688 | 0.695 | 0.425 | +0.188 | +2.8% | 0.61 |
| **40** | 0.4492 | 0.3634 | **0.696** | **0.696** | **0.696** | 0.410 | +0.171 | **−1.0%** | 0.61 |

Summary against the shared baseline:

| | baseline | run02 (e39) | run03 (e40) |
|---|---|---|---|
| recall | 0.527 | 0.693 | **0.696** |
| precision | 0.453 | 0.674 | **0.696** |
| F | 0.487 | 0.683 | **0.696** |
| drift `mean` | — | −20.2% | **−1.0%** |
| drift `d@high` | — | +0.155 | +0.171 |
| drift `r` | — | 0.66 | 0.61 |

run03 improves validation F and real-audio drift simultaneously. Because run03
changed three arguments at once (`focal_gamma`, `distill_gamma`,
`distill_lambda`), these runs do not attribute the improvement to any one of them.

### 3.2 Selection

Both runs selected cleanly and **neither had a single guard rejection in 40
epochs** — no `--bal_tol` regression, no `--drift_high_tol` trip.

```
run02:  Best (highest F, guards satisfied) ... from e39, F 0.683 vs baseline 0.487
run03:  Best (highest F, guards satisfied) ... from e40, F 0.696 vs baseline 0.487
```

Both are effectively converged, and the selected epoch is arbitrary within noise:

* run02 is within 0.005 of its best (0.683, first reached at e37) from e32
  onward; e37 and e39 tie exactly.
* run03's F gain per epoch decays 0.0018 (e14→e25) → 0.0010 (e25→e31) → 0.0006
  (e31→e40), against epoch-to-epoch noise of ±0.001. Its last six epochs read
  0.695, 0.695, 0.695, 0.694, 0.695, 0.696 — the apparent late increase is inside
  the noise band, and every epoch from e32 onward is within 0.005 of best.

~30 epochs would give either model; the final ten epochs cost compute and nothing
else.

Picking the argmax of a noisy metric on 54 validation windows makes any of these
figures slightly optimistic as an estimate of true performance. That is a
property of the number, not of these weights.

## 4. Behaviour on real recordings

### 4.1 The drift screen

Two unannotated recordings, compared against the pretrained model's output on the
same audio every checkpoint. Only `d@high` — the change in the pretrained model's
most confident band, its top 0.5% of bins — gates. `mean` and `r` are diagnostics.

**run02** settles at `mean` ≈ −20%, `d@high` ≈ +0.14, `r` = 0.66 from e08. The map
is stable but about 20% compressed relative to pretrained.

**run03** settles at `mean` ≈ 0% (−1.0% at e40, `worst` −3.0%), `d@high` ≈ +0.18,
`r` = 0.61 unchanged for the final twenty epochs. The overall level is neither
compressed nor inflated. The approach is monotone rather than oscillatory —
+39.0% (e1) → +13.9% (e8) → +9.5% (e10) → +2.6% (e25) → −1.0% (e40) — so the map
starts inflated and converges to the pretrained level over the run.

run03's `r` of 0.61 is lower than run02's 0.66, i.e. more change in the map's
shape relative to pretrained, and it is stable for the last twenty epochs. `r` is
a diagnostic; these logs do not establish what the shape change consists of.

The `d@high` gate is one-sided (`d@high < -drift_high_tol`), so inflation is
unopposed. Both runs sit well above zero (+0.155, +0.171) and neither gate fired.

### 4.2 Held-out evaluation (`alles`, measures 1–8)

Against a different piece with a MIDI score, ±80 cents matching, 2984 reference
pitches — material from neither run's training, validation, anchor or screen set.
This is the only out-of-corpus measurement available.

> **Measured for run02 only.** run03 has not been evaluated on `alles` at the time
> of writing. Every run03 figure in this document comes from the in-corpus
> validation split.

These F values are on a different scale from §3 — different music, wider matching
tolerance (80 cents vs `mir_eval`'s default 50), different reference construction.
The pretrained model scores 0.487 in §3 and 0.822 here.

| | baseline | **run02 e39** |
|---|---|---|
| best F | 0.822 @0.20 | 0.812 @0.25 |
| recall ceiling (@0.15) | 0.898 | 0.892 |
| F @0.35 | 0.587 | **0.784** |
| F @0.50 | 0.198 | **0.681** |
| whole-map mean | 0.1052 | 0.0544 |
| r vs baseline | — | 0.727 |
| mean spread | 0.422 | 0.544 |
| mean min/max | 0.578 | 0.456 |

At a matched operating point — each model at the threshold giving the same number
of detections as run02 at 0.50 — run02 reads P=0.955 R=0.530 F=0.681 (1656 peaks)
against the pretrained model at 0.31 with P=0.956 R=0.545 F=0.694 (1700 peaks).

What this shows for run02:

1. **Best-F detection is at parity, over a much wider usable range.** 0.812 vs
   0.822 is within noise, but run02 holds F > 0.75 across thresholds 0.20–0.40
   while the pretrained model peaks sharply at 0.20 and collapses above it
   (F 0.429 at 0.40, 0.198 at 0.50). Threshold choice stops being critical.
2. **Per-voice evenness is worse than pretrained**: spread 0.544 vs 0.422,
   min/max 0.456 vs 0.578, with the quiet voice better held by the pretrained
   model in 7 of 11 chords.
3. **The evenness gap is driven by the loud voices rising, not the quiet voice
   falling.** Soprano B4, absolute salience:

   | chord | baseline | run02 e39 |
   |---|---|---|
   | 7 | 0.242 | 0.200 |
   | 8 | 0.289 | 0.281 |
   | 9 | 0.208 | 0.178 |
   | 10 | 0.228 | 0.250 |
   | 11 | 0.229 | 0.196 |

   The soprano is at roughly baseline level, while Tenor B3 in the same chords
   goes 0.466 → 0.843, 0.421 → 0.828, 0.420 → 0.807, 0.369 → 0.648. In chords 1–6,
   where the soprano is itself the prominent voice, run02 lifts it above baseline
   (0.410–0.663 vs 0.321–0.456).

At run02's operating threshold of 0.25 the soprano sits at 0.178–0.200 in chords
7, 9 and 11 while the tenor is at 0.81–0.84. On this excerpt run02 still ranks
voices by prominence.

### 4.3 What this does and does not establish

Established: on held-out windows from the same corpus, both models detect
substantially more voices at higher precision than pretrained; the pretrained
model's confident detections were not compressed on two unannotated recordings
disjoint from training; and for run02, out-of-corpus detection on `alles` is at
parity with the pretrained model with a wider usable threshold range, while
per-voice evenness on that excerpt is worse.

Not established: anything about a different choir, room or microphone. The
validation takes are drawn from the same four songs, ensemble and sessions as the
167 training windows. A model recalibrating to *this* ensemble in *this* room
would score clean on every §3 number.

## 5. Limitations

* **run03 has no out-of-corpus evaluation.** §4.2 covers run02 only. run03 is
  retained on the strength of its in-corpus validation and its drift screen.
* **Validation cannot see per-voice evenness.** It is a take-level split — every
  song appears in both training and validation — and its metric is pooled
  recall/precision/F with no evenness term. run02 scored F 0.683 there while
  measuring worse than pretrained on spread and min/max on `alles` (§4.2). The
  spread / min-max statistics are computed offline by `compare_voice_salience.py`
  and are not part of checkpoint selection.
* **run03's improvement is not attributed.** Three arguments changed at once.
* **Two distinct pieces in training.** `late` and `late_nolowbass` are the same
  piece; with `parijs` that is two, plus `sing`. Generalisation to new repertoire
  is not measurable from this split.
* **`d@high` inflation is ungated** in both runs (+0.155, +0.171).
* **Labels are score-derived.** Pitches are read from the MIDI with a per-take
  tuning correction; the model is trained toward equal temperament plus a
  constant offset, not toward what was actually sung frame by frame.
* **40 epochs is more than needed**, though harmless — see §3.2.
