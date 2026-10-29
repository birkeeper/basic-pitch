# distill02.weights.h5

Basic Pitch's **salience (contour) head**
(`scripts/convert_model_to_savedmodel.build_salience_model`), **distilled** from
model3 (`models/exp3multif0`, original unmodified weights) by regression onto
model3's salience.

Goal: make the fast Basic Pitch front end (about 90x real time, against about 1.4x
for model3) reproduce model3's salience on choir recordings, so drift and
congruence can be measured from the student's output. The plan is
`research/model3_distillation_plan.md`.

This document covers **run01** and **run02** (`finetune/runs/run01`,
`finetune/runs/run02`) and reports only what those two runs measured.

| | |
|---|---|
| Student base | `basic_pitch/saved_models/icassp_2022/nmp` (variables checkpoint) |
| Architecture | Basic Pitch contour path only — note and onset outputs dropped; two convolutions, both BatchNorms |
| Adapted parameters | all 7,715 trainable (`--strategy full`: 2 BatchNorm + `conv2d` + `contours-reduced`) |
| Teacher | model3 salience, one whole-file npz per recording, regridded onto the 264-bin Basic Pitch grid |
| Training data | 3,388 windows from 23 recordings / 23 songs (`data/model3_cache/train`) |
| Validation data | 1,072 windows from 8 recordings / 8 songs (`data/model3_cache/valid`), disjoint by song from training |
| Loss | masked soft-target BCE; nothing else (no L2-SP, no supervised term) |
| Retained run | **run02**, 2026-10-28 22:04, log `run02/distill02_20261028-220422.log` |
| Retained checkpoint | **`run02/distill02.weights.h5`** = e90, val BCE 0.0619 (§3.2) |
| Status | val corr 0.454 → 0.808 and BCE excess over the teacher's entropy floor 0.108 → 0.014. Plateaued. **Downstream acceptance test (drift and congruence vs model3) not yet run** |

---

## 1. Method

Pure regression onto the teacher. The only target is model3's salience; there are
no labels and no supervised or anchor loss.

```
L = Σ_cells  mask · BCE( t , p )  /  Σ_cells mask        t = teacher, p = student
BCE(t, p) = −( t·log p + (1 − t)·log(1 − p) )
```

**Mask.** `time_mask[:, None] & bin_mask[None, :]`. `time_mask` is False where a
window frame falls outside the file. `bin_mask` is True for the 216 Basic Pitch
bins (9–224) that lie inside model3's frequency range; the other 48 bins are not
trained, so the student's output outside the covered range is unconstrained and is
meant to be cropped at inference.

**Teacher alignment.** model3 and Basic Pitch frame at the same rate (86.13
fps) but the window hop (36,164 samples = 141.27 frames) is not a multiple of the
frame hop, so each window is sampled from the whole-file teacher at fractional
frame positions by linear interpolation (`teacher.py`). The checked geometry is
exact. An empirical lag check against the pretrained Basic Pitch found model3's
frames about **+4.25 frames (≈49 ms) later** on four of five songs (the fifth
read +5.75 on a flat correlation curve), and `TEACHER_OFFSET_FRAMES = 4.25` is
applied when the cache is built. That offset aligns the teacher to Basic Pitch's
frames; it has **not** been checked against ground truth (§5).

**Initialisation.** Pretrained Basic Pitch weights; every layer trainable.
run01 starts from the pretrained model, run02 from run01's epoch-16 checkpoint.

## 2. Procedure

Windows are the fixed 172-frame (2 s) windows `inference.get_audio_input` scans
with. The length is not a hyperparameter: `NormalizedLog` rescales by each window's
own dynamic range, so a window of a different length would be normalised
differently from anything seen at inference.

```
run01:  --strategy full --batch_size 10 --lr 1e-4 --epochs 20 --l2sp 0
        --weights <pretrained nmp>
        log ends after 18 epochs (stopped early)

run02:  --strategy full --batch_size 10 --lr 3e-5 --epochs 100 --l2sp 0
        --weights run01/distill01_e16.weights.h5
        ran to completion (100 epochs)
```

339 batches per epoch (3,388 windows at batch size 10): about 6,100 optimizer
steps in run01 and 33,900 in run02. Every epoch is checkpointed. Both runs used
seed 0, so nothing here measures seed variance.

`--lr 3e-5` in run02 is a 3.3× reduction chosen after run01's validation had
flattened with large epoch-to-epoch swings in the output level (§3.3); the
`--weights` was set to epoch 16, the run's best, not its last epoch.

> **Log label.** run02's first line reads `epoch 0 (pretrained)`. That label is
> fixed text in `distill.py`; in run02 it is the **starting weights, run01 e16**,
> not the pretrained model.

## 3. Results

Validation is **held-out songs**: no song, and so no take, appears in both
splits. Every number below is over the masked cells only.

| metric | definition |
|---|---|
| val BCE | the training loss on the validation windows. Its floor is the teacher's own entropy, not zero (see below) |
| BCE − floor | val BCE minus 0.0483, the teacher's mean entropy over the validation cells. This is the part a perfect student would remove |
| val MSE | mean squared error in salience units; floor 0 |
| corr | per-frame Pearson correlation across the 216 in-band bins, averaged over frames. Ignores overall level and scale |
| peak P / R / F | local maxima along frequency above 0.3, student against teacher, matched within ±1 bin |
| resid quiet / loud | mean signed error (student − teacher) averaged per window, then over the quietest and loudest quartile of windows by RMS |

**The BCE floor.** With a soft target the lowest BCE is the teacher's entropy
`−(t·log t + (1−t)·log(1−t))`, at most 0.693 where the teacher reads 0.5. Measured
on the validation cache, with the same mask: **0.0483**. About 78% of the best
BCE below is that floor.

### 3.1 Trajectories

**run01** — lr 1e-4, from the pretrained model. Epoch 0 is the true pretrained
student. Bold marks the retained epoch (e16).

| epoch | train | val BCE | BCE − floor | val MSE | corr | peak P | peak R | peak F | resid quiet | resid loud |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | — | 0.1560 | 0.1077 | 0.01423 | 0.454 | 0.791 | 0.200 | 0.319 | +0.0882 | +0.0818 |
| 1 | 0.0984 | 0.0781 | 0.0298 | 0.00428 | 0.736 | 0.851 | 0.419 | 0.561 | +0.0144 | +0.0087 |
| 2 | 0.0719 | 0.0714 | 0.0231 | 0.00372 | 0.762 | 0.828 | 0.520 | 0.638 | +0.0095 | +0.0069 |
| 3 | 0.0667 | 0.0691 | 0.0208 | 0.00354 | 0.774 | 0.809 | 0.577 | 0.674 | +0.0082 | +0.0075 |
| 4 | 0.0644 | 0.0676 | 0.0193 | 0.00344 | 0.778 | 0.806 | 0.589 | 0.681 | +0.0061 | +0.0062 |
| 5 | 0.0632 | 0.0658 | 0.0175 | 0.00337 | 0.781 | 0.828 | 0.556 | 0.665 | +0.0029 | +0.0013 |
| 6 | 0.0625 | 0.0657 | 0.0174 | 0.00351 | 0.783 | 0.865 | 0.483 | 0.620 | -0.0006 | -0.0040 |
| 7 | 0.0620 | 0.0647 | 0.0164 | 0.00328 | 0.786 | 0.815 | 0.591 | 0.685 | +0.0020 | +0.0010 |
| 8 | 0.0617 | 0.0647 | 0.0164 | 0.00325 | 0.791 | 0.834 | 0.559 | 0.669 | +0.0017 | +0.0009 |
| 9 | 0.0614 | 0.0647 | 0.0164 | 0.00335 | 0.791 | 0.864 | 0.501 | 0.634 | +0.0000 | -0.0022 |
| 10 | 0.0611 | 0.0641 | 0.0158 | 0.00317 | 0.792 | 0.821 | 0.592 | 0.688 | +0.0023 | +0.0015 |
| 11 | 0.0609 | 0.0637 | 0.0154 | 0.00315 | 0.796 | 0.827 | 0.586 | 0.686 | +0.0016 | +0.0007 |
| 12 | 0.0607 | 0.0639 | 0.0156 | 0.00324 | 0.795 | 0.845 | 0.543 | 0.661 | +0.0005 | -0.0012 |
| 13 | 0.0606 | 0.0636 | 0.0153 | 0.00314 | 0.793 | 0.800 | 0.631 | 0.706 | +0.0021 | +0.0017 |
| 14 | 0.0604 | 0.0634 | 0.0151 | 0.00309 | 0.797 | 0.818 | 0.611 | 0.700 | +0.0024 | +0.0018 |
| 15 | 0.0603 | 0.0645 | 0.0162 | 0.00338 | 0.791 | 0.868 | 0.507 | 0.640 | -0.0033 | -0.0069 |
| **16** | 0.0602 | **0.0632** | 0.0149 | 0.00313 | **0.799** | 0.847 | 0.560 | 0.674 | +0.0002 | -0.0011 |
| 17 | 0.0600 | 0.0645 | 0.0162 | 0.00325 | 0.798 | 0.746 | 0.718 | 0.732 | +0.0062 | +0.0090 |
| 18 | 0.0599 | 0.0632 | 0.0149 | 0.00308 | 0.800 | 0.777 | 0.684 | 0.727 | +0.0038 | +0.0048 |

Epoch-by-epoch spread in 5-epoch blocks:

| epochs | val BCE min–max | mean | corr min–max | peak F min–max | resid (loud) min–max |
|---|---|---|---|---|---|
| 1–5 | 0.0658–0.0781 | 0.0704 | 0.736–0.781 | 0.561–0.681 | +0.0013–+0.0087 |
| 6–10 | 0.0641–0.0657 | 0.0648 | 0.783–0.792 | 0.620–0.688 | -0.0040–+0.0015 |
| 11–18 | 0.0632–0.0645 | 0.0638 | 0.791–0.800 | 0.640–0.732 | -0.0069–+0.0090 |

**run02** — lr 3e-5, from run01 e16. Epoch 0 is the starting weights (run01 e16).
Bold marks the retained epoch (e90).

| epoch | train | val BCE | BCE − floor | val MSE | corr | peak P | peak R | peak F | resid quiet | resid loud |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | — | 0.0632 | 0.0149 | 0.00313 | 0.799 | 0.847 | 0.560 | 0.674 | +0.0002 | -0.0011 |
| 1 | 0.0600 | 0.0629 | 0.0146 | 0.00305 | 0.799 | 0.806 | 0.635 | 0.710 | +0.0017 | +0.0014 |
| 2 | 0.0600 | 0.0628 | 0.0145 | 0.00307 | 0.798 | 0.826 | 0.602 | 0.696 | -0.0001 | -0.0014 |
| 3 | 0.0599 | 0.0628 | 0.0145 | 0.00304 | 0.800 | 0.819 | 0.616 | 0.703 | +0.0005 | -0.0003 |
| 5 | 0.0599 | 0.0627 | 0.0144 | 0.00305 | 0.799 | 0.827 | 0.603 | 0.698 | +0.0002 | -0.0010 |
| 10 | 0.0597 | 0.0627 | 0.0144 | 0.00302 | 0.799 | 0.797 | 0.653 | 0.718 | +0.0018 | +0.0018 |
| 15 | 0.0596 | 0.0627 | 0.0144 | 0.00306 | 0.800 | 0.839 | 0.586 | 0.690 | -0.0012 | -0.0031 |
| 20 | 0.0595 | 0.0626 | 0.0143 | 0.00299 | 0.803 | 0.804 | 0.649 | 0.718 | +0.0016 | +0.0015 |
| 30 | 0.0594 | 0.0623 | 0.0140 | 0.00297 | 0.805 | 0.827 | 0.615 | 0.705 | +0.0001 | -0.0011 |
| 40 | 0.0594 | 0.0625 | 0.0142 | 0.00300 | 0.800 | 0.813 | 0.631 | 0.711 | -0.0005 | -0.0018 |
| 50 | 0.0593 | 0.0622 | 0.0139 | 0.00293 | 0.806 | 0.809 | 0.648 | 0.720 | +0.0010 | +0.0004 |
| 60 | 0.0592 | 0.0622 | 0.0139 | 0.00298 | 0.805 | 0.835 | 0.600 | 0.698 | -0.0008 | -0.0024 |
| 70 | 0.0591 | 0.0621 | 0.0138 | 0.00291 | 0.807 | 0.808 | 0.654 | 0.723 | +0.0012 | +0.0008 |
| 80 | 0.0591 | 0.0621 | 0.0138 | 0.00292 | 0.808 | 0.786 | 0.689 | 0.734 | +0.0024 | +0.0028 |
| **90** | 0.0591 | **0.0619** | 0.0136 | 0.00289 | **0.808** | 0.808 | 0.654 | 0.723 | +0.0008 | +0.0003 |
| 100 | 0.0590 | 0.0626 | 0.0143 | 0.00304 | 0.805 | 0.852 | 0.568 | 0.682 | -0.0022 | -0.0048 |

Epoch-by-epoch spread in blocks:

| epochs | val BCE min–max | mean | corr min–max | peak F min–max | resid (loud) min–max |
|---|---|---|---|---|---|
| 1–10 | 0.0626–0.0633 | 0.0628 | 0.796–0.801 | 0.669–0.718 | -0.0044–+0.0018 |
| 11–20 | 0.0625–0.0629 | 0.0626 | 0.799–0.803 | 0.685–0.726 | -0.0038–+0.0030 |
| 21–40 | 0.0623–0.0634 | 0.0626 | 0.799–0.805 | 0.654–0.734 | -0.0061–+0.0047 |
| 41–60 | 0.0621–0.0633 | 0.0624 | 0.801–0.807 | 0.655–0.737 | -0.0063–+0.0065 |
| 61–80 | 0.0620–0.0630 | 0.0623 | 0.803–0.808 | 0.674–0.737 | -0.0055–+0.0048 |
| 81–100 | 0.0619–0.0632 | 0.0623 | 0.802–0.809 | 0.657–0.740 | -0.0063–+0.0064 |

Summary:

| | pretrained | run01 e16 | **run02 e90** |
|---|---|---|---|
| val BCE | 0.1560 | 0.0632 | **0.0619** |
| BCE − floor | 0.1077 | 0.0149 | **0.0136** |
| val MSE | 0.01423 | 0.00313 | **0.00289** |
| corr | 0.454 | 0.799 | **0.808** |
| peak P / R / F @0.3 | 0.791 / 0.200 / 0.319 | 0.847 / 0.560 / 0.674 | 0.808 / 0.654 / 0.723 |

BCE excess falls from 0.1077 to 0.0136, closing about 87% of the distance to a
perfect match. run01 reached 0.0149 (86%) in 16 epochs; the 100 further epochs of
run02 closed about 9% of what was left, and only about 0.0004 of that after
epoch 30.

### 3.2 Selection

`distill.py` writes `--out` whenever validation BCE is **strictly** lower than the
best so far (the pretrained model is the first reference), so a tie never
overwrites.

```
run01:  saved at e1–7, 9–11, 13, 14, 16   best BCE 0.0632 (e16)
        e18 has BCE 0.0632 (to 4 d.p.), the best MSE (0.00308) and the best corr (0.800)
        but was not strictly lower than e16, so it was not saved
run02:  saved at e1–4, 6, 8, 11, 12, 16, 22, 30, 35, 42, 45, 47, 56, 65, 69, 81, 90
        best BCE 0.0619 (e90); the best corr (0.809) is at e96
```

run01's log ends at epoch 18 of 20 with no final summary line, so the run was
stopped before completion. run02's retained epoch has corr 0.808 against 0.809 at the best-correlation
epoch, and e90, e98 and e99 tie at BCE 0.0619; the
selected epoch is arbitrary within noise.

BCE alone was the selection criterion. Peak F is **not** usable for it: see §3.3.

### 3.3 Level jitter and peak F

Peak F swings far more than the loss does. In run02 it ranges 0.654–0.740 over
epochs 21–100 while corr moves 0.799–0.809. The cause is the output level: the
residual (student − teacher) wobbles by about ±0.005 from epoch to epoch, in step
with F, and a shift that small moves the many student peaks sitting just below
the 0.3 cutoff across it (recall 0.51 → 0.72 in run01 between e15 and e17). The
fixed threshold makes F a measure of where the level happened to land, not of
the shape.

* **run01** (lr 1e-4): residual in the loud quartile reached −0.0069 (e15) and
  +0.0090 (e17).
* **run02** (lr 3e-5): the swings are smaller at first (loud residual
  −0.0014 to +0.0014 over e1–5) but return later: −0.0063 to +0.0064 over e81–100.

The lower learning rate therefore did not remove the jitter, so it is not only a
step-size effect. These logs do not establish its cause.

### 3.4 Loudness

The per-window loudness check was intended to expose a normalisation mismatch
(model3 normalises per segment, Basic Pitch per 2 s window, so a systematic
difference between loud and quiet passages is unlearnable). In both runs the
residual is nearly the same in every loudness quartile at every epoch, e.g. run02
e90 resid quiet / loud = +0.0008 / +0.0003. **No loudness-dependent error appears on
the validation songs.** The dominant error is the common level shift of §3.3.

The pretrained student reads about +0.085 above the teacher everywhere (its
label-smoothing floor near 0.10 against model3's near-zero floor). The first epoch
of run01 removed most of it (+0.0087 in the loud quartile).

## 4. Capacity

The plan's ladder (§6) says that if the loss plateaus high while the metrics stall,
that is capacity. These runs reached:

* BCE excess 0.0136, MSE 0.0029 (RMS error about 0.054 in salience units) and corr
  0.81. run01 got to corr 0.799 and MSE 0.00313 by e16; 100 more epochs at a
  lower learning rate added 0.009 in corr and −0.0002 in MSE.
* Train loss is only about 0.003 below validation BCE (run01 e16: 0.0602 against
  0.0632) and validation does not rise. This is not overfitting.
* Correlation was still creeping up (best 0.805 at e30, 0.809 at e96), so the
  ceiling is not sharp, but a 3.3× lower learning rate and 5× more epochs gained
  a small fraction of what the first 16 epochs did.

Consistent with the head's limit: a 7.7k-parameter, two-convolution head against
model3's wider context. These runs do not test the alternatives (§5).

## 5. Limitations

* **Acceptance not tested.** The plan's criterion is that drift and congruence
  computed from the student's salience agree with model3's on held-out songs.
  Loss, correlation and peak F are diagnostics. Nothing here measures the
  criterion.
* **Selection on the validation set.** The retained epoch is the argmax of val BCE
  over 100 noisy epochs, and the same eight songs are used to report it, so the
  figures are slightly optimistic. There is **no separate test split**; the plan
  asks for one.
* **Teacher alignment is relative, not absolute.** The +4.25-frame offset was
  fitted to make the teacher agree with the pretrained Basic Pitch on five songs;
  the optimum read +4.0 to +5.75 frames and the correlation gain at it was only
  0.013–0.025. Which of the two models is correct against the
  audio has **not** been checked (a MIDI-based onset check was proposed and not
  run). If model3 is the late one, the student inherits the same lag.
* **No loudness-dependent error was found, but only 8 songs.** The test is on the
  same ensemble, room and microphone as training.
* **Teacher leakage.** The plan asks for flagging any held-out file that was in
  model3's own fine-tuning data. That has not been checked.
* **Coarser pitch grid.** The 33-cent Basic Pitch grid averages away part of
  model3's 20-cent detail. Judge it on the cents-level drift comparison, which has
  not been run.
* **Single seed, two learning rates.** Nothing separates seed noise from the
  differences between epochs, and run02's change of learning rate is not isolated
  from its extra epochs.
* **Peak F depends on the fixed 0.3 threshold** and is not a reliable selection or
  comparison metric (§3.3).
* **Paths in the logs are stale.** Both logs record `finetune/runs/` output paths
  and run02 records `--weights finetune/runs/distill01_e16.weights.h5`; the files
  now live under `runs/run01/` and `runs/run02/`.
