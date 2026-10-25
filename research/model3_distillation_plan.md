# Plan: distilling model3 salience into the basic-pitch student

## Goal

Make the basic-pitch salience (contour) output match that of model3 (`models/exp3multif0`, original unmodified weights), so that drift and congruence of a choir can be determined with the fast basic-pitch front end (about 90x real time against about 1.4x for model3).

Distillation is pure regression onto the teacher's output. The only target is model3's salience.

Out of scope, because they only mattered for the earlier fine-tuning and anchoring design:

- f0 labels and any supervised loss
- bp self-anchor and the alpha blend
- quantile calibration of the teacher
- drift screen vs pretrained (`--drift_high_tol`, `d@high`)
- label smoothing and blurred-ridge target questions
- the model3-vs-bp evenness pre-test

## Inputs

- One model3 `.npz` per recording, from a whole-file run with the original weights.
  - Verified on `alles_model3_exp3multif0_salience.npz`: `salience` (360, T) float16 in 0-0.997, `freq_grid` (360,) 32.7-2068.76 Hz, `time_grid` (T,) at 0.01161 s (86.13 fps, identical to basic-pitch), plus `model_name`, `thresh`, `weights_path`.
- The matching audio, with the same basename (e.g. `alles.wav` and `alles_model3_exp3multif0_salience.npz`).
- At least 1 hour of audio. Several takes of the same song are allowed.

### Splits

- Split by song, never by window or take. All takes of a song go in the same split.
- Hold out at least 3-4 whole songs for validation and a few more as a final test set.
- If any held-out file was in model3's own fine-tuning data, flag it, because that file no longer tests generalisation.

## Design decisions

| Topic | Decision |
|---|---|
| Target | Teacher salience regridded to the bp grid, no calibration |
| Loss | Soft-target BCE (MSE as an alternative), masked to in-band bins |
| Window | 2 s, same as inference (`NormalizedLog` normalises per window) |
| Frequency coverage | Teacher covers bp bins 9-224 only; take the range from `freq_grid`, do not hard-code it |
| Out-of-band bins (0-8 and 225-263) | Masked out of the loss; crop the output to the covered range at inference. Not trained to zero, because the harmonic-stack channels above about 2 kHz are zero-padded and the input changes character |
| Init | Pretrained weights, `--strategy full` |
| Regulariser | L2-SP optional, only if training needs stabilising |

## Steps

### 1. Teacher loader and regrid (`finetune/teacher.py`)

- Load the npz and regrid 360 bins at 20 cents (32.7 Hz) onto the 264-bin bp grid (33.33 cents from 27.5 Hz) by 1-D linear interpolation in cents. The grids coincide at 32.7 Hz (bp bin 9).
- Produce a mask for bins outside the teacher's range.
- Sample along time with `np.interp` at fractional frame positions, because the window hop of 36164 samples (141.27 frames) is not a multiple of 256.
- Window k, output frame j maps to original-audio frame `p = (k * 36164 - 3840) / 256 + j`.

### 2. Alignment self-test (gate before any training)

- Stitch the pretrained-bp output over a file, slice it with the window formula above, and compare against the actual per-window bp outputs.
- They must agree to about 1e-6 in the interior.
- This validates the indexing with a model we already trust, so a wrong offset cannot be mistaken for a distillation failure.

### 3. Cache the dataset

- Precompute (audio window, teacher window, mask) triples once and store them.
- Record the song id with every window so splits stay by song.

### 4. Training mode in `finetune.py`

- Add `--teacher_dir`, which switches to pure distillation with the supervised weight at 0.
- Reuse the existing training loop, caching and epoch logging.
- Replace the pool-building step (`build_distill_pool`, finetune.py:403) with the cached teacher windows.
- Start with `--strategy full` from the pretrained weights.

### 5. Metrics (held-out songs only)

- Masked BCE and MSE against model3.
- Per-frame correlation between student and teacher.
- Peak-pick agreement: F1 of student peaks against teacher peaks at the operating threshold.
- Downstream metric, the decision criterion: run the drift and congruence analysis on student output and on model3 output and compare the numbers, including at cents level.
- Residual vs window loudness, to expose the normalisation mismatch (see Risks).

### 6. Capacity ladder

Model3 sees far more context than the student (3 frames x 13 semitones). Try in order and stop at the first rung that meets the downstream criterion:

1. Fine-tune the existing 7.7k-parameter salience head.
2. Uncomment the skipped first conv ([models.py:230-239](../../basic_pitch/models.py)) and train from scratch. There is no label constraint, so there is no reason not to.
3. Widen the (3,39) kernel or add time context.
4. Optional, only if the 33-cent grid limits the cents-level drift numbers: 5 contour bins per semitone (20 cents), which matches model3's grid exactly and removes the regrid. This changes the output width and harmonic-stack settings, so it means training from scratch.

If the loss plateaus high while the metrics stall, that is capacity, so move to the next rung. More data will not help.

### 7. Deploy

- Export the salience-only model (`Model(inputs, contour.output)`, no note or onset branches).
- Verify it reproduces the trained weights before swapping.
- The shipped `saved_model` does not load in this venv (TF 2.21), so use `tf.train.load_checkpoint` into a hand-built Keras-3 model, or an ONNX extract.

## Risks

- **Capacity.** The student cannot represent the teacher's function exactly. The ladder in step 6 is the response.
- **Normalisation mismatch.** Model3 normalises per segment and bp per 2 s window. The student cannot see the file-level reference, so systematic level differences between loud and quiet passages are unlearnable.
- **Coarser pitch grid.** On the 33-cent grid, the teacher's sub-bin detail is partly averaged away. Judge this on the cents-level drift comparison.
- **Teacher leakage.** Only matters if held-out files were in model3's own fine-tuning set.

## Acceptance

The student is good enough when the drift and congruence numbers computed from its salience on the held-out songs agree with those from model3 within a tolerance you set, for example a cents-level difference well below the drift you need to detect. Loss and correlation are diagnostics. This downstream agreement is the criterion.

## Order of work

1. Steps 1-3: loader, alignment self-test, cached dataset.
2. Step 4: training mode, then rung 1 of the ladder.
3. Step 5: evaluate on held-out songs, then climb the ladder if needed.
4. Step 7: export and verify.
