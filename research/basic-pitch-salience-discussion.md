# Basic Pitch salience — simplification, speed, and fine-tuning

Conversation transcript, 2026-08-21. Repos referenced: `~/Documents/git/basic-pitch`
and `~/Documents/git/multif0-estimation-polyvocals`.

---

## User

I would only need the salience from this basic pitch model. 1) is it possible to simplify this model? 2) Would this model run quicker than model 3 of ~/Documents/git/multif0-estimation-polyvocals/models/exp3multif0/saved_model.pb? 3) could the training and validation dataset used in the multif0 repository under /finetune/data/ be used to fine-tune the salience of the basic-pitch and make it perform much better on recorded choir audio?

## Assistant

All three answers are yes, with caveats. I measured everything on this machine (8 CPU cores, no GPU — CUDA driver missing, so these are CPU numbers).

### 1. Yes — the salience head is 2 conv layers

I read the shipped checkpoint's variable list directly (`basic_pitch/saved_models/icassp_2022/nmp/variables`). It has exactly 9 weight-bearing layers, and only 4 of them feed the `contour` output:

| layer | shape | params |
|---|---|---|
| BatchNorm on CQT | (1,) ×4 | 4 |
| Conv2D 8 filters, kernel (3,39) | (3,39,8,8) | 7,496 |
| BatchNorm | (8,) ×4 | 32 |
| Conv2D 1 filter (5,5), sigmoid | (5,5,8,1) | 201 |
| **salience total** | | **7,733** |
| note branch (7,7)+(7,3) | | 6,946 |
| onset branch (5,5)+BN+(3,3) | | 2,185 |
| **checkpoint total** | | **16,864** |

So 54% of the weights are dead for your purpose. And there is nothing else to prune: the first contour conv is commented out in [models.py:226-239](basic_pitch/models.py#L226-L239) — it was unintentionally skipped when the released model was trained, so the salience path really is just harmonic-stacked CQT → Conv(3,39) → BN/ReLU → Conv(5,5) → sigmoid.

Measured, batch of five 2 s windows = 10 s of audio:

| | params | 10 s | RTF | 1-window latency |
|---|---|---|---|---|
| full 3-output | 16,782 | 155 ms | 64× | 33 ms |
| contour-only | 7,715 | 112 ms | 90× | 26 ms |
| ↳ of which CQT+harmonic stack | 2 | 35 ms | | |

Dropping note+onset is free — same weights, no retraining, `Model(inputs, model.get_layer('contour').output)`. Going further buys little: `n_harmonics` 8→4 gets you to 92 ms (needs retraining), and cutting the 8 filters to 4 gained **nothing** (111 ms) — the cost is the 39-bin-wide kernel sweeping 264×172×8, not the filter count.

One practical wrinkle: in this venv the shipped `saved_model` won't load (TF 2.21 chokes on the old optimizer slots) and `models.model()` won't build under Keras 3 (`NormalizedLog.build` uses `input_shape.rank`, and `tf.expand_dims` on a KerasTensor). Workable routes: pull the four tensors straight out of the checkpoint with `tf.train.load_checkpoint` (`layer_with_weights-0..3` — the shapes are unambiguous, I verified) into a hand-built Keras-3 model, or prune `nmp.onnx` with `onnx.utils.extract_model`.

### 2. Much quicker — ~64× end-to-end

`models/exp3multif0` is `build_model3` (two branches, HCQT mag + phase-diff, output 360×T), **1,252,697 params**. For 10 s of audio:

- pumpp HCQT + phase differentials: **1,215 ms**
- network: **5,976 ms**
- total **7.2 s → RTF 1.4×**

versus basic-pitch salience-only at **112 ms → RTF 90×**. That's **64× faster end to end**, 53× on the network alone. Even `build_model3_mag` (single branch) would only halve it.

The gap is structural: model3 is 8 conv layers at 360 bins with two (70,3) kernels — 215k MACs per output position, twice over — while basic-pitch is 2 conv layers and gets its CQT in-graph and vectorized.

### 3. Fine-tuning basic-pitch salience on that data — yes, and it's a better-conditioned fine-tune than the model3 one, but it won't fix everything

What lines up unusually well:

- **Time grid is identical.** Both are sr 22050 / hop 256 → 86.13 fps. Your f0 CSVs are already on exactly that grid (0.011610 s).
- **The f0 CSVs are already multi-f0** (time + up to 6 freqs per frame), so they convert straight to contour targets.
- **Data volume is ample for 7.7k params**: Archive/train 75.4 min + Archive/valid 30.6 min synthetic with exact per-frame labels, plus 3.3 min of real blocked-chord takes and 7.7 min of full real recordings. You had 1.25M params to constrain before; now it's 160× fewer.
- **Frequency range is a superset**: 27.5–4353 Hz vs model3's 32.7–2067 Hz. No labels dropped.

What has to be rebuilt:

- **Frequency grid**: 264 bins at 33.33 cents from 27.5 Hz, not 360 at 20 cents from 32.7 Hz. Re-grid from the f0 CSVs — the `(F=360, win, H)` npz caches `prepare_real_chords.py` writes are unusable as-is.
- **Target shape**: basic-pitch's contour targets are *hard, one bin per f0* — mirdata `to_sparse_index` → sparse → dense, no blur anywhere ([tf_example_deserialization.py:459](basic_pitch/data/tf_example_deserialization.py#L459)), with `label_smoothing=0.2` doing the softening in the loss. Your pipeline uses a σ=1-bin Gaussian normalized to peak 1. Match basic-pitch's convention, or its sigmoid is calibrated against a different target mass. If you keep a blur, σ≈0.6 bins preserves your 20 cents.
- **Per-window AGC is unavoidable here.** `NormalizedLog` ([signal.py:154](basic_pitch/layers/signal.py#L154)) min/max-normalizes dB over each 2 s window, at train and inference alike. That's precisely the per-segment AGC that `prepare_real_chords.py`'s docstring says you dodged for model3 — you can't dodge it now, it's baked into the trained front end. So use the same 2 s window length for fine-tuning as for inference.
- **Tuning offsets matter less.** Your measured offsets (manifest `cents`, −71.5…+18.5) are ≤1 bin at 33 cents. Still apply them (−71.5 is 2 bins), but the precision pressure drops.

The two risks I'd flag:

- **MIDI-derived labels are piecewise-constant in pitch.** Training a *contour* output on them teaches equal-temperament snapping and will erode pitch precision on vibrato and portamento. On held blocked chords it's mostly harmless — but keep the 106 min of synthetic (real per-frame f0) as the bulk and use the real takes as the level/timbre-domain anchor, the same role they play in your model3 loop. The distillation anchor, L2-SP and real-audio drift guard in `finetune.py` port over conceptually unchanged, and you're now cheap enough to run `full` rather than `conv`.
- **Fine-tuning cannot add receptive field.** basic-pitch's salience kernel sees 3 frames × 13 semitones — 35 ms of context. model3 sees a very large field via two (70,3) harmonic kernels. If, after fine-tuning, the residual failure mode is octave confusion or merged inner voices rather than "quiet voice under-read", that's architecture, not data — and the fix is to un-comment the skipped conv at [models.py:230-239](basic_pitch/models.py#L230-L239) and/or widen the (3,39) kernel, which means training from scratch on your ~106 min, not fine-tuning. For dense close-voiced choir chords I'd rate that a real possibility.

Suggested order: (1) extract the 4-layer salience model and verify it reproduces the shipped contour output bit-for-bit; (2) write the 264-bin gridder, reusing the by-take split in `manifest.csv`; (3) baseline on the real recordings *before* touching weights so you have the drift reference; (4) fine-tune with your existing guards.

---

## User

What is the difference between the contour output of this model and the salience output of the other model? Does contour means that only one bin is activate and has the value either 0 or 1, and no bins around?

## User

*(interrupting a measurement run)* I meant a qualitive description and explanation. Do not quantify

## Assistant

They're the same *kind* of object — a time × frequency map of independent per-bin sigmoids. "Contour" is just Spotify's name for it, chosen because in their model it's the fine-pitch layer that feeds the note layer; "salience" is the term the multi-f0 literature uses for the identical concept. Neither is a classification over frequency.

**No, the contour output is not one-hot and not binary.** Every bin has its own sigmoid, so a frame is a continuous curve across all bins. There is no softmax over frequency, no argmax, nothing that makes bins compete — which is essential, since the whole point is to represent several simultaneous pitches. When you look at an actual frame you see a peak with shoulders: the neighbouring bins carry intermediate values, partly because the CQT itself smears a partial across adjacent bins, and partly because the last layer is a 2-D convolution that mixes neighbours by construction.

Where your intuition is right is the **training target**, and that's the real difference between the two models. basic-pitch's contour labels are hard: each annotated f0 lights the single nearest bin with a one, everything else is zero, no blur at all. The multi-f0 targets are the opposite convention — a Gaussian ridge deliberately spread across neighbouring bins, normalised so the centre sits at one. So the two models were taught different things about what a pitch *looks like* in the map: basic-pitch was rewarded only for the exact bin and got nothing for marking its neighbours, so it's pushed toward the sharpest ridge the evidence allows; the multi-f0 model was explicitly taught to paint a ridge of a particular width.

That has a knock-on effect worth knowing about. Because basic-pitch trains with label smoothing, it is never asked to output zero or one — the background is pulled toward a small positive value and the peaks toward something well short of one. So the contour map has a raised floor and a compressed ceiling: you won't see a true zero anywhere, and even a strongly-sung note doesn't approach one, especially on real choir audio. Practically this means thresholds are not transferable between the two models, and "confidence" on the contour lives on a squashed scale that you have to calibrate rather than read at face value.

Two further qualitative differences that matter for your use:

**What the number is optimised for.** The multi-f0 salience is the model's final answer — it was trained to be a standalone readout of every voice in a vocal mix. basic-pitch's contour is a precursor: a note branch sits downstream and can clean up after it, so during training the contour was allowed to be under-committed and slightly noisy wherever the note layer could recover. If you take it as your only output, you're using it in a role it wasn't scored on.

**What the values are relative to.** basic-pitch processes fixed-length windows and normalises the input dynamic range within each window, so a contour value is implicitly relative to the loudest thing in its own window; a long file is stitched from those windows. The multi-f0 pipeline runs the front end over the whole file with one reference. Given that your concern is quiet-voice under-reading, that difference in what the level is measured against is probably more consequential than anything about the map's shape.

Also, purely as geometry: the contour covers a wider pitch range at coarser frequency spacing, the salience a narrower range at finer spacing, and the axes are transposed between them.

---

## User

Can the contour output still be used to determine f0s with sub-bin resolution?

## Assistant

Yes, in principle — but with a caveat that's specific to how this model was trained, and it changes the answer from "yes, straightforwardly" to "yes, with calibration."

**Why it's possible at all.** The map is continuous and the ridges have shoulders, so the asymmetry between the two bins flanking the peak carries information about where inside the bin the true frequency sits. That's the usual trick: quadratic interpolation across the peak and its neighbours, or a centre-of-mass over the ridge. Nothing about the output format forbids it.

**Why it isn't clean here.** In a model trained on blurred targets — like the multi-f0 one — the ridge *shape* is the label, so the neighbour ratio is a supervised quantity: the model is explicitly taught to place a ridge whose asymmetry encodes the sub-bin offset. basic-pitch's labels are nearest-bin only. A pitch sitting exactly on a bin centre and one sitting a third of a bin off receive **identical** targets. The supervision actively throws the sub-bin information away. The only reason any of it survives in the output is that the model physically can't fit a one-bin spike from smeared input evidence, so the residual shape leaks the truth.

That means the interpolated offset is a byproduct rather than an estimator, and you should expect it to be biased — specifically, pulled toward bin centres, since the training pressure is to concentrate everything in the labelled bin. You can get real sub-bin precision out of it, but only after measuring that bias against audio of known pitch and fitting a correction. Don't trust the raw interpolation.

Note also that basic-pitch's own pitch-bend estimator does no interpolation at all — it takes the arg-max over the frequency window. So bin resolution is what the upstream code assumes; past that you're on your own.

**Two confounds that bite hardest in your material.** Close-voiced choir chords put voices a few bins apart, so adjacent ridges overlap and the shoulders you'd be interpolating are contaminated by the neighbouring voice — sub-bin readout degrades precisely where the music is hardest. And vibrato smears a ridge within a single frame, so the peak shape reflects the excursion, not an instantaneous pitch.

**Two better routes, depending on whether you're fine-tuning.**

If you're using the model as-is: use the contour to *locate* pitches — which voice is where, how many are sounding — and then measure the actual frequency on the CQT or spectrum at that location, or hand the neighbourhood to a dedicated f0 refinement. The map tells you where to look; the signal tells you the frequency. For intonation work, which is clearly what you're after given the cents columns in your manifest, this is much sounder than reading fine pitch off a coarse posteriorgram.

If you are fine-tuning: this is a good argument for **switching the target convention to a blurred ridge** rather than matching basic-pitch's hard labels, which is the opposite of what I suggested earlier. The trade-off is real — matching the original convention preserves the pretrained output calibration, while switching to blur asks the model to relearn its ridge shape. But relearning a local output shape is about the cheapest thing you can ask of a two-layer head with a small final convolution, so the cost is low and it converts sub-bin position from an artifact into a trained, calibrated property. If fine pitch is the goal, I'd take that trade.

---

## User

Would the second option mean that the model structure needs to be adapted?

## Assistant

No. Switching to blurred targets is a change to the labels and the loss, not to the network.

The output layer is already a per-bin sigmoid across the whole frequency axis, computed independently per bin. That can express a ridge exactly as easily as a spike — there's no normalisation, no competition, nothing that presumes a single active bin. Same input tensor, same output tensor, same activation. And the expressivity is comfortably there: the final convolution spans several bins in the frequency direction, so shaping a ridge a couple of bins wide is a local operation well within its reach, and the wide kernel below it already spans many semitones for harmonic reasoning. You're asking the head to redistribute mass it already produces, not to represent something new.

What does change sits outside the architecture:

- **The label builder.** Blurred ridge instead of nearest-bin, which is where the sub-bin information gets encoded in the first place.
- **Label smoothing.** This is the one that will actually bite if you ignore it. Smoothing is designed for hard targets; applying it on top of an already-soft ridge compresses the ridge toward the middle and blurs the distinction between "shoulder" and "background", which is precisely the signal you're trying to teach. You'd want to reduce it or drop it and let the ridge itself do the softening.
- **Positive/negative balance.** A blurred target puts more positive mass in each frame than a single-bin target, so the effective class balance shifts. basic-pitch trains the contour with plain unweighted cross-entropy, so there's nothing to reconfigure, but it's worth checking the loss doesn't drift toward over-predicting.

Worth separating this from a different change you might be conflating it with. **Blurring the target is free structurally. Making the frequency grid finer is not.** If you decided the bin spacing itself was the limit and wanted more bins per semitone, that changes the CQT bin count, the harmonic-stacking offsets, the output width, and the relationship between the pretrained kernel's frequency extent and a semitone — the pretrained weights stop meaning what they meant, and you'd be training from scratch rather than fine-tuning. So: keep the grid, change the target shape, and you stay in fine-tuning territory with the pretrained front end intact.

One thing that would have been a complication isn't one for you: a blurred contour changes what the downstream note branch expects to receive, since it was trained against spike-shaped contours. Since you're dropping that branch anyway, it's moot.

---

## User

Save this conversation verbatim to an markdown file
