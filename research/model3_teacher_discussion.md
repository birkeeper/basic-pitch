# Discussion: using model3's salience as a distillation teacher

## I was thinking to use model3's salience output as a teacher. Still silent voices would get lower salience, but the salience of basic-pitch would more approach the better performing model3. How would you approach that?

Good idea, and the plumbing is more favourable than I expected. Here's how I'd approach it.

### The key structural fact: the change is confined to one function

Your loss already has the right shape:

```
L = supervised + λ · bce(f_teacher(z), f(z)) + L2-SP
```

Today `f_teacher` is the pretrained Basic Pitch itself, computed once in `build_distill_pool()` (finetune/finetune.py:403). Swapping in model3 changes **what array goes into `teacher`**, and nothing else — `make_distill_loss()` (finetune/finetune.py:347), the training loop at finetune/finetune.py:772, `--distill_lambda`, `--distill_gamma` all stay as they are. That's the whole appeal: it's a small, well-isolated diff, and it keeps your run02/run03 baselines directly comparable.

It also flips the anchor's role. Right now the 514 anchor windows are a *regulariser* ("do not change here"). With a model3 teacher they become a **second training signal that needs no annotation** — so the anchor corpus can grow from 17 minutes to hours of your choir recordings at zero labelling cost. That, not the teacher swap itself, is probably where most of the gain lives.

### The three real problems, in order of difficulty

#### 1. Grid alignment (easy — the grids are luckier than they look)

- **Time**: model3 is sr 22050 / hop 256 (utils.py:59-65), Basic Pitch is 22050 / `FFT_HOP` 256. **Identical frame rate**, 86.13 fps.
- **Frequency**: bp bin 9 = 32.7032 Hz, model3 bin 0 = 32.7 Hz; bp bin 225 = 2093.00 Hz, model3 bin 359 = 2092.8 Hz. The grids are *exactly* coincident at the endpoints, because model3's fmin is A0+3 semitones. The mapping is `m = 5b/3 − 15`, exact for every third bp bin. So model3 covers bp bins **9–225 of 264** (82%), and the regrid is a 1-D linear interpolation in cents. Bins 0–8 and 226–263 have no teacher — keep the *existing* bp self-anchor there rather than forcing zeros, or you'll suppress legitimate soprano/high-harmonic activity above 2093 Hz.

The one place to be careful is per-window slicing. model3's HCQT uses pumpp with `log=True` (`ref=np.max` normalisation, effectively per-segment — their own `compute_pump_features_segmented` already runs 10 s segments), so run the teacher **whole-file exactly as you evaluate it**, then slice. Window *k*, output frame *j*, maps to original-audio frame

```
p = (k · 36164 − 3840) / 256 + j        # hop_size=AUDIO_N_SAMPLES−overlap_len, front pad=overlap_len/2
```

`36164/256 = 141.27` is not an integer, so there's a sub-frame offset per window — sample the teacher with `np.interp` along time at fractional `p` rather than rounding.

**Verify this indexing before model3 goes anywhere near it**: build the per-window slices from a *stitched pretrained-bp* map using the same formula, and compare against the actual per-window bp outputs. They must agree to ~1e-6 in the interior. That validates the whole scheme with a model you already trust.

#### 2. Calibration (the one that will bite you)

model3's salience distribution is not Basic Pitch's. Plain BCE toward it will recalibrate the student globally — which is exactly the drift your §4.1 screen measures, and it will move all your tuned thresholds. run03's headline result is `mean` drift −1.0%; a raw model3 teacher throws that away.

Fix: **quantile-map the teacher onto the pretrained student's own marginal** before using it. Pool bp and regridded-model3 values over the anchor corpus restricted to bins 9–225, fit a monotone 1000-quantile map, apply. What then transfers is purely *which bins are high* — the ordering information you actually want — with the level, and therefore your operating points and your drift metrics, preserved by construction. (A single fitted temperature on logits is the cheap fallback, but it's much weaker here.)

#### 3. Conflicting objectives

The self-anchor says "don't move", model3 says "move toward me". Don't run both as separate terms with separate λs — you'll be tuning two weights against each other. Blend the *target* instead, one new knob:

```
t = t_bp                                 bins < 9 or > 225
t = (1−α)·t_bp + α·t̃_model3              bins 9–225
```

α=0 reproduces run03 **exactly**, which gives you the clean single-variable ablation your §5 says run03 lacks. Sweep α ∈ {0, 0.3, 0.5, 0.7, 1.0}.

Worth a second experiment: `t = max(t_bp, t̃_model3)` in-band — only ever raises the target, never asks bp to suppress something it found. It's the pure quiet-voice direction, but it inherits model3's false positives and will push `d@high` up further, and that gate is already one-sided/ungated in both your runs.

Note `--distill_gamma` changes meaning helpfully here. Weighting by teacher confidence used to pin exactly the quiet voices you wanted to free (your §1.1 note); with model3 as teacher, "where the teacher is confident" now *includes* the quiet voice, so `gamma > 0` stops being self-defeating.

### What to check before writing any code

The plan rests on a premise your §4.2 doesn't establish: that **model3 is actually more even per-voice**, not just a better detector. Run `compare_voice_salience.py` on `alles` with model3 vs pretrained bp and look at `spread` and `min/max`. If model3's advantage is detection rather than evenness, distillation won't teach what you're after.

Two other things to decide:

- **Which model3 weights** — the AdaBN-finetuned one is presumably the better teacher, but if any of its finetuning audio overlaps your anchor pool, the teacher is effectively leaking labels there. Keep `heroes_opname` / `My Love_opname` out of the teacher pool regardless, or the drift screen stops being held out.
- **Selection metric.** Your §5 already flags that validation can't see evenness. With a model3 teacher this becomes critical: in-corpus F may barely move while evenness improves a lot. I'd lift the `spread` / `min/max` computation out of `compare_voice_salience.py` into the epoch loop on a small annotated held-out set, and add "correlation with model3 on the screen files" as a diagnostic (it should *rise* — the existing `r` vs pretrained will now fall by design, so it stops being interpretable as drift).

### The ceiling

Per basic-pitch-salience-discussion.md:76: bp's salience kernel sees 3 frames × 13 semitones; model3 sees two (70,3) harmonic kernels. The student **cannot** represent the teacher's function. Expect partial transfer. The diagnostic: if the distillation loss plateaus high while F and evenness stall, that's capacity, not data — and the fix is the architectural one (models.py:230-239), i.e. training from scratch, not fine-tuning.

### Concrete first step

Cache teacher maps to npz (model3 runs ~RTF 1.4, so 17 min of audio ≈ 12 min — worth caching once you grow the pool, unlike the current in-RAM bp teacher). `predict_on_audio.py --save_salience` already writes `salience` / `freq_grid` / `time_grid` in exactly the format `load_salience()` (finetune/compare_voice_salience.py:203) reads, so model3 never has to be importable from the training process — no TF version conflict.

New flags: `--teacher_dir`, `--teacher_alpha`, `--teacher_mode {blend,max}`, `--teacher_calib {quantile,temp,none}`.

Want me to implement it — the regrid + alignment self-test first, so you can confirm the indexing is right before anything trains?

---

## If the driftscreen necessary? I want to move the student's toward the teacher's output

Short answer: the *gate* isn't, but the *files* are. Don't delete it — repoint it.

### Why the gate's premise breaks

`d@high` gates on "the pretrained model's top 0.5% of bins must not drop". That was the right guard when the anchor target was literally the pretrained output. Once you're deliberately moving toward model3, drift versus pretrained is the intended effect, and a guard against the intended effect is just a brake.

But there's a cheap test that tells you whether it's actually in conflict, and you should run it before deciding:

**Compute `d@high` of model3 against pretrained bp on `heroes_opname` and `My Love_opname`.**

bp's top 0.5% of bins are the *loud* voices — the easy case both models get right. If model3 itself passes the gate (plausible), the gate never fires on a student moving toward model3 either, it costs nothing, and it still catches a genuinely broken run. If model3 itself fails it, the gate would reject the teacher, and it has to go. One teacher pass on two files settles it.

### Why you need the held-out files more than before, not less

Today the anchor pool is a regulariser with a target you already have for free. Under your plan it becomes **training data** — 514 windows (soon many more) of supervised-by-model3 signal. The screen files are the only audio in neither training, nor validation, nor the anchor. Your §4.3 already concedes validation shares songs, ensemble and room with training; if the anchor also becomes training, the screen is the *only* thing left that generalisation could be measured on.

So keep the two files and change what you measure on them:

| now | instead |
|---|---|
| `d@high` vs pretrained (gate) | keep only if model3 passes it — as a broken-run detector, not a conservatism gate |
| `mean` vs pretrained | **mean and `r` vs the calibrated teacher, on the screen files** — student-teacher agreement on audio it never distilled on |
| `r` vs pretrained | diagnostic only; it will fall by design, don't read it as drift |

That second row is the valuable one: it's the only number that tells you whether the distillation transfers or whether the student is memorising the anchor corpus. It's free — you're already running model3 offline to build the teacher, just run it on two more files.

### One thing that does still need guarding

Bins outside 9–225 (below 32.7 Hz, above 2093 Hz) have no teacher. If you keep the bp self-anchor there as I suggested, the loss handles it and the screen is redundant for that. If you drop the self-anchor entirely and distill only in-band, those bands are unconstrained and free to drift anywhere — and nothing in your validation metric looks up there. In that case keep a band-restricted drift check on >2093 Hz specifically.

### What the guard should become

If you want a stop condition at all, put it on what you now care about rather than on similarity to pretrained: per-voice `spread` / `min max` on a small annotated held-out set, computed per epoch. Your §5 already lists "validation cannot see per-voice evenness" as a limitation, and run02 is the proof — F 0.683 on validation while evenness went *backwards* on `alles`. With a model3 teacher that failure mode gets more likely, not less, because in-corpus F may barely move while the thing you're actually buying is evenness.

Net: drop `--drift_high_tol` as a gate if model3 fails the pre-test, keep `--real_audio` as a measurement surface, and move the guard onto evenness.

---

## Can't I set below 32.7 and above 2093 Hz to zero? It is not needed in model3 and well above the harmonics of sopranos

Let me check whether the salience head can even represent a band-dependent output.

[Checked basic_pitch/nn.py HarmonicStacking and models.py — computed which harmonic channels are zero-padded at each bin.]

```
MAX_N_SEMITONES 103   n_semitones 103   CQT bins 309
harmonics [0.5, 1, 2, 3, 4, 5, 6, 7]
shifts    [-36, 0, 36, 57, 72, 84, 93, 101]
  bin 210 (1568.0 Hz) zero-padded channels: [7]
  bin 225 (2093.0 Hz) zero-padded channels: [5, 6, 7]
  bin 226 (2133.7 Hz) zero-padded channels: [5, 6, 7]
  bin 240 (2793.8 Hz) zero-padded channels: [4, 5, 6, 7]
```

Now I can answer this precisely — I'd checked the wrong thing before.

### The bottom band is harmless; the top band is not

**Below 32.7 Hz** (bins 0–8): zero them, it costs nothing. That's an octave below a bass low C2 (65 Hz), and those bins sit inside the region 0–35 where `HarmonicStacking` zero-pads the 0.5-harmonic channel, so the model has a genuine positional cue there. No risk, but also nearly no gain — 9 bins.

**Above 2093 Hz** (bins 226–263) is the problem, and it's architectural. The contour head is `Conv2D(..., padding="same")` over (time, freq) with no positional encoding (models.py:241) — one shared kernel applied at every frequency. The *only* thing that distinguishes one bin from another is which harmonic channels `HarmonicStacking` has zero-padded. I computed those:

```
bin 210 (1568 Hz, G6)   zero-padded channels: [7]
bin 225 (2093 Hz, C7)   zero-padded channels: [5, 6, 7]
bin 226 (2134 Hz)       zero-padded channels: [5, 6, 7]   ← identical to 225
bin 240 (2794 Hz)       zero-padded channels: [4, 5, 6, 7]
```

Bins 225 and 226 are **architecturally indistinguishable**. Your teacher would say "bin 225: whatever model3 reports, possibly a ridge" and "bin 226: zero" — with the same input signature and the same shared weights. The boundary is not representable.

The model can only satisfy it by finding a *content*-based rule, and the nearest available one is "suppress activity that looks like a high f0 with dead upper-harmonic channels" — which is precisely the signature of bins 210–225, i.e. **real soprano f0s from G6 to C7**. You'd be spending gradient on a constraint that can only be met by damaging the top of the soprano range. The soprano is the voice your `alles` evaluation (§4.2) already shows sitting at 0.18–0.20 while the tenor is at 0.84. That's the wrong thing to put at risk.

Note also that your framing — "well above the harmonics of sopranos" — is the wrong test, though your conclusion is right and for a stronger reason. Salience is a *pitch* representation; harmonics shouldn't appear there at all. The relevant quantity is soprano **f0**, which tops at C6–G6. So 2093 Hz genuinely carries no wanted signal. The band is empty; it's the *learnability of the boundary* that fails, not the premise.

### You don't need the model to do it

If you never want output above 2093 Hz, slice it off **after** the model, at inference. Identical result, zero gradient spent, zero risk to the soprano, and reversible. 38 bins.

### The version of your idea that is actually interesting

There's a real hypothesis hiding in this, but it's a different one: whatever activity *is* in bins 226+ under the pretrained model is octave-error / harmonic-leakage response to real f0s below. Training it toward zero is then not "cleaning up an unused band" — it's **proxy supervision against harmonic leakage**, and because the kernel is shared, that suppression would generalise to leakage everywhere, improving in-band precision. That's a coherent and attractive argument.

It cuts both ways for exactly the same reason, of course — the shared kernel can't separate "leak at 2500 Hz" from "quiet real f0 at 800 Hz that looks similar". But it's worth testing rather than dismissing.

**The cheap prerequisite either way**: measure what's actually up there. Run the pretrained salience model on `heroes_opname` and `parijs_take10` and report mean / max / fraction-above-0.3 for bins 0–8, 9–225, 226–263. If the top band is already ~0, zeroing teaches nothing and the whole question collapses into "crop at inference". If it's substantially non-zero, the leakage hypothesis has something to work with and is worth an α-sweep of its own.

That's the measurement I was starting when you stopped me — want me to run it? It's two files through the salience head, a minute or two.

---

## yes

[Ran the pretrained salience model over heroes_opname.flac, parijs_take10.wav, "My Love_opname.flac", measuring bands 0-8, 9-210, 210-226, 226-264.]

Measured on three files (40,600 frames total), pretrained salience head:

| band | mean | max | frac>0.15 | frac>0.30 |
|---|---|---|---|---|
| <32.7 Hz (bins 0–8) | 0.100 | 0.159 | 0.0005 | **0** |
| in-band (9–210) | 0.108 | 0.805 | 0.054 | 0.011 |
| 1568–2093 Hz (210–225) | 0.102 | 0.297 | 0.003 | **0** |
| >2093 Hz (226–263) | 0.101 | 0.232 | 0.0006 | **0** |

Per-bin means across 200→263 are flat to three decimals: `0.1021, 0.1021, 0.1020, 0.1021, 0.1018, 0.1013, 0.1017, 0.1008, ... 0.0989, 0.1001`.

### The answer, and it's not the one I expected

**There is nothing up there to suppress.** Across 40,600 frames, the >2093 Hz band never once exceeds 0.30, and its mean sits at 0.101 — indistinguishable from the in-band background. The harmonic-leakage hypothesis from my last message is dead: there's no leakage. (Same for <32.7 Hz.)

But the interesting number is the one that kills the proposal. The floor is **0.101, not 0**. That's the model's background level *everywhere* — note in-band mean is 0.108, barely above it, and your §4.2 "whole-map mean 0.1052" for the baseline matches exactly. So padding the teacher with zeros doesn't teach "nothing lives above 2093 Hz" — the model already behaves as if it knows that. It teaches **"lower your floor from 0.101 to 0"**, on 47 of 264 bins (18% of the loss mass), every bin pushing the same direction, through a kernel shared with the entire frequency axis.

That is a global recalibration dressed up as a band constraint. It is the same whole-map compression that gave run02 `mean` −20% and that run03 spent its λ and focal tuning to eliminate (−1.0%). You'd be re-introducing it through the back door, and your drift screen — if you'd dropped the gate, as we discussed — wouldn't be there to catch it.

BCE at p=0.101 against t=0 is 0.106 per bin, roughly the same magnitude as the in-band supervised term. It isn't a rounding error.

### What to do instead

Keep the **bp self-anchor** outside 9–225, i.e. blend with α=0 out of band. Concretely `t = t_bp` there, which says "stay at your 0.101 floor", costs nothing, and — the reason to prefer it over simply masking those bins out of the loss — it also guards against the student developing *new* leakage up there once a model3 teacher starts reshaping the shared kernel. Masking would leave that unobserved and unconstrained.

Then crop above 2093 Hz at inference if you want a clean output. Free, reversible, no gradient.

If you separately decide the 0.101 floor is too high — and there's an argument for that, it compresses your usable threshold range at the bottom — treat it as its own deliberate change applied uniformly, not something smuggled in through 18% of the bins.

---

## Should I not train those regions to meet the background of model 3?

Let me check whether a model3 salience map already exists so I can give you its actual background level.

[Checked for existing model3 npz files — none on disk.]

None on disk, so I can't give you model3's actual floor — that's the first number to get. But the principle settles the design either way.

### The constraint that matters

The model cannot represent a discontinuity at bin 225/226 — they have identical zero-padded harmonic channels and share a kernel. So whatever you do, **the target must be continuous across that boundary.** That single rule sorts the options:

- **target 0 out-of-band** → a 0.101 step at the boundary. Maximally violates it. This is why it fails.
- **target = model3's background out-of-band, model3's calibration in-band** → continuous. Coherent.
- **target = bp's own floor out-of-band, model3 quantile-mapped onto bp's marginal in-band** → continuous. Also coherent.

Your instinct is right: matching model3's background is strictly better than zero. But which of the last two you want depends on a choice you haven't made yet.

### With calibration, the question dissolves

If you quantile-map model3 onto the pretrained bp marginal (what I recommended for the calibration problem), then model3's background maps to bp's background — ≈0.101 — *by construction*. So "train out-of-band to model3's background" and "keep the bp self-anchor out-of-band" become **the same target**. There's no special out-of-band rule to write; continuity falls out for free, and you can't get it wrong.

That's an argument for quantile calibration I hadn't made: it isn't only about preserving your thresholds and drift metrics, it also makes the band-edge question disappear.

### Without calibration, your version is the only correct one

If you feed model3's raw output as the teacher, you're adopting its calibration wholesale, and then yes — out-of-band must be model3's background, not 0 and not 0.101, or you get the discontinuity again in smaller form.

Be clear-eyed about what that costs, though. Adopting model3's calibration is a global recalibration of the whole map: every threshold in your report moves, `mean` in the drift screen stops being interpretable, and the §4.2 "F > 0.75 across 0.20–0.40" analysis — which is currently your strongest out-of-corpus result for run02 — has to be redone from scratch. And it's orthogonal to what you actually want, which is the *relative* lift of the quiet voice. Calibration is the part of model3 you have no reason to import.

### So

Go with the calibrated teacher and let the out-of-band target be whatever the calibration maps model3's background to. It'll land on ~0.101 and the band edge takes care of itself.

Worth measuring regardless, once you have a model3 map: its background level, and its activity above 2093 Hz and below 32.7 Hz. If model3's floor is far from 0.101 — likely, given its default threshold is 0.4 against your 0.2–0.3 — that's the size of the recalibration you'd be taking on in the uncalibrated design, and it makes the case concrete rather than theoretical.

---

## You can find a model3 salience map here: ~/Downloads/alles_1_8_model3_exp3multif0_salience.npz. The latest finetuned nmp model results in ~/Downloads/alles_1_8_nmp_salience_finetuned_conv_e40_salience.npz

[Loaded both npz files.]

```
== alles_1_8_model3_exp3multif0_salience.npz
   salience (360, 1245) float16   freq_grid (360,) 32.70..2068.76 Hz
   time_grid (1245,) 0.0000..14.443 s  dt=0.011610
   model_name = model3   thresh = 0.5

== alles_1_8_nmp_salience_finetuned_conv_e40_salience.npz
   salience (264, 1251) float16   freq_grid (264,) 27.50..4350.35 Hz
   time_grid (1251,) 0.0000..14.512 s  dt=0.011610
   model_name = nmp_salience_finetuned_conv_e40   thresh = 0.41
```

[Computed distributions, band structure, regrid, correlation, quantile mapping.]

```
model3  mean 0.0152  median 0.0000  max 0.991
   pct 1:0.0000  5:0.0000  25:0.0000  50:0.0000  75:0.0001  90:0.0014  95:0.0276  99:0.5415  99.9:0.8965
   frac>0.1  0.03489   frac>0.2  0.02589   frac>0.3  0.01993   frac>0.4  0.01548   frac>0.5  0.01166

bp-e40  mean 0.0853  median 0.0583  max 0.938
   pct 1:0.0234  5:0.0296  25:0.0405  50:0.0583  75:0.0931  90:0.1523  95:0.2188  99:0.5601  99.9:0.7977
   frac>0.1  0.22100   frac>0.2  0.05903   frac>0.3  0.03067   frac>0.4  0.01980   frac>0.5  0.01317

model3 cents 299.83..7479.83 step 20.000
bp     cents 0.00..8766.67 step 33.333
bp bins covered by model3: 9..224  (32.7..2053.1 Hz)  = 216 of 264

bp-e40 band means / max / frac>0.3:
   below model3  bins   0-  8  mean 0.0924  max 0.356  frac>0.3 0.00107
   in-band       bins   9-224  mean 0.0931  max 0.938  frac>0.3 0.03744
   above model3  bins 225-263  mean 0.0404  max 0.133  frac>0.3 0.00000

model3 band means (its own top/bottom rows):
   bottom 20     mean 0.0000  max 0.001  frac>0.3 0.00000
   middle        mean 0.0171  max 0.991  frac>0.3 0.02242
   top 20        mean 0.0000  max 0.000  frac>0.3 0.00000
```

```
bp-e40 per-bin mean around the model3 top edge (bin 224 = 2053 Hz):
   bin 200  1293.4 Hz  0.0420  ...  bin 224  2053.1 Hz  0.0409
   bin 226  2133.7 Hz  0.0457  ...  bin 262  4267.4 Hz  0.0438
```

```
bp-e40 MEDIAN (the floor) per band:
   <32.7Hz     bins   0-  8  median 0.0844  mean 0.0924
   in-band     bins   9-224  median 0.0643  mean 0.0931
   >2053Hz     bins 225-263  median 0.0386  mean 0.0404
   1293-2053   bins 200-224  median 0.0393  mean 0.0409

regridded model3 (216, 1245)  vs bp in-band (216, 1245)
   corr (raw)            0.7396
   corr vs frame shift:  -4:0.7544  -3:0.7551  -2:0.7528  -1:0.7475  +0:0.7396  +1:0.7302  +2:0.7199  +3:0.7096  +4:0.6999

quantile map model3 -> bp marginal:
   model3 0.000  ->  bp 0.0095      model3 0.500  ->  bp 0.5796
   model3 0.001  ->  bp 0.1581      model3 0.700  ->  bp 0.7085
   model3 0.010  ->  bp 0.2114      model3 0.900  ->  bp 0.9175
   model3 0.100  ->  bp 0.3118      model3 0.990  ->  bp 0.9380
   model3 0.300  ->  bp 0.4448
   corr after mapping    0.7960   (mean 0.0931 vs bp 0.0932)
   out-of-band target would be qmap(0) = 0.0095 ; bp floor above 2053Hz = 0.0386
```

```
model3: frac exactly 0: 0.0097   frac <1e-4: 0.7391

aligned corr (raw, at -3 frame shift)        0.7551

floor-affine calibration  t = floor + (1-floor)*m3^g   with floor=0.0386
   g=0.5  corr 0.8139  mean 0.0713 (bp 0.0932)  p95 0.207 (bp 0.248)
   g=0.7  corr 0.7920  mean 0.0599 (bp 0.0932)  p95 0.123 (bp 0.248)
   g=1.0  corr 0.7551  mean 0.0531 (bp 0.0932)  p95 0.068 (bp 0.248)
   g=1.5  corr 0.6978  mean 0.0481 (bp 0.0932)  p95 0.044 (bp 0.248)
   g=2.0  corr 0.6486  mean 0.0454 (bp 0.0932)  p95 0.039 (bp 0.248)

blend t=(1-a)*bp + a*teacher(g=1): mean drift vs bp-e40
   a=0.0  mean 0.0932 (+0.0%)    a=0.5  mean 0.0732 (-21.5%)
   a=0.3  mean 0.0812 (-12.9%)   a=1.0  mean 0.0531 (-43.0%)
```

Both maps analysed. Three of my earlier claims need correcting, and your instinct turns out to be right — with a caveat that matters more than the band edges.

### Your question, answered: yes, and it coincides with bp's own floor

model3's background is essentially exact zero — **73.9% of its bins are below 1e-4**, median 0.0000, 75th percentile 0.0001. bp-e40's floor, measured where nothing is singing, is **0.039**.

So "train the out-of-band region to model3's background" and "hold it at bp's floor" are the same instruction *provided you calibrate model3's zero onto bp's floor* — which you have to do anyway for the in-band region. Build the calibration as

```
t = floor + (1 − floor) · m3^γ        floor = 0.039
```

and model3's 0 maps to 0.039 by construction, in-band and out-of-band alike. No special rule at the boundary, no discontinuity, nothing to get wrong. Your version is the right one; it just gets expressed in bp's units.

### Correction 1: the boundary really is smooth, but my floor number was wrong

I quoted 0.101 — that was the *pretrained* model on your other files. On e40 over `alles`, per-bin means across the edge are flat from 1293 Hz to 4350 Hz, no step at the model3 edge. The continuity argument stands. (The "in-band 0.093 vs above-band 0.040" band means I might have quoted are ridges vs no ridges, not floor vs floor — the 1293–2053 Hz sub-band, in-band but above the voices, has median 0.039, identical to above-band.)

### Correction 2: quantile mapping is wrong here — retract that

I recommended it. The data says don't. With 74% of model3's mass tied at ≈0, the quantile map is a near-step function:

```
model3 0.000 → bp 0.0095      model3 0.500 → bp 0.580
model3 0.001 → bp 0.158       model3 0.010 → bp 0.211
```

It crushes the floor to a quarter of bp's, then catapults anything barely non-zero to four times the floor. And it buys less than the simple map: aligned correlation with bp goes 0.755 → **0.796** for quantile, versus **0.814** for the floor-affine map at γ=0.5. Use the one-parameter version.

### Correction 3: the level compression is the real risk, not the band edges

This is the number I'd have missed. model3 is sparse in the *mid*-range too, not just at zero. With the floor-affine map at γ=1:

| α | teacher-blend mean | vs bp-e40 | frac>0.3 |
|---|---|---|---|
| 0.0 | 0.0932 | — | 0.0377 |
| 0.3 | 0.0812 | −12.9% | 0.0306 |
| 0.5 | 0.0732 | **−21.5%** | 0.0274 |
| 1.0 | 0.0531 | −43.0% | 0.0218 |

A blend at α=0.5 imposes a −21.5% whole-map compression — essentially exactly the −20% that run02 suffered and that run03's tuning eliminated. You'd undo your best result via the teacher's calibration rather than via λ.

So **fit γ and floor by matching bp's mean and p95, don't guess**. At γ=1 the teacher's p95 is 0.068 against bp's 0.248; at γ=0.5 it's 0.207. Somewhere below 0.5 matches. That's a two-parameter fit on the anchor corpus, done once, and it's what makes α a *shape* knob instead of a level knob.

### Two other things the maps confirm

**Time offset is real: −3 frames (35 ms).** Correlation is 0.7396 at zero shift, peaking at 0.7551 at −3, on two maps that both nominally start at t=0.000 with identical `dt=0.011610`. Free to fix, and worth it — an uncorrected 3-frame offset blurs every ridge.

**There's genuine information to transfer.** Aligned in-band correlation between model3 and your *already fine-tuned* e40 is 0.755. Substantial agreement, substantial disagreement remaining — which is the precondition for the whole idea being worth doing. If it had been 0.95 there'd be nothing to teach.

Want me to write the teacher-building step — regrid, fit the alignment offset, fit (floor, γ) against the anchor corpus, cache to npz — as a module `finetune/teacher.py` that `build_distill_pool` calls?

---

## Would you recommend first tuning basic-pitch to match the model3 model and then do a separate finetune to raise voices that are not that salience in the mix?

That decision hinges on something we can now actually measure. Let me check whether the `alles` score is available.

[Found DiggyDex_Alles.mid and ran finetune/compare_voice_salience.py on baseline, model3, and e40 salience maps against it.]

I ran the evenness comparison on `alles` with all three maps — baseline, model3, and your e40. It overturns the premise, so answer the strategy question after this:

```
MEAN OVER 11 CHORDS
  model                     spread    min/max
  nmp_salience_baseline      0.423      0.577
  model3_exp3multif0         0.511      0.489     ← LESS even than baseline
  nmp_finetuned_conv_e40     0.387      0.613     ← most even of the three
```

**model3 is the least even of the three models.** It is not a teacher for the thing you want to teach. Your run03 model already beats it on the target metric.

Chord 9 shows the mechanism exactly:

| voice | baseline | model3 | e40 |
|---|---|---|---|
| Soprano B4 (the quiet one) | 0.170 (rel 0.39) | 0.231 (rel **0.26**) | 0.233 (rel 0.28) |
| Tenor B3 | 0.436 (rel 1.00) | 0.879 (rel 1.00) | 0.818 (rel 1.00) |

model3 raises the soprano in absolute terms — and raises the tenor twice as much, so *relatively* the soprano ends up worse than under pretrained bp. That's the same pattern across the excerpt: model3's advantage is a global contrast increase that preserves the prominence ranking, not a redistribution across voices.

### So: no, and not in the other order either

The two-stage plan assumed the stages compose — stage 1 installs model3's better representation, stage 2 evens it out. On this evidence stage 1 installs model3's *prominence ranking*, which is the thing stage 2 then has to remove. You'd be teaching a bias and then paying to unlearn it, and the unlearning happens on 167 labelled windows against a teacher fitted on hours of audio. The small, correct signal would lose.

Joint training doesn't rescue it either — it just moves the same conflict inside one loss, where α and `focal_gamma` pull against each other and your report acquires another un-attributed multi-variable run.

### What the measurement does *not* kill

There's a real asset in model3 that's worth separating out. Its whole-map sparsity matches yours almost exactly — `frac>0.5` is 0.0117 for model3 against 0.0132 for e40 — yet it reads far higher *on the scored voices* (0.81 vs 0.65 in chord 6). Same fraction of the map is hot, but more of that hot mass sits on true pitches. That's genuine **contrast**, not just gain, and it survives calibration.

So the model3 teacher isn't worthless; it's mis-scoped. It's a detection/precision fix, and it should be judged on the §4.2 threshold-range metrics, not on spread and min/max. Whether that's worth the regrid + alignment + calibration machinery is a different trade than the one you've been weighing, and given run02 was already at detection parity with pretrained, I'd rate it low priority.

### What I'd do instead

The lever that demonstrably works on evenness is the one you already have. run03 differs from baseline by `--focal_gamma 1`, and it's the most even of the three models here. Your own docstring at finetune/finetune.py:270-280 works out that γ=2 takes the quiet-voice gradient advantage from ~5x to ~28x. That's a **one-knob change on an existing pipeline**, ~680 steps, with clean attribution — exactly what §5 says your runs are missing — versus building a teacher that measures worse than your current model on the target.

Sweep `--focal_gamma` ∈ {1, 2, 3} and keep everything else at run03's settings.

### The one thing that could revive it

You mentioned a finetuned AdaBN model3. This measurement used the stock `exp3multif0` weights. If the AdaBN version was adapted on your choir it may well be more even — and it's one command to check, same three-way comparison. Worth doing before writing the idea off; if it comes out at spread < 0.387 it's a live teacher again and the whole plan is back on, joint rather than staged.

Want me to add this three-way comparison to `nmp_salience_finetuned.md`? It's a §4.2-shaped result and it's the first evenness number you have for e40 on held-out material.

---

## When looking at salience maps. model3's seems to have much more contrast than the finetune basic_pitch model's. The basic_pitch version shows much more sub harmonics.

Your eye is on something real and it has a clean architectural cause. Let me quantify it.

[Computed band-relative contrast (below sub-octave, sub-octave of bass, voice band, 2nd harmonic region, upper harmonics) and sub-octave ghost statistics for all three models on the alles excerpt.]

Both halves of your observation check out, and the second one is more consequential than it looks. Measured over the whole excerpt, by frequency band (the voices span E3 164.8 Hz to B4 493.9 Hz):

```
band mean / voice-band mean      ← lower off-voice = more contrast
              <82Hz   82-165Hz   VOICES   500-1000   1000-2000
  baseline    0.879     0.934     1.000     0.889      0.865
  model3      0.004     0.366     1.000     0.010      0.000
  e40         0.812     0.962     1.000     0.411      0.325
```

**Contrast**: confirmed, and it's the floor. model3 puts 0.4% of its voice-band level in the octave below the bass and essentially nothing above; bp sits at 81–96% everywhere because of its ~0.10 background. e40 cleaned up the harmonic region above (0.89→0.41, 0.87→0.33) but left the sub-octave untouched (0.93→0.96).

**Subharmonics**: confirmed, and it got *worse* with fine-tuning. In 82–165 Hz — below the lowest sung note:

| | frac > 0.3 | p99.9 | voice-band p99.9 |
|---|---|---|---|
| baseline | 0.0087 | 0.450 | 0.624 |
| model3 | 0.0221 | 0.778 | 0.955 |
| **e40** | **0.0599** | **0.839** | 0.859 |

e40's strongest sub-octave ghosts reach 0.839 against 0.859 for its actual voices — **the ghosts are as strong as the singers**, and there are 7× more of them than in baseline. model3's ghosts sit well below its voices (0.778 vs 0.955).

### This invalidates the evenness comparison I gave you last turn

Look at the voicing: **Sopraan B4 / Alt E4 / Tenor B3 / Bas E3**. B4→B3 is exactly an octave. E4→E3 is exactly an octave. And B4/3 = 164.6 Hz, which is E3 to within a cent.

Every lower voice in this excerpt sits precisely on a subharmonic of an upper voice. So when `compare_voice_salience.py` reads salience at the tenor's B3, part of what it reads is the soprano's octave ghost. The model with the most leakage scores as the most even — and that is exactly the ordering we got: e40 leaks most (0.0599) and scored most even (spread 0.387); model3 leaks least relative to its voices and scored least even (0.511).

I reported that model3 is a worse teacher for evenness. On this excerpt that conclusion isn't supported — the metric can't separate "holds the quiet low voice well" from "ghosts the loud high voice down an octave". It needs re-measuring on material without octave doubling before it decides anything. You have `late_nolowbass` takes; whatever prompted those is the same problem.

### Why fine-tuning made it worse — and it's not a mystery

The labels are score-derived, so a ridge is placed at every sung pitch. In octave-doubled choral voicing, **the ghost position is also a labelled position**. A subharmonic error at B3 under a B4 is never penalised as a false positive, because a tenor really is singing B3. With `--pos_weight 3` and `--focal_gamma 1` driving hard on those bins, the training signal actively rewards whatever produces activity there — including the octave-ghost pathway. Your loss has no way to tell the two apart on this material.

That explains the 7× increase, and it means no amount of tuning the supervised term fixes it.

### Which revives the model3 teacher — for a third purpose

We've now ruled out evenness (unmeasured) and global contrast (that's calibration, which you'd strip). But subharmonic suppression is different, and it's the one that's **architecturally learnable**.

`HarmonicStacking` hands the contour conv exactly the channels needed: at output bin *b*, the h=2 channel reads an octave above, h=3 a twelfth above, h=0.5 an octave below. The rule "weak evidence at h=1, strong at h=2 and h=3 → suppress" is content-based and translation-equivariant in log-frequency, so a shared kernel can represent it perfectly — unlike the absolute-band zeroing we rejected. The pretrained 2-layer head simply doesn't use those channels well, and run03 pushed it further in the wrong direction.

model3, with 8 layers and two (70,3) harmonic kernels, clearly does use them. That's a real, transferable, learnable difference — and it plausibly accounts for a good share of the 0.755 correlation gap.

So the teacher plan is back on, scoped to this: **distil model3's harmonic reasoning, evaluate on sub-octave ghost rate and on detection precision, not on spread.** And drop the whole calibration argument — you now know exactly which part of model3 you want and it isn't the level.

Caveats worth holding: one excerpt; I assumed nothing is sung below E3 across all 11 chords (true for the 6 I can see); and I've shown the sub-octave activity is *there*, not that the voices cause it — though model3 reading 0.0002 below 82 Hz on the same audio makes room rumble unlikely.

The measurement I'd want next is the ghost rate on material with no octave doubling, for all three models. Do you have a take where the voices aren't octave-paired — or should we use `late_nolowbass`?
