"""Fine-tune Basic Pitch's salience (contour) head on real choir recordings.

Basic Pitch under-reads a quiet voice in an ensemble. On the blocked chords
prepare_real_chords.py extracts it reads a median of 0.13 at annotated voice
bins against 0.10 in the background, so a soft part is barely distinguishable
from silence. This adapts the salience path -- two convolutions, 7.7k
parameters -- to real choir audio without losing what the pretrained model
already knows.

Input is one npz per window from prepare_real_chords.py, holding `audio`
(AUDIO_N_SAMPLES samples), `tgt` (frames x 264 salience ridges) and `mask`
(per frame, 1 where the score-derived labels hold). The mask carries real
weight: inside a chord's attack, release or reverb tail the score cannot
distinguish a staggered entry from a decaying voice from a quiet one, so no
target is defensible there -- but those frames are still wanted as INPUT,
because they feed the receptive field of the frames that are labelled.

--strategy chooses what adapts:

    bn   : the two BatchNorms only -- 18 parameters, too few to matter here.
    conv : the two convolutions, BatchNorm frozen (7,697).
    full : both (7,715).

'head' is intentionally absent: a quiet voice is lost before the head, so
adapting only the head cannot recover it. And the bn/conv distinction that
matters for a deep network barely exists in a two-layer head -- conv and full
differ by 18 parameters.

Three things hold the fine-tune in place. With 7.7k parameters against a few
minutes of audio they are not optional:

  * --l2sp anchors the conv kernels to their pretrained values.

  * --distill_dir anchors OUTPUTS on real audio. The target is whatever the
    PRE-TRAINED model itself predicts for that audio, so the term simply says
    "do not change here" -- no annotation needed. Weight-space distance is a
    poor proxy for output-space distance, so this constrains what --l2sp
    cannot, on the input distribution that actually matters.

  * --real_audio is the held-out screen: every checkpoint's salience map is
    compared with the pretrained model's on the same file, and one that buys
    validation recall by compressing real-audio salience is rejected rather
    than saved. Do not run without it. The characteristic failure is a model
    that scores better on the validation windows and worse on everything else,
    and nothing else in the loop can see that happening.

Checkpoints are ranked on validation F, and must not regress recall or precision
(--bal_tol) or drift on real audio (--drift_high_tol). Small
learning rate, few epochs.
"""

from __future__ import print_function

import os
import re
import sys
import csv
import json
import glob
import argparse
import datetime

import numpy as np
import tensorflow as tf

# allow running from anywhere: the repo root and its scripts/ dir
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bp_grid
# The salience model is defined by the converter, which is also what turns the
# result of this script back into a deployable SavedModel -- one definition of
# the architecture and one weight loader, shared by training and export.
from convert_model_to_savedmodel import (DEFAULT_MODEL_DIR, build_salience_model,
                                         load_pretrained_weights)
from basic_pitch.constants import AUDIO_N_SAMPLES, FFT_HOP
from basic_pitch import inference

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

CHUNK_LEN = 2000          # time frames per model.predict call (for eval)

# Frames to drop from each end of a window before comparing student and teacher.
#
# Zero, and the receptive field is not why. A crop is needed when the teacher
# runs whole-file while the student sees a padded window, because then their
# outputs cannot agree near the edges. Here both run on the SAME fixed-length
# window through the same graph, so whatever the boundary does it does
# identically to both and cancels. Cropping would only discard usable signal.
DISTILL_EDGE_CROP = 0

# Teacher peak below this and the window counts as 'quiet'. Measured, not
# inherited: 0.10 is the obvious guess but Basic Pitch's contour floor sits at
# ~0.10 by construction (label smoothing trains the background toward it), so
# that threshold would tag nothing. On real choir audio the per-window peak runs
# 0.52-0.75 across the 5th-95th percentile wherever anyone is singing, so 0.30
# sits clear of both the floor and the quietest singing.
_QUIET_MAX = 0.30


# --------------------------------------------------------------------------
# Logging: mirror stdout to a log file so a long run's results survive the
# terminal scrollback (or a lost ssh session).
# --------------------------------------------------------------------------
class Tee(object):
    """Duplicate stdout to a log file, line-buffered so nothing is lost if the
    run is killed. In-place progress updates (the '\\r batch i/n' line) go to
    the terminal only, keeping the log line-oriented."""

    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, 'w', buffering=1)

    def write(self, s):
        self.terminal.write(s)
        if not s.startswith('\r'):
            self.log.write(s)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_logging(out_weights, args):
    """Start teeing stdout to a NEW log per run, next to the output weights and
    stamped with the start time:

        <out minus .weights.h5>_YYYYmmdd-HHMMSS.log

    One file per run rather than one appended file, so runs can be compared
    side by side and a re-run can never be mistaken for a continuation of the
    previous one. Returns the log path."""
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    path = "%s_%s.log" % (out_weights[:-len('.weights.h5')], stamp)
    sys.stdout = Tee(path)
    print("=" * 78)
    print("run started %s" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("args: %s" % json.dumps(vars(args), sort_keys=True))
    print("=" * 78)
    return path














# --------------------------------------------------------------------------
# Prepare: featurize + slice chords + cache to disk (one npz per chord)
# --------------------------------------------------------------------------
def is_prebuilt_cache(d):
    """True if `d` is a finished window directory: a DONE marker and *.npz
    sitting directly in it, as prepare_real_chords.py writes."""
    return (os.path.exists(os.path.join(d, 'DONE'))
            and bool(glob.glob(os.path.join(d, '*.npz'))))


def prepare(train_dir):
    """Return the window files to train on.

    Windows are built ahead of time by prepare_real_chords.py, which writes
    them as DONE + *.npz directly in `train_dir`.
    """
    if not is_prebuilt_cache(train_dir):
        raise SystemExit(
            "%s is not a prepared window directory (no DONE + *.npz). "
            "Build it with prepare_real_chords.py first." % train_dir)
    wins = sorted(glob.glob(os.path.join(train_dir, '*.npz')))
    print("Using %d prepared windows in %s" % (len(wins), train_dir))
    return wins


def select_first_chords(win_files, n_chords):
    """Keep only windows belonging to the first `n_chords` chords OF EACH FILE.

    A window is centred on one chord but its 2 s of context can reach others, so
    the chord indices it covers are stored in the npz's `chords` field (1-based)
    rather than encoded in the filename."""
    kept = []
    for p in win_files:
        with np.load(p) as d:
            chords = d['chords'] if 'chords' in d.files else None
        if chords is not None and len(chords) and int(chords.min()) <= n_chords:
            kept.append(p)
    return kept


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def set_trainable(model, strategy):
    """strategy in {'bn', 'conv', 'full'}. 'head' is intentionally absent: a quiet
    voice is lost before the head, so adapting only the head cannot recover it."""
    if strategy == 'full':
        for layer in model.layers:
            layer.trainable = True
    elif strategy == 'bn':
        # Only BatchNorm adapts (conv weights frozen). trainable=True also keeps
        # BN in training mode, so its running statistics recalibrate to the new
        # amplitude distribution.
        for layer in model.layers:
            layer.trainable = isinstance(layer, tf.keras.layers.BatchNormalization)
    elif strategy == 'conv':
        # The inverse of 'bn': adapt the conv/dense weights, freeze BatchNorm.
        #
        # Keras special-cases BatchNormalization -- trainable=False also puts the
        # layer in INFERENCE mode, even when the model is called with
        # training=True -- so gamma/beta AND the running mean/variance are all
        # held fixed. The normalisation therefore stays as pretrained instead of
        # drifting toward this handful of takes. That drift is what 'bn' does by
        # design and what 'full' does incidentally: --l2sp anchors only variables
        # whose name contains 'kernel', and running statistics are not trainable
        # variables at all, so nothing can anchor them.
        #
        # Pair with --l2sp, which does apply here, and verify with --real_audio.
        for layer in model.layers:
            layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    else:
        raise ValueError("unknown strategy %r (use 'bn', 'conv' or 'full')" % strategy)


def build_model(weights_path, strategy):
    """Basic Pitch's salience path, with pretrained weights.

    `weights_path` is either a Basic Pitch SavedModel directory (weights are
    read out of its variables checkpoint) or a .weights.h5 written by a previous
    run of this script, so a fine-tune can be resumed or continued.
    """
    model = build_salience_model()
    if weights_path.endswith('.h5'):
        model.load_weights(weights_path)
        print("Loaded weights from %s" % weights_path)
    else:
        load_pretrained_weights(model, weights_path)
    set_trainable(model, strategy)
    # Report weight-BEARING layers only: the CQT, stacking and reshape layers
    # hold no variables, so listing them as "trainable" would overstate what is
    # actually being adapted.
    tr = [l.name for l in model.layers if l.trainable and l.weights]
    n_par = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print("Strategy '%s': %d trainable layer(s), %d parameters %s"
          % (strategy, len(tr), n_par, tr))
    if strategy == 'bn':
        print("  note: only %d trainable parameters -- the salience path has just "
              "two BatchNorms. Their running statistics adapt as well, but there "
              "is very little here to move." % n_par)
    return model


def make_bkld(pos_weight=1.0):
    """Binary cross-entropy, optionally upweighting positive (annotated voice)
    target bins by pos_weight.

    No label smoothing: Basic Pitch pretrained with 0.2 against single-bin
    targets, but these targets are ridges whose shoulders carry the sub-bin
    pitch, and smoothing squashes exactly those.

    Accepts an optional per-frame `mask` of shape (batch, T): 1 where the label
    is known, 0 where it is not. Real recordings need this. Inside a chord's
    attack, release or reverb tail the score cannot distinguish a staggered
    entry from a decaying voice from a quiet one, so no target is defensible
    there -- but those frames are still wanted as INPUT, because they feed the
    receptive field of the frames that are labelled. Masking lets them do that
    without contributing gradient.

    The mask normalises by its own sum, not by the element count, so a batch of
    mostly-masked windows is not silently scaled down relative to a full one.
    """
    eps = 1e-7
    pw = float(pos_weight)

    def loss(y_true, y_pred, mask=None):
        y_true = tf.clip_by_value(y_true, eps, 1.0 - eps)
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        per = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        if pw != 1.0:
            w = 1.0 + (pw - 1.0) * tf.cast(y_true > 0.5, per.dtype)
            per = per * w
        if mask is None:
            return tf.reduce_mean(per)
        # per is (batch, T, F); the mask is per FRAME, so it broadcasts over F.
        m = tf.cast(mask, per.dtype)[:, :, tf.newaxis]
        denom = tf.reduce_sum(m) * tf.cast(tf.shape(per)[2], per.dtype)
        return tf.reduce_sum(per * m) / tf.maximum(denom, eps)

    return loss


def make_distill_loss(gamma=0.0, crop=DISTILL_EDGE_CROP):
    """Real-audio anchor: how far the student's salience has moved from the
    teacher's on the same audio.

    SYMMETRIC by design. The failure looks like compression, which tempts a
    one-sided penalty on downward deviation -- but calibration drifts in both
    directions, and inflating the background is just as destructive as
    compressing the voices, so a one-sided penalty would let half of it through.

    `gamma` weights the penalty by the teacher's own confidence, t**gamma:
      gamma=0  uniform -- pin everything equally. Also pins the quiet voices the
               teacher misses, which is exactly what we want to change, so this is
               the conservative baseline, not the ideal.
      gamma>0  pin hard where the teacher reads high, weakly where it reads near
               zero: "keep what you already know, stay free where you were
               unsure". Lets a quiet voice rise without licensing a general
               inflation of the background.

    Both variants are normalised by the mean weight, so the term's scale (and
    hence a given --distill_lambda) does not shift when gamma changes."""
    eps = 1e-7
    g = float(gamma)

    def loss(t_true, y_pred):
        if crop:
            # time is axis 1 here, not 2
            t_true = t_true[:, crop:-crop, :]
            y_pred = y_pred[:, crop:-crop, :]
        t = tf.clip_by_value(t_true, eps, 1.0 - eps)
        p = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        per = -(t * tf.math.log(p) + (1.0 - t) * tf.math.log(1.0 - p))
        if g > 0.0:
            w = tf.pow(t, g)
            return tf.reduce_sum(w * per) / (tf.reduce_sum(w) + eps)
        return tf.reduce_mean(per)

    return loss


def audio_windows(path):
    """Overlapping fixed-length windows of `path`, exactly as inference does.

    Returns (windows, original_length) with windows shaped
    (n, AUDIO_N_SAMPLES, 1). Reusing basic_pitch.inference's own windowing is
    not fussiness: NormalizedLog rescales by each window's dynamic range, so a
    different partition changes every frame's value rather than just the seams.
    """
    n_olap = inference.DEFAULT_OVERLAPPING_FRAMES
    overlap_len = n_olap * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len
    wins, orig_len = [], None
    for windowed, _t, orig_len in inference.get_audio_input(path, overlap_len, hop_size):
        wins.append(windowed)
    return np.concatenate(wins, axis=0), orig_len


def build_distill_pool(model, distill_dir, quiet_cap=0.33, exclude=(), rng=None):
    """Real-audio windows paired with the PRETRAINED model's salience on them.

    Call before the first optimizer step, while `model` still holds the
    untouched weights: the teacher is frozen by definition, so its outputs are
    computed once here and become plain target arrays. No second model is ever
    in memory.

    Held in RAM rather than cached to npz: a window is just audio and the
    teacher pass is one batched predict per file, so there is nothing expensive
    enough to be worth caching.

    Windows whose teacher peak is below _QUIET_MAX are capped at `quiet_cap` of
    the pool rather than dropped: a window whose correct answer is "stay near
    zero" is the cheapest available constraint against upward inflation, but it
    must not crowd out windows containing actual singing.

    `exclude` is the --real_audio drift screen. Overlap is a hard error:
    training on those files would turn the only held-out signal in the log into
    a training metric.
    """
    excl = {os.path.realpath(p) for p in exclude}
    files = sorted(f for ext in ('*.wav', '*.flac', '*.ogg')
                   for f in glob.glob(os.path.join(distill_dir, ext)))
    if not files:
        raise SystemExit("No audio in %s" % distill_dir)
    clash = [f for f in files if os.path.realpath(f) in excl]
    if clash:
        raise SystemExit("--distill_dir overlaps --real_audio: %s"
                         % ', '.join(os.path.basename(f) for f in clash))

    audio, teacher = [], []
    for f in files:
        wins, _ = audio_windows(f)
        sal = model.predict(wins, verbose=0)
        audio.append(wins)
        teacher.append(sal)
        print("  %-40s %d windows" % (os.path.basename(f)[:40], len(wins)))
    audio = np.concatenate(audio, 0)
    teacher = np.concatenate(teacher, 0)

    quiet = teacher.reshape(len(teacher), -1).max(1) < _QUIET_MAX
    n_q, n_l = int(quiet.sum()), int((~quiet).sum())
    if n_l and quiet_cap < 1.0 and n_q:
        allowed = int(quiet_cap * n_l / max(1e-9, 1.0 - quiet_cap))
        if n_q > allowed:
            qi = np.where(quiet)[0]
            drop = (rng or np.random.RandomState(0)).permutation(len(qi))[allowed:]
            keep = np.ones(len(audio), bool)
            keep[qi[drop]] = False
            audio, teacher = audio[keep], teacher[keep]
            print("  capped quiet windows to %d (%.0f%% of pool)"
                  % (allowed, 100.0 * allowed / (allowed + n_l)))
    return audio, teacher


def predict_salience(model, path):
    """Whole-file (T, F) salience for the audio at `path`.

    Basic Pitch consumes one fixed-length window, so a whole file is scanned
    with overlapping windows and stitched by inference.unwrap_output, which
    drops half the overlap from each window's output. Verified to reproduce
    basic_pitch.inference.predict's contour to 2e-07.
    """
    n_olap = inference.DEFAULT_OVERLAPPING_FRAMES
    hop_size = AUDIO_N_SAMPLES - n_olap * FFT_HOP
    wins, orig_len = audio_windows(path)
    sal = model.predict(wins, verbose=0)
    return inference.unwrap_output(sal, orig_len, n_olap, hop_size)






# --------------------------------------------------------------------------
# Distillation pool: real audio, no annotation.
#
# The loss is measured only on a few minutes of blocked chords, so the cheapest
# way for the optimiser to reduce it is to recalibrate the model to that narrow
# slice -- which changes the salience map on everything else. Nothing in the
# supervised term opposes that.
#
# The fix is a second loss term on real audio. No annotation is required: the
# target is what the PRE-TRAINED model predicts for that audio, so the term says
# "do not change here". That is the same reference drift_stats() compares every
# checkpoint against -- this moves it out of the accept/reject decision and into
# the gradient, where the optimiser can steer around the failure instead of
# stumbling into it and being rejected.
#
# Equivalently: --l2sp anchors WEIGHTS to their pretrained values; this anchors
# OUTPUTS on the input distribution that actually matters. Weight-space distance
# is a poor proxy, and under --strategy bn it anchors nothing at all -- no
# variable is named 'kernel', hence "L2-SP anchoring 0 kernels" in the log.
# --------------------------------------------------------------------------















def evaluate_real(model, valid_dir, thresh, loss_fn, n_chords=None):
    """Validate on the prepared windows in `valid_dir`, one npz per window.

    Reports recall/precision/loss over the SUPERVISED frames only (mask == 1):
    the score-derived attack/release/reverb margins prepare_real_chords.py
    masked out have no defensible target, and scoring them would only add noise
    to the comparison against baseline.

    `thresh` None re-tunes the peak threshold on this checkpoint and returns the
    value it chose. That is not a convenience -- an absolute threshold is not
    comparable across checkpoints here. Basic Pitch pretrained with label
    smoothing, so its background sits near 0.10; these targets are 0 in the
    background, so the first thing fine-tuning does is drop that floor. Peaks
    then clear any fixed threshold more easily, and a threshold picked from the
    pretrained distribution progressively understates every later checkpoint.
    Tuning per checkpoint scores each model at its own best operating point.

    Chosen by F rather than recall: recall alone is maximised by thresholding at
    zero, so the threshold has to be picked by something that prices in the
    false positives it buys.
    """
    import mir_eval
    wins = sorted(glob.glob(os.path.join(valid_dir, '*.npz')))
    if n_chords is not None:
        wins = select_first_chords(wins, n_chords)
    if not wins:
        return None

    # Salience once per window; the threshold sweep below is peak-picking only.
    cached, losses = [], []
    for p in wins:
        with np.load(p) as d:
            audio, tgt, mask = d['audio'], d['tgt'], d['mask']
        if mask.sum() == 0:
            continue
        sal = model.predict(audio[np.newaxis, :, np.newaxis], verbose=0)[0]
        losses.append(float(loss_fn(
            tf.constant(tgt[np.newaxis]), tf.constant(sal[np.newaxis]),
            tf.constant(mask[np.newaxis].astype(np.float32)))))
        idx = np.where(mask > 0.5)[0]
        times = bp_grid.get_time_grid(mask.shape[0])
        # One reference per VOICE, not per bin over threshold -- see
        # bp_grid.target_to_multif0 for why the difference matters.
        ref = bp_grid.target_to_multif0(tgt)
        cached.append((sal, times[idx], [ref[i] for i in idx], idx))
    if not cached:
        return None

    def score(th):
        rec, prec = [], []
        for sal, t, ref, idx in cached:
            _, est = bp_grid.salience_to_multif0(sal, th)
            m = mir_eval.multipitch.evaluate(t, ref, t, [est[i] for i in idx])
            rec.append(m['Recall']); prec.append(m['Precision'])
        r, pr = float(np.mean(rec)), float(np.mean(prec))
        return r, pr, 2 * r * pr / max(r + pr, 1e-9)

    if thresh is None:
        best = None
        for th in THRESH_GRID:
            r, pr, f = score(th)
            if best is None or f > best[3]:
                best = (th, r, pr, f)
        th, r, pr, f = best
    else:
        th = thresh
        r, pr, f = score(th)
    return dict(recall=r, precision=pr, f=f, thresh=th, loss=float(np.mean(losses)))


def guard_failures_real(inv, baseline, args):
    """Recall/precision against the pre-training baseline, plus the real-audio
    drift screen when --real_audio was given.

    Only `real_d_high` gates. The whole-map mean (`real_rel_mean`) is reported
    but never rejects, because it is not measuring what it appears to: about 93%
    of the map is background, so it is dominated by the background level. Basic
    Pitch pretrained with label smoothing and so carries a background floor near
    0.10; these targets are 0 there, so every run drops that floor and the mean
    falls ~40% on the first epoch while the voices are untouched. Gating on it
    rejects every checkpoint for doing exactly what it was asked to do.
    """
    if baseline is None:
        return []
    failed = []
    for key in ('recall', 'precision'):
        if inv[key] < baseline[key] - args.bal_tol:
            failed.append("%s %.3f<%.3f" % (key, inv[key], baseline[key] - args.bal_tol))
    if 'real_d_high' in inv and inv['real_d_high'] < -args.drift_high_tol:
        failed.append("real d@high %.3f" % inv['real_d_high'])
    return failed


def format_metrics_real(m):
    return ("val_loss=%.4f  recall=%.3f  precision=%.3f  F=%.3f  @thresh=%.3f"
            % (m['loss'], m['recall'], m['precision'], m['f'], m['thresh']))


# --------------------------------------------------------------------------
# Real-audio drift screen (no annotation required)
# --------------------------------------------------------------------------
# Fitting a few minutes of blocked chords can silently recalibrate the model.
# When that happens the salience map on REAL audio is compressed downward --
# confident activations pulled down hardest -- which destroys detections while
# the validation windows above still report an improvement.
#
# Catching it needs no MIDI, no alignment, no tuning correction and no ground
# truth: run the current weights over a few seconds of real audio and compare
# the salience map with the one the PRE-TRAINED model produced for the same
# file. Both come from the same audio, so they line up bin for bin.
HIGH_PCT = 99.5           # percentile of the baseline defining its 'confident' band

# Candidate peak thresholds swept per checkpoint when --thresh is left at auto.
# Starts well below the pretrained model's 0.10 floor, because a fine-tuned
# model's background drops toward 0 and its best operating point moves with it.
THRESH_GRID = np.round(np.arange(0.02, 0.451, 0.015), 4)


def drift_stats(base_sals, sals):
    """Compare candidate salience maps against the pre-training ones, per real
    excerpt, and average. Several short excerpts are much better than one long
    one here: drift shows up as a consistent shift across independent material,
    and a single file can mislead.

    `real_d_high` is the change where the baseline is most confident, and is
    the only one of these that gates a checkpoint. `real_rel_mean`, the change
    over the whole map, is a DIAGNOSTIC: ~93% of the map is background, so it
    mostly reports where the background floor sits, which every run moves.

    That band is defined as the baseline's own top HIGH_PCT of bins, not the
    absolute 0.80-0.90 an uncalibrated map would use. The contour never
    reaches 0.80 --
    label smoothing caps it around 0.7, and on real choir audio the 0.80-0.90
    band holds 1 bin in 5 million -- so an absolute band would leave the guard
    permanently switched off by the `hi.sum() >= 50` fallback. A percentile band
    asks the question that was actually meant: is the model backing off where it
    used to be surest?

    `real_worst` is the least favourable per-file mean change, so one bad
    excerpt cannot be averaged away."""
    per = []
    for b, s in zip(base_sals, sals):
        hi = b >= np.percentile(b, HIGH_PCT)
        per.append((s.mean() / b.mean() - 1.0 if b.mean() else 0.0,
                    float((s[hi] - b[hi]).mean()) if hi.sum() >= 50 else 0.0,
                    float(np.corrcoef(b.ravel().astype(np.float64),
                                      s.ravel().astype(np.float64))[0, 1])))
    a = np.mean(per, axis=0)
    return dict(real_rel_mean=float(a[0]), real_d_high=float(a[1]),
                real_r=float(a[2]), real_worst=float(min(p[0] for p in per)),
                real_n=len(per))


def format_drift(m):
    """d@high is the gate; mean and r are diagnostics -- see guard_failures_real."""
    s = ("REAL(%d) d@high %+.3f | mean %+.1f%% r=%.2f"
         % (m['real_n'], m['real_d_high'], 100 * m['real_rel_mean'], m['real_r']))
    if m['real_n'] > 1:
        s += " worst %+.1f%%" % (100 * m['real_worst'])
    return s


def format_train_loss(pairs, distilling):
    """Mean (supervised, distillation) training loss over a list of per-step pairs.
    The two terms are reported SEPARATELY, never summed: the whole point is to
    watch the trade-off between fitting the chords and holding the
    real-audio calibration, and a single total hides it."""
    a = np.mean(pairs, axis=0)
    if not distilling:
        return "train_loss=%.4f" % a[0]
    return "train_loss=%.4f distill=%.4f" % (a[0], a[1])






# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
def train(args):
    # Keras 3 requires weight files to end in .weights.h5
    if not args.out.endswith('.weights.h5'):
        args.out = (args.out[:-3] if args.out.endswith('.h5') else args.out) + '.weights.h5'
        print("Adjusted output weights path to %s (Keras 3 requirement)" % args.out)

    # Tag the output with what produced it: the adaptation strategy, plus TEST
    # for a smoke run (whose weights, trained on a handful of chords, are not a
    # usable model). Keeps strategies and smoke tests from overwriting each
    # other's results; the log filename follows suit.
    suffix = '_' + args.strategy + ('_TEST' if args.TEST is not None else '')
    args.out = args.out[:-len('.weights.h5')] + suffix + '.weights.h5'

    log_path = setup_logging(args.out, args)
    print("Output weights: %s" % args.out)
    if args.TEST is not None:
        print("--TEST run: output marked TEST -- not a usable model")
    print("Logging to %s" % log_path)

    win_files = prepare(args.train_dir)
    if not win_files:
        raise SystemExit("No windows to train on. Render the MIDIs to wav first.")
    if args.TEST is not None:
        n_all = len(win_files)
        win_files = select_first_chords(win_files, args.TEST)
        if not win_files:
            raise SystemExit("--TEST %d selected no windows." % args.TEST)
        print("--TEST: first %d chord(s) per file -> %d of %d windows"
              % (args.TEST, len(win_files), n_all))

    # Validation windows come from prepare_real_chords.py the same way the
    # training ones do: fixed-length npz sitting directly in valid_dir.
    if args.valid_dir and not is_prebuilt_cache(args.valid_dir):
        raise SystemExit(
            "%s is not a prepared window directory (no DONE + *.npz)."
            % args.valid_dir)
    if args.valid_dir and args.TEST is not None:
        print("--TEST: validating on the first %d chord(s) of each file" % args.TEST)

    # Real-audio drift screen. Basic Pitch reads audio directly, so there is
    # nothing to featurize up front -- just resolve the paths and check them.
    real_feat = []
    for path in (args.real_audio or []):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise SystemExit("--real_audio file not found: %s" % path)
        real_feat.append(path)
        print("Real-audio drift screen: %s" % os.path.basename(path))

    model = build_model(args.weights, args.strategy)
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = make_bkld(args.pos_weight)
    # pos_weight is deliberately NOT applied to the anchor: its 'y_true > 0.5'
    # test would upweight bins wherever the TEACHER happened to be confident,
    # which is unrelated to the annotated-voice reweighting it exists for.
    distill_fn = make_distill_loss(args.distill_gamma)

    # L2-SP: snapshot pretrained conv/dense kernels so we can penalise deviation
    # from them (anchors 'full' fine-tuning against forgetting).
    anchors = []
    if args.l2sp > 0:
        anchors = [(v, tf.constant(v.numpy()))
                   for v in model.trainable_variables if 'kernel' in v.name]
        print("L2-SP anchoring %d kernels (lambda=%g)" % (len(anchors), args.l2sp))

    @tf.function
    def train_step(x, y, rx=None, rt=None, ymask=None):
        with tf.GradientTape() as tape:
            loss_s = loss_fn(y, model(x, training=True), ymask)
            loss = loss_s
            loss_r = tf.constant(0.0)
            if rx is not None:
                # Separate forward pass, NOT one concatenated batch: BatchNorm in
                # training mode normalises by batch statistics, and a mixed
                # batch would give a blended statistic matching neither the
                # chord windows nor the unannotated audio. Under --strategy bn
                # this second pass does double duty -- it also puts the wider
                # material into the running-mean/variance updates, which is what
                # stops them converging onto the handful of prepared takes.
                loss_r = distill_fn(rt, model(rx, training=True))
                loss = loss + args.distill_lambda * loss_r
            if anchors:
                loss = loss + args.l2sp * tf.add_n(
                    [tf.reduce_sum(tf.square(v - v0)) for v, v0 in anchors])
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss_s, loss_r

    # Pre-training baseline so we can require the other metrics not to regress.
    baseline = (evaluate_real(model, args.valid_dir, args.thresh, loss_fn,
                              n_chords=args.TEST)
                if args.valid_dir else None)
    if baseline is not None:
        print("baseline     | " + format_metrics_real(baseline))

    # Pre-training salience on the real excerpt: the reference every epoch is
    # compared against. Taken from the untouched weights, so it is the model's
    # own real-audio behaviour before any training data was seen.
    base_real = [predict_salience(model, path) for path in real_feat]

    # Distillation pool. Built HERE, before the first optimizer step, so `model`
    # still holds the pretrained weights and is its own teacher -- no second copy
    # in memory, the same trick base_real uses.
    rng = np.random.RandomState(args.seed)
    next_real = None
    if args.distill_dir:
        d_audio, d_teacher = build_distill_pool(
            model, args.distill_dir, args.distill_quiet_cap,
            exclude=real_feat, rng=np.random.RandomState(args.seed + 1))
        print("Distilling on %d real windows (%.1f min), lambda=%g gamma=%g"
              % (len(d_audio), len(d_audio) * AUDIO_N_SAMPLES / 22050.0 / 60,
                 args.distill_lambda, args.distill_gamma))

        # Its OWN RandomState, deliberately not `rng`: if the pool drew from the
        # same generator it would shift every subsequent epoch shuffle, so a
        # --distill_lambda 0 ablation would not see the same ordering as a
        # --distill_lambda 1 run and the comparison would be confounded.
        drng = np.random.RandomState(args.seed + 2)

        def make_cycler(n_items, batch, r):
            order, pos = [list(r.permutation(n_items))], [0]

            def nxt():
                out = []
                while len(out) < batch:
                    if pos[0] >= len(order[0]):
                        order[0] = list(r.permutation(n_items))
                        pos[0] = 0
                    out.append(order[0][pos[0]])
                    pos[0] += 1
                return np.array(out)
            return nxt

        next_real = make_cycler(len(d_audio), args.batch_size, drng)
    bs = args.batch_size
    # Seed the ranking with the PRE-TRAINING recall, so a checkpoint
    # has to beat the base model to be saved -- not merely survive the guards.
    # The guards are tolerances (--bal_tol etc.), so a checkpoint slightly worse
    # than baseline on the objective still passes them; starting from None meant
    # the first such checkpoint was written to --out unconditionally and the run
    # could ship a model worse than the one it started from.
    # F, not recall: the peak threshold is re-tuned per checkpoint (see
    # evaluate_real) and recall alone is maximised by thresholding at zero, so
    # ranking on recall would reward whichever checkpoint the sweep happened to
    # place at the loosest operating point. F is what the sweep optimises, so
    # selection and threshold choice agree.
    rank_key = 'f'
    state = dict(best_rank=baseline[rank_key] if baseline else None,
                 saved=None)
    history = []

    def assess(tag, label):
        """Checkpoint + evaluate + guard the CURRENT weights, and update the
        best-so-far. Called at the end of every epoch and, when --eval_every is
        set, part-way through one as well.

        Sub-epoch evaluation exists because the useful movement and the
        real-audio drift do not happen on the same timescale: the validation
        metrics can converge while the salience is already collapsing inside the
        first epoch, so epoch granularity cannot show where the two separate. `tag` names the checkpoint file, `label` opens the log line."""
        msg = label

        # Keep every checkpoint so it can be chosen AFTER the run, from real
        # audio, instead of only by the in-loop metric. ~5 MB each.
        if args.save_every_epoch:
            ck_path = args.out[:-len('.weights.h5')] + '_%s.weights.h5' % tag
            model.save_weights(ck_path)
            msg += "  [-> %s]" % os.path.basename(ck_path)

        drift = drift_stats(base_real,
                            [predict_salience(model, path) for path in real_feat]) \
            if base_real else {}
        if drift:
            print("  " + format_drift(drift))

        if args.valid_dir:
            inv = evaluate_real(model, args.valid_dir, args.thresh, loss_fn,
                                n_chords=args.TEST)
            if inv is not None:
                inv.update(drift)
                history.append(dict(tag=tag, **inv))
                msg += "  | " + format_metrics_real(inv)
                if drift:
                    msg += "  | " + format_drift(drift)
                # Rank checkpoints on F, and refuse any that pays for it by
                # regressing recall or precision against the baseline.
                failed = guard_failures_real(inv, baseline, args)
                if failed:
                    msg += "  [rejected: %s]" % ", ".join(failed)
                elif state['best_rank'] is None or inv[rank_key] > state['best_rank']:
                    improved = (baseline is None or
                                inv[rank_key] - baseline[rank_key])
                    state['best_rank'] = inv[rank_key]
                    state['saved'] = tag
                    model.save_weights(args.out)
                    msg += ("  [saved best%s]" % ('' if baseline is None
                                                  else ', %+.3f vs baseline' % improved))
        else:
            model.save_weights(args.out)
        print(msg)

    step = 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(win_files))
        losses = []
        since_eval = []
        n_batches = (len(order) + bs - 1) // bs
        for bi, b in enumerate(range(0, len(order), bs)):
            batch = [np.load(win_files[i]) for i in order[b:b+bs]]
            # (batch, AUDIO_N_SAMPLES, 1): the model takes raw audio and
            # computes its own CQT, so there is nothing else to feed it.
            x = tf.convert_to_tensor(
                np.stack([d['audio'] for d in batch])[..., np.newaxis], tf.float32)
            y = tf.convert_to_tensor(np.stack([d['tgt'] for d in batch]), tf.float32)
            # Every window prepare_real_chords.py writes carries a per-frame mask.
            ymask = tf.convert_to_tensor(
                np.stack([d['mask'] for d in batch]), tf.float32)
            if next_real is not None:
                idx = next_real()
                ls, lr = train_step(
                    x, y,
                    tf.convert_to_tensor(d_audio[idx], tf.float32),
                    tf.convert_to_tensor(d_teacher[idx], tf.float32),
                    ymask)
            else:
                ls, lr = train_step(x, y, ymask=ymask)
            losses.append((float(ls), float(lr)))
            since_eval.append(losses[-1])
            step += 1
            print("\r  batch %d/%d  loss=%.4f%s"
                  % (bi + 1, n_batches, losses[-1][0],
                     '  distill=%.4f' % losses[-1][1] if next_real else ''),
                  end='', flush=True)
            # Mid-epoch checkpoint. Skipped on the final batch of an epoch,
            # where it would duplicate the end-of-epoch one below.
            if args.eval_every and step % args.eval_every == 0 and bi + 1 < n_batches:
                print()
                assess('e%02d_s%06d' % (epoch + 1, step),
                       "  step %d (epoch %d, batch %d/%d)  %s"
                       % (step, epoch + 1, bi + 1, n_batches,
                          format_train_loss(since_eval, next_real is not None)))
                since_eval = []
        print()

        assess('e%02d' % (epoch + 1),
               "epoch %d/%d  %s"
               % (epoch + 1, args.epochs,
                  format_train_loss(losses, next_real is not None)))
    rank_label = 'F'
    if not args.valid_dir:
        print("Saved final weights to %s" % args.out)
    elif state['saved'] is None:
        # Deliberately do NOT fall back to the final-epoch weights: nothing beat
        # the pre-trained model on the objective, so writing anything to --out
        # would ship a regression under a name that implies an improvement. The
        # per-checkpoint files are still on disk if the run is worth salvaging.
        print("No checkpoint beat the baseline %s (%.3f) within "
              "the guards; %s NOT written -- use %s instead."
              % (rank_label, state['best_rank'], args.out, os.path.basename(args.weights)))
    else:
        print("Best (highest %s, guards satisfied) weights saved "
              "to %s -- from %s, %s %.3f vs baseline %.3f"
              % (rank_label, args.out, state['saved'], rank_label, state['best_rank'],
                 baseline[rank_key]))

    if history:
        print("\ncheckpoint summary")
        w = max(len(h['tag']) for h in history)
        fmt = format_metrics_real
        for h in history:
            print("  %-*s | %s" % (w, h['tag'], fmt(h)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train_dir', required=True,
                   help='prepared window directory (DONE + *.npz), as '
                        'prepare_real_chords.py writes')
    p.add_argument('--valid_dir', default=None,
                   help='prepared window directory to validate on, same format '
                        'as --train_dir. Split BY TAKE, never by chord within a '
                        'take.')
    p.add_argument('--weights', default=DEFAULT_MODEL_DIR,
                   help='weights to fine-tune from: a Basic Pitch SavedModel '
                        'directory, or a .weights.h5 from a previous run')
    p.add_argument('--out', default='./nmp_salience_finetuned.weights.h5',
                   help='where to write fine-tuned weights (must end .weights.h5)')
    p.add_argument('--TEST', type=int, default=None,
                   help='smoke-test on the first N chords of every file, for both '
                        'training and validation. The caches are still built in full; '
                        'only their use is subset, so a TEST run and a full run share '
                        'one cache. Chords are randomly generated, so the first N are '
                        'representative.')

    p.add_argument('--no_save_every_epoch', dest='save_every_epoch',
                   action='store_false',
                   help='by default every checkpoint is also written to '
                        '<out>_eNN[_sNNNNNN].weights.h5 (~5 MB each), so it can be '
                        'selected afterwards from real audio rather than only by '
                        'the in-loop metric. This switch turns that off.')
    p.add_argument('--eval_every', type=int, default=0, metavar='STEPS',
                   help='also checkpoint, validate and guard every STEPS optimizer '
                        'steps within an epoch (0 = at epoch boundaries only). The '
                        'recall gain and the real-audio drift can both happen inside '
                        'epoch 1, so epoch granularity cannot show where they '
                        'separate. Each evaluation costs a full pass over the '
                        'validation windows plus the --real_audio files, so set this '
                        'to a fraction of an epoch (e.g. 1/10th of the batch count), '
                        'not to a handful of steps.')

    p.add_argument('--batch_size', type=int, default=10,
                   help='windows per step. Keep small on CPU: the (360,1) distribution '
                        'layer backprop scales with batch*win (batch=1 ~1.6GB at win=50).')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--thresh', type=float, default=None,
                   help='peak threshold for eval. Default: re-tuned on every '
                        'checkpoint (by F) and reported as @thresh in the log. An '
                        'absolute value is not comparable across checkpoints -- '
                        'fine-tuning drops the pretrained background floor, so a '
                        'threshold chosen from the pretrained model understates '
                        'every later epoch. Pass a float to pin it instead.')
    p.add_argument('--seed', type=int, default=0)

    p.add_argument('--strategy', choices=['bn', 'conv', 'full'], default='conv',
                   help="which weights adapt. 'conv' (default) = the two "
                        "convolutions with BatchNorm frozen, so the normalisation "
                        "stays as pretrained rather than drifting onto this handful "
                        "of takes; pair with --l2sp. 'bn' = the two BatchNorms only, "
                        "18 parameters, too few to do much here. 'full' = both, "
                        "which can drift since --l2sp cannot anchor BatchNorm's "
                        "running statistics.")
    p.add_argument('--l2sp', type=float, default=1e-3,
                   help='L2-SP anchor strength for full fine-tuning (0 disables). '
                        'Penalises deviation of conv kernels from pretrained values.')
    p.add_argument('--distill_dir', default=None, metavar='DIR',
                   help='directory of REAL recordings (%s), no annotation needed, '
                        'used as a second loss term: the pre-trained model is run '
                        'over them once and the student is penalised for moving '
                        'away from its output. This is --l2sp measured on OUTPUTS '
                        'instead of weights, and it is the same quantity the '
                        'REAL(n) drift line reports -- moved from the accept/reject '
                        'decision into the gradient. Must be DISJOINT from '
                        '--real_audio, which stays the held-out screen. Diversity '
                        'matters more than duration: 8 recordings x 3 min beats one '
                        '25 min file.')
    p.add_argument('--distill_lambda', type=float, default=1.0,
                   help='weight of the real-audio anchor relative to the supervised '
                        'loss (default 1.0; 0 disables the term but STILL passes '
                        'real audio through the network, which under --strategy bn '
                        'alone stops the BatchNorm running statistics converging '
                        'onto the prepared takes -- a useful ablation)')
    p.add_argument('--distill_gamma', type=float, default=0.0,
                   help='confidence-weight the anchor by teacher_salience**gamma. '
                        '0 (default) = uniform, which also pins the quiet voices '
                        'the teacher misses. >0 pins hard where the teacher is '
                        'confident and leaves the student free where it read near '
                        'zero: "keep what you know, stay free where you were '
                        'unsure". Try 1.0 once a uniform run has given a baseline.')
    p.add_argument('--distill_quiet_cap', type=float, default=0.33,
                   help='max share of the distillation pool made up of near-silent '
                        'windows (teacher peak < %.2f). These are kept rather than '
                        'dropped -- a window whose correct answer is "stay near '
                        'zero" is the cheapest constraint against upward salience '
                        'inflation -- but must not crowd out actual singing.'
                        % _QUIET_MAX)
    p.add_argument('--pos_weight', type=float, default=1.0,
                   help='loss upweight on annotated (voice) target bins (1.0 = off)')
    p.add_argument('--bal_tol', type=float, default=0.03,
                   help='max allowed regression of recall or precision, vs the '
                        'pre-training baseline, when selecting the best epoch')
    p.add_argument('--real_audio', nargs='+', default=None, metavar='WAV',
                   help='one or more REAL recordings (no annotation needed). Each '
                        'epoch their salience maps are compared with the ones the '
                        'pre-trained weights produced for the same files, catching '
                        'the case where fine-tuning recalibrates the model to the '
                        'prepared takes -- which the validation windows cannot see, '
                        'being drawn from the same material. Several short excerpts '
                        'beat one long one: '
                        'a consistent shift across independent material is the '
                        'signal. Check the line after epoch 1 and abort if the '
                        'salience has already collapsed.')
    p.add_argument('--drift_high_tol', type=float, default=0.15,
                   help='reject an epoch whose salience on --real_audio drops more '
                        'than this in the pre-trained model\'s most confident band '
                        '(its top %.1f%%%% of bins), averaged over files -- the '
                        'signature of downward compression (default 0.15). This is '
                        'the ONLY drift gate: the whole-map mean is reported but '
                        'not gated, being dominated by a background floor that '
                        'every run legitimately removes.'
                        % (100 - HIGH_PCT))
    train(p.parse_args())
