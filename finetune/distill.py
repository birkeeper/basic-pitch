"""Distil model3's salience into the Basic Pitch salience head.

Pure regression onto the teacher: soft-target binary cross-entropy between the
student's salience and the model3 salience cached by build_teacher_cache.py,
masked to the frames inside the file and the bins inside model3's range
(time_mask[:, None] & bin_mask[None, :]). No labels, no supervised loss, no
anchoring -- see research/model3_distillation_plan.md.

Input is the cache build_teacher_cache.py writes:

    <cache>/train/*.npz    training recordings
    <cache>/valid/*.npz    held-out songs

The pretrained model is evaluated first, as epoch 0, so every later number reads
against the student before any distillation. Validation reports, over the masked
cells only:

    bce, mse   against the teacher (bce includes the teacher's own entropy, so it
               has a floor above zero)
    corr       per-frame Pearson correlation across the in-band bins
    peak P/R/F peaks along frequency (local maxima above --thresh) in the
               student against the teacher, matched within +-1 bin
    residual   mean (student - teacher) by window-loudness quartile, which is
               where the per-window normalisation mismatch would show

The best checkpoint by validation BCE goes to --out. These are diagnostics; the
decision criterion in the plan is the downstream drift and congruence agreement.

    python finetune/distill.py --cache finetune/data/model3_cache \\
        --strategy full --epochs 20 --out finetune/runs/distill01.weights.h5
"""

import os
import sys
import glob
import argparse

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import finetune as F
from convert_model_to_savedmodel import DEFAULT_MODEL_DIR

_EPS = 1e-7


def load_split(folder, max_windows=None, rng=None):
    """Concatenate every cached recording in `folder`.

    Returns audio (n, AUDIO_N_SAMPLES) f32, teacher (n, 172, 264) f16,
    time_mask (n, 172) bool, bin_mask (264,) bool, song (n,) str."""
    files = sorted(glob.glob(os.path.join(folder, '*.npz')))
    if not files:
        raise SystemExit("No cached recordings in %s -- run build_teacher_cache.py" % folder)
    audio, teacher, tmask, songs, bmask = [], [], [], [], None
    for f in files:
        d = np.load(f)
        audio.append(d['audio'])
        teacher.append(d['teacher'])
        tmask.append(d['time_mask'])
        songs += [str(d['song'])] * len(d['audio'])
        if bmask is not None and not np.array_equal(bmask, d['bin_mask']):
            raise SystemExit("%s has a different bin_mask from the other recordings" % f)
        bmask = d['bin_mask']
    audio, teacher, tmask = (np.concatenate(a, 0) for a in (audio, teacher, tmask))
    songs = np.array(songs)
    if max_windows is not None and max_windows < len(audio):
        keep = np.sort(rng.choice(len(audio), max_windows, replace=False))
        audio, teacher, tmask, songs = audio[keep], teacher[keep], tmask[keep], songs[keep]
    print("%s: %d windows from %d recordings, %d songs, %d in-band bins"
          % (folder, len(audio), len(files), len(set(songs)), bmask.sum()))
    return audio, teacher, tmask, bmask, songs


def masked_bce(target, pred, mask):
    """Soft-target BCE summed over masked cells; returns (sum, n_cells)."""
    p = tf.clip_by_value(pred, _EPS, 1.0 - _EPS)
    ce = -(target * tf.math.log(p) + (1.0 - target) * tf.math.log(1.0 - p))
    return tf.reduce_sum(ce * mask), tf.reduce_sum(mask)


def local_peaks(sal, thresh):
    """Boolean map of local maxima along the last (frequency) axis above thresh."""
    pad = np.pad(sal, [(0, 0)] * (sal.ndim - 1) + [(1, 1)], constant_values=-1.0)
    return (sal >= pad[..., :-2]) & (sal > pad[..., 2:]) & (sal > thresh)


def dilate(peaks, tol=1):
    out = peaks.copy()
    for s in range(1, tol + 1):
        out[..., s:] |= peaks[..., :-s]
        out[..., :-s] |= peaks[..., s:]
    return out


def evaluate(model, data, thresh, batch_size):
    audio, teacher, tmask, bmask, _ = data
    sum_bce = sum_se = n_cells = 0.0
    corr_sum = corr_n = 0.0
    tp_s = n_s = tp_t = n_t = 0.0
    rms = np.sqrt((audio.astype(np.float64) ** 2).mean(1))
    win_err = np.zeros(len(audio))       # signed mean error per window
    bins = np.where(bmask)[0]
    for b in range(0, len(audio), batch_size):
        sl = slice(b, b + batch_size)
        x = tf.convert_to_tensor(audio[sl][..., np.newaxis], tf.float32)
        pred = model(x, training=False).numpy()
        t = teacher[sl].astype(np.float32)
        m = tmask[sl][:, :, None] & bmask[None, None, :]

        bce, n = masked_bce(tf.constant(t), tf.constant(pred), tf.constant(m, tf.float32))
        sum_bce += float(bce)
        n_cells += float(n)
        err = (pred - t) * m
        sum_se += float((err ** 2).sum())
        win_err[sl] = err.sum((1, 2)) / np.maximum(m.sum((1, 2)), 1)

        # per-frame correlation across the in-band bins
        p, q = pred[:, :, bins], t[:, :, bins]
        p = p - p.mean(-1, keepdims=True)
        q = q - q.mean(-1, keepdims=True)
        den = np.sqrt((p ** 2).sum(-1) * (q ** 2).sum(-1))
        ok = tmask[sl] & (den > 1e-8)
        corr_sum += float(((p * q).sum(-1)[ok] / den[ok]).sum())
        corr_n += float(ok.sum())

        # peak agreement, in-band frames only
        ps, pt = local_peaks(pred, thresh) & m, local_peaks(t, thresh) & m
        tp_s += float((ps & dilate(pt)).sum())
        n_s += float(ps.sum())
        tp_t += float((pt & dilate(ps)).sum())
        n_t += float(pt.sum())

    prec = tp_s / max(n_s, 1.0)
    rec = tp_t / max(n_t, 1.0)
    q = np.quantile(rms, [0.25, 0.5, 0.75])
    quart = np.digitize(rms, q)
    resid = [float(win_err[quart == i].mean()) if (quart == i).any() else float('nan')
             for i in range(4)]
    return dict(bce=sum_bce / n_cells, mse=sum_se / n_cells,
                corr=corr_sum / max(corr_n, 1.0), prec=prec, rec=rec,
                f=2 * prec * rec / max(prec + rec, 1e-9), resid=resid)


def format_metrics(m):
    return ("val bce=%.4f mse=%.5f corr=%.3f | peaks P=%.3f R=%.3f F=%.3f | "
            "resid by loudness quartile (quiet->loud) %s"
            % (m['bce'], m['mse'], m['corr'], m['prec'], m['rec'], m['f'],
               ' '.join('%+.4f' % r for r in m['resid'])))


def train(args):
    if not args.out.endswith('.weights.h5'):
        args.out = (args.out[:-3] if args.out.endswith('.h5') else args.out) + '.weights.h5'
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    log_path = F.setup_logging(args.out, args)
    print("Output weights: %s\nLogging to %s" % (args.out, log_path))

    rng = np.random.RandomState(args.seed)
    tf.random.set_seed(args.seed)
    tr = load_split(os.path.join(args.cache, 'train'), args.max_windows, rng)
    va = load_split(os.path.join(args.cache, 'valid'), args.max_windows, rng)
    if set(tr[4]) & set(va[4]):
        raise SystemExit("songs in both splits: %s" % sorted(set(tr[4]) & set(va[4])))
    audio, teacher, tmask, bmask, _ = tr
    bmask_f = tf.constant(bmask, tf.float32)

    model = F.build_model(args.weights, args.strategy)
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    anchors = []
    if args.l2sp > 0:
        anchors = [(v, tf.constant(v.numpy()))
                   for v in model.trainable_variables if 'kernel' in v.name]
        print("L2-SP anchoring %d kernels (lambda=%g)" % (len(anchors), args.l2sp))

    @tf.function
    def train_step(x, y, tm):
        mask = tm[:, :, None] * bmask_f[None, None, :]
        with tf.GradientTape() as tape:
            s, n = masked_bce(y, model(x, training=True), mask)
            loss = s / tf.maximum(n, 1.0)
            for v, v0 in anchors:
                loss += args.l2sp * tf.reduce_sum(tf.square(v - v0))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    base = evaluate(model, va, args.thresh, args.batch_size)
    print("epoch 0 (pretrained)  " + format_metrics(base))
    best = base['bce']
    saved = False

    bs = args.batch_size
    for epoch in range(args.epochs):
        order = rng.permutation(len(audio))
        losses = []
        n_batches = (len(order) + bs - 1) // bs
        for bi, b in enumerate(range(0, len(order), bs)):
            idx = np.sort(order[b:b + bs])
            x = tf.convert_to_tensor(audio[idx][..., np.newaxis], tf.float32)
            y = tf.convert_to_tensor(teacher[idx], tf.float32)
            tm = tf.convert_to_tensor(tmask[idx], tf.float32)
            losses.append(float(train_step(x, y, tm)))
            print("\r  batch %d/%d  loss=%.4f" % (bi + 1, n_batches, losses[-1]),
                  end='', flush=True)
        print()
        m = evaluate(model, va, args.thresh, bs)
        msg = "epoch %d/%d  train_loss=%.4f  %s" % (
            epoch + 1, args.epochs, float(np.mean(losses)), format_metrics(m))
        if args.save_every_epoch:
            ck = args.out[:-len('.weights.h5')] + '_e%02d.weights.h5' % (epoch + 1)
            model.save_weights(ck)
            msg += "  [-> %s]" % os.path.basename(ck)
        if m['bce'] < best:
            best = m['bce']
            saved = True
            model.save_weights(args.out)
            msg += "  [saved best, bce %+.4f vs pretrained]" % (m['bce'] - base['bce'])
        print(msg)

    if saved:
        print("Best (lowest val bce %.4f, pretrained %.4f) weights saved to %s"
              % (best, base['bce'], args.out))
    else:
        print("No epoch beat the pretrained val bce (%.4f); %s NOT written."
              % (base['bce'], args.out))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--cache', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   'data', 'model3_cache'),
                   help='cache written by build_teacher_cache.py (train/ and valid/)')
    p.add_argument('--weights', default=DEFAULT_MODEL_DIR,
                   help='Basic Pitch SavedModel directory or a .weights.h5 to continue from')
    p.add_argument('--out', default='./nmp_salience_distilled.weights.h5')
    p.add_argument('--strategy', choices=['bn', 'conv', 'full'], default='full')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--l2sp', type=float, default=0.0,
                   help='L2-SP strength toward the pretrained kernels; off by default, '
                        'use only if training needs stabilising')
    p.add_argument('--thresh', type=float, default=0.3,
                   help='peak threshold for the peak-agreement metric')
    p.add_argument('--max_windows', type=int, default=None,
                   help='random subset per split, for a smoke test')
    p.add_argument('--no_save_every_epoch', dest='save_every_epoch', action='store_false')
    p.add_argument('--seed', type=int, default=0)
    train(p.parse_args())
