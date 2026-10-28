"""Alignment self-test for the model3 teacher windows (teacher.py).

Two checks, both runnable before any training:

1. GEOMETRY (exact). Window a ramp signal exactly as inference.get_audio_input
   does and confirm window k, sample i holds original sample
   k * HOP_SIZE - OVERLAP_LEN / 2 + i. This is the arithmetic window_frame_positions
   rests on.

2. LAG (empirical). The pretrained Basic Pitch is run on the file's windows and
   compared with the teacher sampled at the same frames, over a grid of
   fractional shifts. If the mapping is right the best shift is ~0 in every
   window and does not trend with the window index; a trend means the time
   axis is wrong (the stitched-output stretch described in teacher.py would show
   up here as a slope), and a constant offset means model3 and Basic Pitch frame
   conventions differ by that much. The shift is estimated between two
   different models, so expect a broad, noisy optimum -- the point is the mean
   and the slope, not the per-window value.

    python check_alignment.py ../finetune/data/alles.flac
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import teacher as T
from basic_pitch import inference
from basic_pitch.constants import AUDIO_N_SAMPLES, ANNOT_N_FRAMES


def check_geometry():
    n = 3 * T.HOP_SIZE + 5000
    ramp = np.arange(n, dtype=np.float32)
    padded = np.concatenate([np.zeros(T.OVERLAP_LEN // 2, np.float32), ramp])
    wins = [w[:, 0] for w, _ in inference.window_audio_file(padded, T.HOP_SIZE)]
    assert len(wins) == T.n_windows_for(n), (len(wins), T.n_windows_for(n))
    for k, w in enumerate(wins):
        want = k * T.HOP_SIZE - T.OVERLAP_LEN // 2
        got = int(w[0]) if want >= 0 else int(w[-want])
        assert got == (want if want >= 0 else 0), (k, got, want)
        assert len(w) == AUDIO_N_SAMPLES
    p = T.window_frame_positions(len(wins))
    assert p.shape == (len(wins), ANNOT_N_FRAMES)
    print("geometry: OK  (%d windows, hop %d samples = %.4f frames, front pad %d)"
          % (len(wins), T.HOP_SIZE, T.HOP_SIZE / 256.0, T.OVERLAP_LEN // 2))


def check_lag(audio_path, teacher_npz, max_shift=12.0, step=0.25, time_fix=True):
    import finetune as F   # imports TF; kept out of the geometry check
    model = F.build_model(F.DEFAULT_MODEL_DIR, 'full')
    wins, orig_len = F.audio_windows(audio_path)
    n = len(wins)
    assert n == T.n_windows_for(orig_len), (n, T.n_windows_for(orig_len))
    bp = model.predict(wins, verbose=0)

    sal, freq = T.load_teacher(teacher_npz)
    if time_fix:
        sal = T.to_true_time(sal)
    grid, bin_mask = T.regrid_freq(sal, freq)
    print("teacher time axis: %s (%d frames, audio %.1f)" % (
        "segment-corrected" if time_fix else "RAW npz", len(grid), orig_len / 256.0))

    shifts = np.arange(-max_shift, max_shift + 1e-9, step)
    corr = np.full((n, len(shifts)), np.nan)
    # Central frames only: the window edges are where bp itself is least reliable.
    sl = slice(N_EDGE, ANNOT_N_FRAMES - N_EDGE)
    for si, s in enumerate(shifts):
        tw, tm = T.sample_windows(grid, n, offset=T.TEACHER_OFFSET_FRAMES + s)
        for k in range(n):
            m = tm[k, sl]
            if m.sum() < 40:
                continue
            a = bp[k, sl][m][:, bin_mask].ravel()
            b = tw[k, sl][m][:, bin_mask].ravel()
            if a.std() < 1e-6 or b.std() < 1e-6:
                continue
            corr[k, si] = np.corrcoef(a, b)[0, 1]

    ok = ~np.isnan(corr).all(1)
    best = shifts[np.nanargmax(np.where(np.isnan(corr), -2, corr), axis=1)][ok]
    k_idx = np.where(ok)[0]
    glob = np.nanmean(corr[ok], axis=0)
    gi = int(np.argmax(glob))
    slope = np.polyfit(k_idx, best, 1)[0] * 100 if len(k_idx) > 2 else float('nan')
    print("lag: %d/%d windows usable" % (ok.sum(), n))
    print("  mean-correlation-optimal global shift : %+.2f frames (r=%.3f; r at 0 = %.3f)"
          % (shifts[gi], glob[gi], glob[int(np.argmin(abs(shifts)))]))
    print("  per-window best shift                 : mean %+.2f  std %.2f  median %+.2f frames"
          % (best.mean(), best.std(), np.median(best)))
    print("  drift with window index               : %+.3f frames per 100 windows" % slope)
    total = slope * ok.sum() / 100.0
    print("  total drift over the file             : %+.2f frames" % total)
    bad = abs(shifts[gi]) > 0.5 or abs(total) > 1.5
    print("lag: %s" % ("WARN -- mapping or frame convention is off, see above" if bad else "OK"))
    return not bad


N_EDGE = 20


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('audio', nargs='?', default=None,
                    help="audio with a *_model3_exp3multif0_salience.npz beside it; "
                         "omit to run the geometry check only")
    ap.add_argument('--raw_time', action='store_true',
                    help="skip the segment-loss correction, to see its effect")
    a = ap.parse_args()
    check_geometry()
    if a.audio:
        t = T.teacher_path_for(a.audio)
        if not os.path.exists(t):
            raise SystemExit("no teacher npz at %s" % t)
        check_lag(a.audio, t, time_fix=not a.raw_time)


if __name__ == '__main__':
    main()
