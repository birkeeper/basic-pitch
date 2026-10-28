"""model3 salience as a distillation teacher for Basic Pitch's contour head.

Maps the whole-file salience npz written by predict_on_audio.py --save_salience
(`salience` (360, T), `freq_grid`, `time_grid`) onto what the student sees: the
overlapping fixed-length windows basic_pitch.inference scans a file with, on
Basic Pitch's 264-bin contour grid.

Frequency
    model3 is 360 bins at 20 cents from 32.7 Hz; Basic Pitch is 264 bins at
    33.33 cents from 27.5 Hz. The two grids coincide at 32.7 Hz, and bins are
    resampled by linear interpolation in log frequency (= cents). The teacher
    covers only the bins between its own lowest and highest frequency, read from
    `freq_grid` rather than hard-coded; every other bin is excluded through
    `bin_mask` and has no target.

Time
    Both models run at sr 22050 / hop 256, so one frame is the same 11.61 ms and
    no resampling is needed. What differs is the window geometry. Window k
    starts at sample k * HOP_SIZE of the front-padded audio, i.e.
    k * HOP_SIZE - OVERLAP_LEN / 2 of the original, and output frame j of that
    window is therefore original-audio frame

        p = (k * HOP_SIZE - OVERLAP_LEN / 2) / FFT_HOP + j

    HOP_SIZE / FFT_HOP = 141.27 is not an integer, so p is fractional and the
    teacher is sampled by linear interpolation along time rather than by
    rounding. Frames with p outside the teacher's extent have no target
    (`time_mask`).

    Note this is NOT the position of that frame in basic_pitch.inference's
    stitched output: unwrap_output keeps 142 frames per window while a window
    advances 141.27, so the stitched axis is stretched by ~0.5%. Training
    windows are therefore sampled from the teacher in true time here.
"""

import os
import re

import numpy as np

from basic_pitch import inference
from basic_pitch.constants import (
    ANNOT_N_FRAMES,
    AUDIO_N_SAMPLES,
    FFT_HOP,
    FREQ_BINS_CONTOURS,
)

N_OLAP = inference.DEFAULT_OVERLAPPING_FRAMES
OVERLAP_LEN = N_OLAP * FFT_HOP
HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LEN

TEACHER_SUFFIX = '_model3_exp3multif0_salience.npz'
AUDIO_EXTS = ('.wav', '.flac', '.ogg')

# Teacher frame spacing must match Basic Pitch's to within this, in seconds,
# before the shared time axis assumption is rejected.
_DT_TOL = 1e-6

# model3's frames sit this many frames later than Basic Pitch's. Measured with
# check_alignment.py as the mean-correlation-optimal shift: +4.0 to +4.5 frames
# on 4 of 5 songs (+5.75 on the fifth, whose first run clipped at +4.0), ~49 ms.
# Teacher windows are sampled at position + TEACHER_OFFSET_FRAMES.
TEACHER_OFFSET_FRAMES = 4.25


def song_id(path):
    """'late_nolowbass_take06.wav' -> 'late_nolowbass'. Takes of one song must
    share a split, so splitting keys on this rather than on the file."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r'_take\d+.*$', '', stem)


def teacher_path_for(audio_path):
    return os.path.splitext(audio_path)[0] + TEACHER_SUFFIX


def load_teacher(path):
    """Whole-file teacher: (salience (T, 360) float32, freq_grid (360,))."""
    d = np.load(path, allow_pickle=True)
    sal = np.asarray(d['salience'], dtype=np.float32).T
    freq = np.asarray(d['freq_grid'], dtype=np.float64)
    t = np.asarray(d['time_grid'], dtype=np.float64)
    if sal.shape != (len(t), len(freq)):
        raise ValueError("%s: salience %s does not match time_grid %d x freq_grid %d"
                         % (path, d['salience'].shape, len(t), len(freq)))
    dt = np.median(np.diff(t))
    if abs(dt - float(FFT_HOP) / 22050) > _DT_TOL:
        raise ValueError("%s: frame spacing %.9f s is not Basic Pitch's %.9f s"
                         % (path, dt, float(FFT_HOP) / 22050))
    if abs(t[0]) > _DT_TOL:
        raise ValueError("%s: time_grid starts at %g, expected 0" % (path, t[0]))
    return sal, freq


# model3's features come from utils.compute_pump_features_segmented: 10 s
# segments with 1 s of left context, each keeping round(220500 / 256) = 861
# frames although it spans 861.33. The npz therefore has fewer frames than the
# audio (15939 against 15944.9 on a 185 s file) and its time_grid, which is just
# t * hop, runs early by 0.33 frame per segment -- 70 ms by the end of a three
# minute take. The audio sample behind teacher frame t = 861 * s + i is
# s * SEG - (CTX - 86 * hop) + i * hop for s > 0 (left_trim = round(CTX / hop)
# = 86), and i * hop for the first segment.
SEG_SAMPLES = 220500
CTX_SAMPLES = 22050
SEG_FRAMES = int(round(SEG_SAMPLES / float(FFT_HOP)))
_LEFT_TRIM = int(round(CTX_SAMPLES / float(FFT_HOP)))


def true_frame_positions(n_frames):
    """Original-audio frame position of each teacher frame, shape (n_frames,)."""
    t = np.arange(n_frames)
    s, i = t // SEG_FRAMES, t % SEG_FRAMES
    start = (s * SEG_SAMPLES - CTX_SAMPLES + _LEFT_TRIM * FFT_HOP) / float(FFT_HOP)
    return np.where(s > 0, start + i, i.astype(np.float64))


def to_true_time(sal):
    """Resample a segment-stitched (T, F) teacher onto the audio's integer
    frame grid, undoing the 0.33 frame/segment loss described above."""
    pos = true_frame_positions(sal.shape[0])
    q = np.arange(int(np.floor(pos[-1])) + 1)
    f = np.interp(q, pos, np.arange(sal.shape[0]))
    i0 = np.minimum(np.floor(f).astype(int), sal.shape[0] - 2)
    w = (f - i0).astype(np.float32)[:, None]
    return sal[i0] * (1 - w) + sal[i0 + 1] * w


def regrid_freq(sal, freq_grid):
    """(T, F_teacher) -> (T, 264) on Basic Pitch's contour grid, plus bin_mask.

    `bin_mask` (264,) is True where the bp bin lies inside the teacher's
    frequency range. Values at masked-out bins are 0 and must be ignored.
    """
    f_bp = np.asarray(FREQ_BINS_CONTOURS, dtype=np.float64)
    lo, hi = freq_grid[0] * (1 - 1e-6), freq_grid[-1] * (1 + 1e-6)
    bin_mask = (f_bp >= lo) & (f_bp <= hi)
    pos = np.interp(np.log(f_bp[bin_mask]), np.log(freq_grid), np.arange(len(freq_grid)))
    i0 = np.clip(np.floor(pos).astype(int), 0, len(freq_grid) - 2)
    w = (pos - i0).astype(np.float32)
    out = np.zeros((sal.shape[0], len(f_bp)), np.float32)
    out[:, bin_mask] = sal[:, i0] * (1 - w) + sal[:, i0 + 1] * w
    return out, bin_mask


def n_windows_for(audio_length):
    """Number of windows inference.get_audio_input yields for `audio_length`
    original samples (front pad of OVERLAP_LEN / 2, hop HOP_SIZE)."""
    return len(range(0, audio_length + OVERLAP_LEN // 2, HOP_SIZE))


def window_frame_positions(n_windows, offset=0.0):
    """(n_windows, ANNOT_N_FRAMES) original-audio frame position of every output
    frame of every window. `offset` shifts all positions, in frames (used by the
    alignment check)."""
    k = np.arange(n_windows, dtype=np.float64)[:, None]
    j = np.arange(ANNOT_N_FRAMES, dtype=np.float64)[None, :]
    return (k * HOP_SIZE - OVERLAP_LEN / 2.0) / FFT_HOP + j + offset


def sample_windows(grid_sal, n_windows, offset=0.0):
    """Sample a (T, 264) map at every window frame by linear interpolation.

    Returns (windows (n, ANNOT_N_FRAMES, 264) float32, time_mask (n, ANNOT_N_FRAMES)
    bool). time_mask is False where the position falls outside [0, T-1].
    """
    T = grid_sal.shape[0]
    p = window_frame_positions(n_windows, offset)
    valid = (p >= 0) & (p <= T - 1)
    pc = np.clip(p, 0, T - 1)
    i0 = np.minimum(np.floor(pc).astype(int), T - 2)
    w = (pc - i0).astype(np.float32)[..., None]
    return grid_sal[i0] * (1 - w) + grid_sal[i0 + 1] * w, valid


def teacher_windows(teacher_npz, n_windows):
    """Teacher targets for one file: (windows, time_mask, bin_mask)."""
    sal, freq = load_teacher(teacher_npz)
    grid, bin_mask = regrid_freq(to_true_time(sal), freq)
    win, tmask = sample_windows(grid, n_windows, offset=TEACHER_OFFSET_FRAMES)
    win[..., ~bin_mask] = 0.0
    return win, tmask, bin_mask


def find_pairs(folder):
    """[(audio_path, teacher_path)] in `folder`, sorted. Audio with no teacher
    npz is returned in the second list so the caller can warn about it."""
    audio = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(AUDIO_EXTS))
    pairs, missing = [], []
    for a in audio:
        t = teacher_path_for(a)
        if os.path.exists(t):
            pairs.append((a, t))
        else:
            missing.append(a)
    return pairs, missing
