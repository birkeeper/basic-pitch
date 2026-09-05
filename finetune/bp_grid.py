"""Time/frequency grids and salience targets on Basic Pitch's contour grid.

This replaces the handful of grid functions the multif0 pipeline took from its
`utils` module (`get_freq_grid`, `get_time_grid`, `grid_to_bins`,
`create_annotation_target`) with equivalents on Basic Pitch's grid. The HCQT
half of that module has no counterpart here and is deliberately absent: Basic
Pitch computes its own CQT inside the graph, so there are no features to
precompute, no pumpp dependency, and nothing to cache.

Three things differ from the multif0 versions, and all three matter:

TRANSPOSED
    multif0 salience is (freq, time); Basic Pitch contours are (time, freq).
    Every array here is (n_frames, n_bins). Getting this wrong produces arrays
    that broadcast silently rather than raising.

COARSER, WIDER GRID
    264 bins at 3 per semitone -- 33.33 cents -- from 27.5 Hz up to ~4.4 kHz,
    against multif0's 360 bins at 20 cents from 32.7 Hz to ~2.1 kHz. The range
    is a superset, so no annotation is lost, but the spacing is 1.67x coarser.
    Blur widths quoted in *bins* do not carry over between the two; quote them
    in cents (see `create_annotation_target`).

NEAREST BIN IN LOG FREQUENCY
    multif0's `grid_to_bins` splits bins at the arithmetic midpoint of adjacent
    grid frequencies. Basic Pitch's own training labels come from mirdata's
    `to_sparse_index`, which takes the nearest bin in *log* frequency, i.e. the
    geometric midpoint. On a log-spaced grid that is the principled split and
    it is what the pretrained weights were fitted against, so it is what is used
    here. The two disagree by under a tenth of a cent -- irrelevant in itself,
    but free to get right, and it keeps the labels consistent with pretraining.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from basic_pitch.constants import (
    AUDIO_SAMPLE_RATE,
    CONTOURS_BINS_PER_SEMITONE,
    FFT_HOP,
    FREQ_BINS_CONTOURS,
    N_FREQ_BINS_CONTOURS,
)

# Cents per contour bin: 100 / 3 = 33.333...
CENTS_PER_BIN = 100.0 / CONTOURS_BINS_PER_SEMITONE

# Seconds per frame. Deliberately NOT basic_pitch.constants.ANNOTATION_HOP:
# that is 1 / ANNOTATIONS_FPS with ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP,
# an integer division that rounds 86.13 fps down to 86 and so puts frame times
# ~1.8 us/frame fast -- 3 ms across one window, and a third of a second across a
# four minute take. The CQT hops by FFT_HOP samples, so this is the real grid,
# and it is the grid the multif0 f0 CSVs were written on.
FRAME_HOP_SECONDS = float(FFT_HOP) / AUDIO_SAMPLE_RATE


def get_freq_grid():
    """Centre frequency of each contour bin, in Hz. Shape (264,)."""
    return FREQ_BINS_CONTOURS


def get_time_grid(n_time_frames):
    """Start time of each frame, in seconds. Shape (n_time_frames,)."""
    return np.arange(n_time_frames) * FRAME_HOP_SECONDS


def freq_to_bin(frequencies):
    """Nearest contour bin for each frequency, by log distance.

    Returns -1 for frequencies outside the grid by more than half a bin, so
    out-of-range annotations can be dropped rather than piling up on the edge
    bins. Frequencies <= 0 (unvoiced markers) also return -1.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    out = np.full(frequencies.shape, -1, dtype=np.int64)

    voiced = frequencies > 0
    if not np.any(voiced):
        return out

    log_grid = np.log(FREQ_BINS_CONTOURS)
    log_f = np.log(frequencies[voiced])

    idx = np.searchsorted(log_grid, log_f)
    idx = np.clip(idx, 1, len(log_grid) - 1)
    # Pick whichever neighbour is nearer in log space (== geometric midpoint).
    left, right = log_grid[idx - 1], log_grid[idx]
    nearest = np.where(log_f - left <= right - log_f, idx - 1, idx)

    # Drop anything more than half a bin outside the grid.
    half_bin = 0.5 * (CENTS_PER_BIN / 1200.0) * np.log(2.0)
    in_range = np.abs(log_f - log_grid[nearest]) <= half_bin + 1e-12
    nearest = np.where(in_range, nearest, -1)

    out[voiced] = nearest
    return out


def create_annotation_target(
    annotation_times,
    annotation_freqs,
    n_time_frames,
    blur_cents=25.0,
    hard=False,
):
    """Salience target on the contour grid. Returns (n_time_frames, 264) float32.

    `annotation_times` and `annotation_freqs` are flat, parallel arrays: one
    entry per (frame, voice), so a four-part chord contributes four entries per
    frame. This is the same convention the multif0 pipeline used.

    Two target conventions, and the choice is not cosmetic:

    hard=True reproduces how Basic Pitch itself was trained -- the nearest bin
    gets a 1, everything else a 0, with `label_smoothing` in the loss doing the
    softening. Use it to keep the pretrained sigmoid's calibration intact.

    hard=False (default) writes a Gaussian ridge centred on the true frequency
    and sampled at the bin centres. This is what makes sub-bin position a
    *supervised* quantity: with a single-bin target, a pitch at a bin centre and
    one a third of a bin away get identical labels, so nothing teaches the ridge
    to lean. Note this differs from multif0's target, which snaps to a bin and
    blurs afterwards -- that produces a ridge symmetric about the bin centre and
    encodes no sub-bin offset at all. If you train against a ridge, drop label
    smoothing, or it squashes the very shoulders you are teaching.

    `blur_cents` is quoted in cents, not bins, precisely because the two grids
    disagree: multif0's sigma of 1 bin was 20 cents, which is 0.6 bins here.

    The 25 cent default (0.75 bins) was measured, not inherited. With clean
    targets a log-parabola fit recovers a Gaussian centre exactly at any width,
    so the width only buys noise robustness, and sweeping simulated salience
    noise puts the optimum flatly at 20-25 cents: narrower collapses (at 15
    cents the ridge is nearly one bin and the three-point fit has nothing to
    work with), wider flattens the curvature the fit depends on. Voice
    separation agrees -- two voices a semitone apart (3 bins) leave a valley at
    41% of peak here, against 61% at one sigma per bin.
    """
    annotation_times = np.asarray(annotation_times, dtype=np.float64)
    annotation_freqs = np.asarray(annotation_freqs, dtype=np.float64)
    if annotation_times.shape != annotation_freqs.shape:
        raise ValueError(
            "times and freqs must be parallel, got {} and {}".format(
                annotation_times.shape, annotation_freqs.shape
            )
        )

    target = np.zeros((n_time_frames, N_FREQ_BINS_CONTOURS), dtype=np.float64)

    time_idx = np.round(annotation_times / FRAME_HOP_SECONDS).astype(np.int64)
    in_time = (time_idx >= 0) & (time_idx < n_time_frames)

    if hard:
        freq_idx = freq_to_bin(annotation_freqs)
        keep = in_time & (freq_idx >= 0)
        target[time_idx[keep], freq_idx[keep]] = 1.0
        return target.astype(np.float32)

    # Ridge centred on the TRUE frequency, not on the nearest bin.
    #
    # This is the whole point of a soft target. Snapping to a bin and then
    # blurring -- which is what multif0's create_annotation_target does -- gives
    # a ridge symmetric about the bin centre, so the sub-bin offset is gone
    # before the blur is applied and no blur width can bring it back. Sampling a
    # continuous Gaussian at the bin centres instead makes the imbalance between
    # neighbouring bins a direct, supervised function of where inside the bin the
    # pitch actually sits.
    log_grid = np.log(FREQ_BINS_CONTOURS)
    sigma_log = (blur_cents / 1200.0) * np.log(2.0)
    # Beyond ~4 sigma the ridge is numerically zero; keep the window tight so a
    # dense chord does not cost a full (voices x 264) outer product.
    half_width = int(np.ceil(4.0 * blur_cents / CENTS_PER_BIN))

    # Same in-range rule as freq_to_bin, so the two conventions agree about
    # which annotations exist at all: without it a pitch below the grid would
    # still deposit the tail of its ridge on bin 0.
    half_bin_log = 0.5 * (CENTS_PER_BIN / 1200.0) * np.log(2.0)
    in_grid = (np.log(np.maximum(annotation_freqs, 1e-12)) >= log_grid[0] - half_bin_log) & (
        np.log(np.maximum(annotation_freqs, 1e-12)) <= log_grid[-1] + half_bin_log
    )

    voiced = in_time & (annotation_freqs > 0) & in_grid
    for t, f in zip(time_idx[voiced], annotation_freqs[voiced]):
        centre = np.searchsorted(log_grid, np.log(f))
        lo = max(0, centre - half_width - 1)
        hi = min(N_FREQ_BINS_CONTOURS, centre + half_width + 1)
        if lo >= hi:
            continue
        ridge = np.exp(-0.5 * ((log_grid[lo:hi] - np.log(f)) / sigma_log) ** 2)

        # Rescale so this voice's tallest sampled bin reads 1.
        #
        # Without it the peak height would encode tuning: a pitch sitting on a
        # bin centre catches the Gaussian's apex and trains toward 1, one sitting
        # between bins toward ~0.8, and the model would learn to report a voice
        # as less salient for being off-grid -- confusing how present a voice is
        # with how close it happens to sit to a bin. Sub-bin recovery is
        # untouched: scaling all bins by a constant shifts a log-parabola
        # vertically without moving its vertex.
        peak = ridge.max()
        if peak > 0:
            ridge = ridge / peak

        # Combine voices with max, not sum: overlapping ridges then stay bounded
        # by 1 without the clipping that would flatten the shoulders carrying the
        # sub-bin information.
        np.maximum(target[t, lo:hi], ridge, out=target[t, lo:hi])

    return target.astype(np.float32)


def read_f0_csv(path, cents_offset=0.0):
    """Read a multif0 CSV into flat (times, freqs) arrays.

    Each line is `time<TAB>f0 [<TAB>f0 ...]` with a variable number of
    frequencies -- one per sounding voice in that frame. `cents_offset` shifts
    every label frequency, for takes whose measured tuning is off concert pitch;
    it is applied to the LABELS, never to the audio.
    """
    times, freqs = [], []
    scale = 2.0 ** (cents_offset / 1200.0)
    with open(path, "r") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            t = float(parts[0])
            for value in parts[1:]:
                f = float(value)
                if f > 0:
                    times.append(t)
                    freqs.append(f * scale)
    return np.array(times), np.array(freqs)


def salience_to_multif0(salience, threshold=0.5):
    """Peak-pick a (n_frames, 264) salience map into per-frame f0 lists.

    The Basic Pitch counterpart of `utils_train.pitch_activations_to_mf0`,
    transposed: peaks are found along frequency, which is now axis 1.
    """
    import scipy.signal

    times = get_time_grid(salience.shape[0])
    freqs = get_freq_grid()

    peaks = scipy.signal.argrelmax(salience, axis=1)
    picked = np.zeros_like(salience)
    picked[peaks] = salience[peaks]

    frame_idx, bin_idx = np.where(picked >= threshold)
    est_freqs = [[] for _ in range(len(times))]
    for t, f in zip(frame_idx, bin_idx):
        est_freqs[t].append(freqs[f])
    return times, [np.array(lst) for lst in est_freqs]


def target_to_multif0(target, threshold=0.5):
    """Per-frame reference f0 lists from a ridge target: one entry per VOICE.

    Not `target > threshold`. A ridge normalised to peak 1 puts two bins over
    0.5 whenever the pitch sits more than about a fifth of a bin off centre --
    and both bins read exactly 1.0 when it sits halfway between them -- so
    thresholding counts each voice once or twice depending on its tuning. Used
    as a reference for multipitch scoring that inflates the voice count (8.8 per
    frame on four-to-six part chords here) and caps recall near 0.5 for a model
    that is answering perfectly.

    Instead each contiguous run of bins above `threshold` is taken as one voice
    and reduced to a single frequency by its centre of mass in log frequency,
    which recovers the sub-bin position the ridge was built to encode.
    """
    log_grid = np.log(FREQ_BINS_CONTOURS)
    out = []
    for row in np.asarray(target):
        hits = np.flatnonzero(row > threshold)
        freqs = []
        if len(hits):
            # split where the bin index jumps: each run is one voice
            for run in np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1):
                w = row[run]
                freqs.append(float(np.exp(np.average(log_grid[run], weights=w))))
        out.append(np.array(freqs))
    return out
