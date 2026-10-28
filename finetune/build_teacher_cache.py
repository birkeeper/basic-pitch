"""Cache (audio window, model3 teacher window, mask) triples for distillation.

Reads every recording that has a model3 salience npz beside it from

    data/model3_train/    <song>_take##.wav + <song>_take##_model3_exp3multif0_salience.npz
    data/model3_valid/    same, different songs

and writes one npz per recording to data/model3_cache/<split>/, holding

    audio      (n, AUDIO_N_SAMPLES) float32   the windows inference scans with
    teacher    (n, 172, 264) float16          model3 salience on the bp grid
    time_mask  (n, 172) bool                  frame has a teacher (inside the file)
    bin_mask   (264,) bool                    bin lies inside model3's range
    song       str

The loss mask for a window is time_mask[:, None] & bin_mask[None, :]. Audio with
no teacher npz is skipped with a warning, so files can be added in batches.
Takes of one song share a split: a song in both folders is a hard error, since
the validation numbers would then measure memorisation.

    python build_teacher_cache.py [--data data] [--force]
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import teacher as T
from basic_pitch import inference

SPLITS = ('train', 'valid')


def audio_windows(path):
    """Same windows as finetune.audio_windows, without importing TensorFlow."""
    wins, orig_len = [], None
    for w, _t, orig_len in inference.get_audio_input(path, T.OVERLAP_LEN, T.HOP_SIZE):
        wins.append(w)
    return np.concatenate(wins, axis=0)[..., 0], orig_len


def build_one(audio_path, teacher_npz, out_path):
    wins, orig_len = audio_windows(audio_path)
    assert len(wins) == T.n_windows_for(orig_len), (len(wins), T.n_windows_for(orig_len))
    tw, tmask, bmask = T.teacher_windows(teacher_npz, len(wins))
    np.savez_compressed(
        out_path,
        audio=wins.astype(np.float32),
        teacher=tw.astype(np.float16),
        time_mask=tmask,
        bin_mask=bmask,
        song=T.song_id(audio_path),
    )
    return len(wins), float(tmask.mean())


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default=os.path.join(here, 'data'))
    ap.add_argument('--force', action='store_true', help="rebuild existing cache files")
    a = ap.parse_args()

    pairs_by_split, split_of_song = {}, {}
    for split in SPLITS:
        src = os.path.join(a.data, 'model3_' + split)
        if not os.path.isdir(src):
            print("%s: no folder %s, skipped" % (split, src))
            continue
        pairs, missing = T.find_pairs(src)
        for m in missing:
            print("  WARNING no teacher npz for %s, skipped" % os.path.basename(m))
        pairs_by_split[split] = pairs
        for audio, _ in pairs:
            split_of_song.setdefault(T.song_id(audio), set()).add(split)

    clash = sorted(s for s, sp in split_of_song.items() if len(sp) > 1)
    if clash:
        raise SystemExit("song(s) in both train and valid: %s -- all takes of a song "
                         "must share a split" % ', '.join(clash))

    for split, pairs in pairs_by_split.items():
        out_dir = os.path.join(a.data, 'model3_cache', split)
        os.makedirs(out_dir, exist_ok=True)
        n_files = n_win = 0
        for audio, teach in pairs:
            out = os.path.join(out_dir, os.path.splitext(os.path.basename(audio))[0] + '.npz')
            if os.path.exists(out) and not a.force:
                print("  %-6s %-40s cached" % (split, os.path.basename(audio)[:40]))
                continue
            n, cov = build_one(audio, teach, out)
            n_files += 1
            n_win += n
            print("  %-6s %-40s %4d windows, %.0f%% frames have a teacher"
                  % (split, os.path.basename(audio)[:40], n, 100 * cov))
        print("%s: %d built (%d windows), %d song(s)"
              % (split, n_files, n_win, len({T.song_id(p[0]) for p in pairs})))


if __name__ == '__main__':
    main()
