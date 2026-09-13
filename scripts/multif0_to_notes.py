"""
Convert a multi-F0 CSV (as predict_and_plot.py --thresh writes) into note
events, so the detections can be read against a score.

Each detected frequency is snapped to the nearest equal-tempered note, and
consecutive frames carrying the same note are merged into one event. Two knobs
control what survives: --gap bridges brief dropouts within a held note, and
--min_frames discards blips too short to be a sung note.

Tuning is measured, not assumed. A choir singing 40 cents flat would otherwise
have every note reported as 40 cents flat, and notes near a semitone boundary
could snap to the wrong one. The median deviation across all detections is taken
as the ensemble's own pitch centre and reported both ways: `cents` is deviation
from A440, `cents_rel` from that centre.

With --scale/--offset the score timeline is emitted alongside the audio one,
using the same mapping prepare_real_chords.py fitted:

    audio_seconds = scale * score_seconds + offset

so `score_onset` can be compared with the MIDI directly. Those values are in the
prep manifest, per take.

--per_frame writes one row per timestamp of the input instead of merged events:
every frame is kept, including silent ones, so the output lines up row for row
with the detections file. Notes within a frame are ordered high to low, the same
order read_blocked_chords uses for score chords.

Usage:
    python scripts/multif0_to_notes.py detections.csv [-o notes.csv]
    python scripts/multif0_to_notes.py detections.csv --scale 0.885 --offset 0.13
    python scripts/multif0_to_notes.py detections.csv --per_frame --with_cents
"""

import argparse
import csv
import pathlib

import numpy as np

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi):
    return "%s%d" % (NAMES[int(midi) % 12], int(midi) // 12 - 1)


def write_per_frame(times, freqs, tuning_cents, path, with_cents):
    """One row per input timestamp: time, then the nearest note per detection."""
    ref = 440.0 * 2.0 ** (tuning_cents / 1200.0)
    with open(path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["time", "notes"])
        for t, fs in zip(times, freqs):
            cells = []
            for f in sorted(fs, reverse=True):
                midi = int(round(69 + 12 * np.log2(f / ref)))
                name = note_name(midi)
                if with_cents:
                    et = 440.0 * 2.0 ** ((midi - 69) / 12.0)
                    name += "%+.0f" % (1200.0 * np.log2(f / et))
                cells.append(name)
            w.writerow([round(float(t), 6)] + cells)


def read_multif0(path):
    times, freqs = [], []
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            times.append(float(parts[0]))
            freqs.append(np.array([float(v) for v in parts[1:] if float(v) > 0]))
    return np.array(times), freqs


def to_notes(times, freqs, tuning_cents, gap, min_frames):
    """Merge per-frame detections into (onset, offset, midi, freqs) events."""
    ref = 440.0 * 2.0 ** (tuning_cents / 1200.0)
    # frame -> {midi: [f0, ...]}, snapped against the ensemble's own reference
    per_frame = []
    for fs in freqs:
        d = {}
        for f in fs:
            midi = int(round(69 + 12 * np.log2(f / ref)))
            d.setdefault(midi, []).append(f)
        per_frame.append(d)

    open_notes, done = {}, []
    for i, d in enumerate(per_frame):
        for midi, fs in d.items():
            if midi in open_notes and i - open_notes[midi]["last"] - 1 <= gap:
                open_notes[midi]["last"] = i
                open_notes[midi]["freqs"].extend(fs)
                open_notes[midi]["n"] += 1
            else:
                if midi in open_notes:
                    done.append(open_notes.pop(midi))
                open_notes[midi] = dict(midi=midi, first=i, last=i, freqs=list(fs), n=1)
        for midi in [m for m, v in open_notes.items() if i - v["last"] - 1 > gap]:
            done.append(open_notes.pop(midi))
    done.extend(open_notes.values())

    hop = times[1] - times[0] if len(times) > 1 else 0.0
    events = []
    for ev in done:
        if ev["n"] < min_frames:
            continue
        f = float(np.median(ev["freqs"]))
        events.append(dict(
            onset=float(times[ev["first"]]),
            offset=float(times[ev["last"]] + hop),
            midi=ev["midi"],
            note=note_name(ev["midi"]),
            freq=f,
            cents=1200.0 * np.log2(f / (440.0 * 2.0 ** ((ev["midi"] - 69) / 12.0))),
            frames=ev["n"],
        ))
    return sorted(events, key=lambda e: (e["onset"], -e["midi"]))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_file", type=pathlib.Path)
    p.add_argument("-o", "--output", type=pathlib.Path, default=None,
                   help="output CSV (default: <input>_notes.csv)")
    p.add_argument("--gap", type=int, default=3,
                   help="frames a note may vanish for and still count as held (default 3, ~35 ms)")
    p.add_argument("--min_frames", type=int, default=6,
                   help="discard events shorter than this many frames (default 6, ~70 ms)")
    p.add_argument("--tuning_cents", default="auto",
                   help="'auto' (default) measures the ensemble's pitch centre from the "
                        "median deviation; or give a number, or 0 for concert pitch")
    p.add_argument("--per_frame", action="store_true",
                   help="one row per input timestamp with the nearest notes, "
                        "instead of merged note events")
    p.add_argument("--with_cents", action="store_true",
                   help="--per_frame only: append each note's deviation from "
                        "equal temperament, e.g. A4-13")
    p.add_argument("--scale", type=float, default=None,
                   help="score->audio time scale from the prep manifest")
    p.add_argument("--offset", type=float, default=None,
                   help="score->audio time offset from the prep manifest")
    args = p.parse_args()

    times, freqs = read_multif0(args.csv_file)
    flat = np.concatenate([f for f in freqs if len(f)]) if any(len(f) for f in freqs) else np.array([])
    if not len(flat):
        raise SystemExit("No detections in %s" % args.csv_file)

    # Deviation of every detection from its own nearest equal-tempered note,
    # wrapped to +-50 cents; the median is the ensemble's pitch centre.
    dev = 1200.0 * np.log2(flat / 440.0) + 6900.0
    measured = float(np.median((dev + 50) % 100 - 50))
    tuning = measured if args.tuning_cents == "auto" else float(args.tuning_cents)
    print("tuning: measured %+.1f cents, using %+.1f" % (measured, tuning))

    if args.per_frame:
        out = args.output or args.csv_file.with_name(args.csv_file.stem + "_frames.csv")
        write_per_frame(times, freqs, tuning, out, args.with_cents)
        print("%d frames -> %s" % (len(times), out))
        return

    events = to_notes(times, freqs, tuning, args.gap, args.min_frames)
    out = args.output or args.csv_file.with_name(args.csv_file.stem + "_notes.csv")

    fields = ["onset", "offset", "duration", "midi", "note", "freq", "cents", "cents_rel", "frames"]
    if args.scale is not None and args.offset is not None:
        fields = ["score_onset", "score_offset"] + fields
    with open(out, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for e in events:
            row = dict(onset=round(e["onset"], 3), offset=round(e["offset"], 3),
                       duration=round(e["offset"] - e["onset"], 3),
                       midi=e["midi"], note=e["note"], freq=round(e["freq"], 2),
                       cents=round(e["cents"], 1),
                       cents_rel=round(e["cents"] - tuning, 1), frames=e["frames"])
            if args.scale is not None and args.offset is not None:
                row["score_onset"] = round((e["onset"] - args.offset) / args.scale, 3)
                row["score_offset"] = round((e["offset"] - args.offset) / args.scale, 3)
            w.writerow(row)
    print("%d note events -> %s" % (len(events), out))


if __name__ == "__main__":
    main()
