"""
Run basic-pitch's salience head on an input audio file and plot the pitch
salience (contour) map with librosa, in a style comparable to the
multif0-estimation-polyvocals salience plots.

--weights selects which salience head to run, so the pretrained model and a
fine-tune can be plotted from the same code path and the images compared.
--thresh sets the peak-picking threshold: detected F0s are overlaid on the plot
and written alongside as a multi-F0 CSV, which makes the effect of the threshold
directly visible -- e.g. whether a quiet tenor's peaks survive it.

Note that the right threshold is model-dependent. Fine-tuning removes the
pretrained background floor (Basic Pitch trained with label smoothing, so its
background sits near 0.10), which moves the useful operating point, so a
threshold carried over from one model understates the other.

Usage:
    python scripts/predict_and_plot.py path/to/audio.wav [--output-dir DIR]
    python scripts/predict_and_plot.py path/to/audio.wav --weights ft.weights.h5 --thresh 0.2
"""

import argparse
import csv
import os
import pathlib
import sys

import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "finetune"))

import bp_grid
from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
from basic_pitch.constants import AUDIO_SAMPLE_RATE, FREQ_BINS_CONTOURS
from basic_pitch.inference import predict, save_note_events
from convert_model_to_savedmodel import (
    DEFAULT_MODEL_DIR,
    build_salience_model,
    load_pretrained_weights,
    predict_salience,
)

# The bundled TensorFlow SavedModel was serialized with an older Keras
# optimizer API and fails to load under TensorFlow>=2.16 (Keras 3). The
# TFLite model loads fine via tf.lite.Interpreter, so use that instead.
MODEL_PATH = build_icassp_2022_model_path(FilenameSuffix.tflite)

FREQ_CEILING_HZ = 2048


def plot_salience(contour, save_path, title, est=None):
    """Plot the pitch salience (contour) map, optionally with detected F0 on top.

    contour has shape (n_times, n_freq_bins) on the non-uniform
    FREQ_BINS_CONTOURS axis, so the bin centre frequencies go to librosa's
    specshow directly instead of letting it assume a log/cqt-spaced axis.
    """
    freq_mask = FREQ_BINS_CONTOURS <= FREQ_CEILING_HZ

    plt.figure(figsize=(15, 7))
    librosa.display.specshow(
        contour[:, freq_mask].T,
        x_axis="time",
        y_coords=FREQ_BINS_CONTOURS[freq_mask],
        y_axis="cqt_hz",
        hop_length=256,
        sr=AUDIO_SAMPLE_RATE,
        cmap="inferno",
    )
    plt.colorbar(label="Activation")
    if est is not None:
        times, freqs = est
        xs = [t for t, fs in zip(times, freqs) for f in np.atleast_1d(fs) if f > 0]
        ys = [f for fs in freqs for f in np.atleast_1d(fs) if f > 0]
        if xs:
            plt.scatter(xs, ys, s=1, c="cyan", marker=".", linewidths=0, label="detected F0")
            plt.legend(loc="upper right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_multif0(times, freqs, path):
    with open(path, "w") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for t, fs in zip(times, freqs):
            writer.writerow([t] + list(np.atleast_1d(fs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio_file", type=pathlib.Path, help="Path to the input audio file.")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Directory to write the outputs to. Defaults to the audio file's directory.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="A .weights.h5 from finetune.py to run instead of the bundled model. "
             "No notes CSV is written for one: a fine-tune is the contour path "
             "only, with no note or onset branch to derive events from.",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=None,
        help="Peak-picking threshold. Overlays the detected F0s on the plot and "
             "writes them as a multi-F0 CSV. Model-dependent -- see the module "
             "docstring -- so re-check it rather than reusing one across models.",
    )
    args = parser.parse_args()

    audio_path = args.audio_file
    output_dir = args.output_dir or audio_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = pathlib.Path(args.weights).name.split(".")[0] if args.weights else "basic_pitch"

    if args.weights:
        model = build_salience_model()
        model.load_weights(args.weights)
    else:
        model = build_salience_model()
        load_pretrained_weights(model, DEFAULT_MODEL_DIR)
    contour = predict_salience(model, str(audio_path))
    print("salience: mean %.4f  max %.3f  (%s)" % (contour.mean(), contour.max(), tag))

    est = None
    if args.thresh is not None:
        times, freqs = bp_grid.salience_to_multif0(contour, args.thresh)
        est = (times, freqs)
        n = sum(len(f) for f in freqs)
        mf0_path = output_dir / ("%s_%s_multif0.csv" % (audio_path.stem, tag))
        save_multif0(times, freqs, mf0_path)
        print("Saved %d detections at thresh=%.3f to %s" % (n, args.thresh, mf0_path))

    salience_path = output_dir / ("%s_%s_salience.png" % (audio_path.stem, tag))
    title = "Pitch salience: %s  --  %s" % (audio_path.name, tag)
    if args.thresh is not None:
        title += "  (thresh %.3f)" % args.thresh
    plot_salience(contour, salience_path, title, est)
    print("Saved salience plot to %s" % salience_path)

    if not args.weights:
        _out, _midi, note_events = predict(audio_path, model_or_model_path=MODEL_PATH)
        notes_csv_path = output_dir / ("%s_basic_pitch_notes.csv" % audio_path.stem)
        save_note_events(note_events, notes_csv_path)
        print("Saved estimated notes to %s" % notes_csv_path)


if __name__ == "__main__":
    main()
