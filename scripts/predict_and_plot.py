"""
Run basic-pitch's predict() on an input audio file, save the estimated
notes as CSV, and plot the pitch salience (contour) map with librosa,
in a style comparable to the multif0-estimation-polyvocals salience plots.

Usage:
    python scripts/predict_and_plot.py path/to/audio.wav [--output-dir OUTPUT_DIR]
"""

import argparse
import pathlib

import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP, FREQ_BINS_CONTOURS
from basic_pitch.inference import predict, save_note_events

# The bundled TensorFlow SavedModel was serialized with an older Keras
# optimizer API and fails to load under TensorFlow>=2.16 (Keras 3). The
# TFLite model loads fine via tf.lite.Interpreter, so use that instead.
MODEL_PATH = build_icassp_2022_model_path(FilenameSuffix.tflite)


def plot_salience(contour: np.ndarray, save_path: pathlib.Path, title: str = "Pitch salience (basic-pitch)") -> None:
    """Plot the basic-pitch pitch salience (contour) map.

    contour has shape (n_times, n_freq_bins), with a non-uniform (per-bin)
    frequency axis given by FREQ_BINS_CONTOURS, so we pass those bin center
    frequencies to librosa's specshow directly instead of assuming a
    log/cqt-spaced axis.
    """
    freq_mask = FREQ_BINS_CONTOURS <= 2048

    plt.figure(figsize=(15, 7))
    librosa.display.specshow(
        contour[:, freq_mask].T,
        x_axis="time",
        y_coords=FREQ_BINS_CONTOURS[freq_mask],
        y_axis="cqt_hz",
        hop_length=256,
        sr=AUDIO_SAMPLE_RATE,
        cmap="inferno"
    )
    plt.title(title)
    plt.colorbar(label="Activation")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_file", type=pathlib.Path, help="Path to the input audio file.")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Directory to write the notes CSV and salience plot to. Defaults to the audio file's directory.",
    )
    args = parser.parse_args()

    audio_path: pathlib.Path = args.audio_file
    output_dir = args.output_dir or audio_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model_output, midi_data, note_events = predict(audio_path, model_or_model_path=MODEL_PATH)

    notes_csv_path = output_dir / f"{audio_path.stem}_basic_pitch_notes.csv"
    save_note_events(note_events, notes_csv_path)
    print(f"Saved estimated notes to {notes_csv_path}")

    salience_path = output_dir / f"{audio_path.stem}_basic_pitch_salience.png"
    plot_salience(model_output["contour"], salience_path, title=f"Pitch salience: {audio_path.name}")
    print(f"Saved salience plot to {salience_path}")


if __name__ == "__main__":
    main()
