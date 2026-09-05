"""Export Basic Pitch's salience (contour) head to the TensorFlow SavedModel format.

Basic Pitch has three outputs -- contour, note and onset -- but the contour
("salience") map is produced by only four weight-bearing layers hanging off the
harmonically stacked CQT:

    audio -> CQT -> NormalizedLog -> BatchNorm -> HarmonicStacking
          -> Conv2D(8, 3x39) -> BatchNorm/ReLU -> Conv2D(1, 5x5, sigmoid) -> contour

The note and onset branches are more than half the checkpoint's weights and are
dead code once you only want salience, so this script rebuilds just the contour
path and exports that. Only the outputs are dropped: every layer that survives
keeps its pretrained weights, so the exported map is bit-for-bit what the full
model would have produced.

Weights are read straight out of the SavedModel's variables checkpoint with
`tf.train.load_checkpoint`, which works regardless of TensorFlow version:
`tf.saved_model.load` on the shipped `saved_model.pb` trips over Keras 2
optimizer slot variables on newer TensorFlow (2.21 fails, 2.19 is fine), and
`basic_pitch.models.model()` does not build under Keras 3 at all. In the
checkpoint the contour path is `layer_with_weights-0` through
`layer_with_weights-3`, in the same order the layers are created here.

Note that the exported model takes ONE fixed-length window and is not a
drop-in for `basic_pitch.inference.predict`, which front-pads the audio, scans
it with overlapping windows (hop = AUDIO_N_SAMPLES - 30 * FFT_HOP) and trims 15
frames off each window's output. Because `NormalizedLog` rescales by each
window's own dynamic range, chunking the audio differently changes every frame's
value, not just the ones near a boundary -- so to reproduce `predict`'s map,
reuse its `get_audio_input` and `unwrap_output`.

Usage:
    python convert_model_to_savedmodel.py [model_dir] [output_dir] [--batch-size N] [--verify]

Defaults to the bundled ICASSP 2022 model, resolved through the installed
`basic_pitch` package so it does not depend on the working directory, and writes
to ./nmp_salience.
"""

import argparse
import os

import numpy as np
import tensorflow as tf
import keras
from keras import layers as klayers

from basic_pitch import FilenameSuffix, build_icassp_2022_model_path, nn
from basic_pitch.constants import (
    ANNOTATIONS_BASE_FREQUENCY,
    ANNOTATIONS_N_SEMITONES,
    AUDIO_N_SAMPLES,
    AUDIO_SAMPLE_RATE,
    CONTOURS_BINS_PER_SEMITONE,
    FFT_HOP,
    N_FREQ_BINS_CONTOURS,
)
from basic_pitch.layers import nnaudio, signal

# Resolved through basic_pitch itself rather than as a path relative to the
# working directory, so the defaults hold wherever the script is run from.
DEFAULT_MODEL_DIR = str(build_icassp_2022_model_path(FilenameSuffix.tf))
DEFAULT_N_HARMONICS = 8  # models.model()'s n_harmonics default

MAX_N_SEMITONES = int(np.floor(12.0 * np.log2(0.5 * AUDIO_SAMPLE_RATE / ANNOTATIONS_BASE_FREQUENCY)))


class NormalizedLog(signal.NormalizedLog):
    """`signal.NormalizedLog` with a Keras 3 compatible `build()`.

    The library version reads `input_shape.rank`, which is a TensorShape
    attribute; Keras 3 hands `build()` a plain tuple. Only the shape check
    differs -- `call()`, and so the normalization itself, is inherited.
    """

    def build(self, input_shape):
        rank = len(input_shape)
        if rank == 4:
            assert input_shape[1] == 1, "If the rank is 4, the second dimension must be length 1"
            self.squeeze_batch = lambda batch: tf.squeeze(batch, axis=1)
        else:
            assert rank == 3, f"Only ranks 3 and 4 are supported!. Received rank {rank} for {input_shape}."
            self.squeeze_batch = lambda batch: batch


class ExpandFreqCh(keras.layers.Layer):
    """Add a trailing channel axis: (batch, time, freq) -> (batch, time, freq, 1).

    `basic_pitch.models.get_cqt` calls `tf.expand_dims` directly, which Keras 3
    rejects on a symbolic KerasTensor.
    """

    def call(self, x):
        return tf.expand_dims(x, -1)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape) + (1,)


def _initializer():
    return keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg", distribution="uniform", seed=None)


def _kernel_constraint():
    return keras.constraints.UnitNorm(axis=[0, 1, 2])


def build_salience_model(n_harmonics=DEFAULT_N_HARMONICS, batch_size=None):
    """Basic Pitch's contour path only, as a Keras model.

    Input: (batch, AUDIO_N_SAMPLES, 1) raw audio at AUDIO_SAMPLE_RATE.
    Output: (batch, n_frames, N_FREQ_BINS_CONTOURS) salience in [0, 1].

    Layer order matters -- `load_pretrained_weights` matches the checkpoint by
    position, so keep this in sync with `basic_pitch.models.model`.
    """
    n_semitones = int(
        np.min([int(np.ceil(12.0 * np.log2(n_harmonics)) + ANNOTATIONS_N_SEMITONES), MAX_N_SEMITONES])
    )

    inputs = keras.Input(shape=(AUDIO_N_SAMPLES, 1), batch_size=batch_size, name="audio")

    x = nn.FlattenAudioCh()(inputs)
    x = nnaudio.CQT(
        sr=AUDIO_SAMPLE_RATE,
        hop_length=FFT_HOP,
        fmin=ANNOTATIONS_BASE_FREQUENCY,
        n_bins=n_semitones * CONTOURS_BINS_PER_SEMITONE,
        bins_per_octave=12 * CONTOURS_BINS_PER_SEMITONE,
    )(x)
    x = NormalizedLog()(x)
    x = ExpandFreqCh()(x)
    x = klayers.BatchNormalization()(x)

    harmonics = [0.5] + list(range(1, n_harmonics)) if n_harmonics > 1 else [1]
    x = nn.HarmonicStacking(CONTOURS_BINS_PER_SEMITONE, harmonics, N_FREQ_BINS_CONTOURS)(x)

    # The (5, 5) contour conv that models.model() defines first is commented out
    # upstream -- it was skipped when the released model was trained -- so the
    # pretrained salience path really is just these two convolutions.
    x = klayers.Conv2D(
        8,
        (3, 39),
        padding="same",
        kernel_initializer=_initializer(),
        kernel_constraint=_kernel_constraint(),
    )(x)
    x = klayers.BatchNormalization()(x)
    x = klayers.ReLU()(x)
    x = klayers.Conv2D(
        1,
        (5, 5),
        padding="same",
        activation="sigmoid",
        kernel_initializer=_initializer(),
        kernel_constraint=_kernel_constraint(),
        name="contours-reduced",
    )(x)
    outputs = nn.FlattenFreqCh(name="contour")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="basic_pitch_salience")


def _weighted_layers(model):
    """The model's weight-bearing layers, in creation order.

    Matches the checkpoint's `layer_with_weights-N` numbering. The CQT and the
    stacking/flattening layers hold no variables, so they are skipped here just
    as they are there.
    """
    return [layer for layer in model.layers if layer.weights]


def load_pretrained_weights(model, model_dir):
    """Copy the contour path's weights out of a Basic Pitch SavedModel checkpoint."""
    checkpoint_path = os.path.join(model_dir, "variables", "variables")
    reader = tf.train.load_checkpoint(checkpoint_path)

    def read(index, name):
        key = "layer_with_weights-{}/{}/.ATTRIBUTES/VARIABLE_VALUE".format(index, name)
        return reader.get_tensor(key)

    layers = _weighted_layers(model)
    expected = ["BatchNormalization", "Conv2D", "BatchNormalization", "Conv2D"]
    actual = [type(layer).__name__ for layer in layers]
    if actual != expected:
        raise RuntimeError("Unexpected layer order {}, expected {}".format(actual, expected))

    for index, layer in enumerate(layers):
        if isinstance(layer, klayers.BatchNormalization):
            weights = [read(index, n) for n in ("gamma", "beta", "moving_mean", "moving_variance")]
        else:
            weights = [read(index, n) for n in ("kernel", "bias")]
        layer.set_weights(weights)

    n_params = int(sum(np.prod(w.shape) for layer in layers for w in layer.get_weights()))
    print("Loaded {} salience parameters from {}".format(n_params, checkpoint_path))
    return model


def verify_against_full_model(model, model_dir, tolerance=1e-5):
    """Check the salience map against the untouched full Basic Pitch model.

    Loading `model_dir` gives the real reference: the same weights, with the
    note and onset branches still attached, so agreement proves the two dropped
    outputs were the only thing removed. That load fails on some TensorFlow
    versions (Keras 2 optimizer slot variables), in which case this falls back
    to the bundled TFLite build -- a weaker check, because TFLite's own op
    fusion puts it about 1e-5 away from the SavedModel, so a real mismatch
    smaller than that would hide inside the tolerance.

    Silence is included deliberately: `NormalizedLog` divides by the window's
    own dynamic range, so an all-zero window is the one input where the
    normalization could plausibly diverge.
    """
    rng = np.random.default_rng(0)
    audio = np.concatenate(
        [
            rng.standard_normal((2, AUDIO_N_SAMPLES, 1)) * 0.1,
            np.zeros((1, AUDIO_N_SAMPLES, 1)),
        ]
    ).astype("float32")

    try:
        signature = tf.saved_model.load(model_dir).signatures["serving_default"]
        input_name = list(signature.structured_input_signature[1])[0]
        reference = signature(**{input_name: tf.constant(audio)})["contour"].numpy()
        source = "full SavedModel"
    except Exception as exc:
        tflite_path = str(build_icassp_2022_model_path(FilenameSuffix.tflite))
        if not os.path.exists(tflite_path):
            print("Skipping verification: {} did not load ({}) and {} is missing".format(
                model_dir, type(exc).__name__, tflite_path))
            return
        print("Full SavedModel did not load ({}), falling back to TFLite".format(type(exc).__name__))
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.resize_tensor_input(interpreter.get_input_details()[0]["index"], audio.shape)
        interpreter.allocate_tensors()
        contour_detail = next(
            d for d in interpreter.get_output_details() if d["shape"][-1] == N_FREQ_BINS_CONTOURS
        )
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], audio)
        interpreter.invoke()
        reference = interpreter.get_tensor(contour_detail["index"])
        source = "TFLite build"
        tolerance = max(tolerance, 1e-4)

    ours = model.predict(audio, verbose=0)
    max_diff = float(np.max(np.abs(ours - reference)))
    agreement = float(np.mean(ours.argmax(-1) == reference.argmax(-1)))
    print(
        "Deviation from the {}: max {:.3e}, peak-bin agreement {:.2%}".format(source, max_diff, agreement)
    )
    if max_diff > tolerance:
        raise RuntimeError(
            "Salience output does not match the reference (max diff {:.3e} > {:.3e})".format(max_diff, tolerance)
        )


def export_saved_model(model, output_dir, batch_size=None):
    """Write `model` out as a SavedModel with an explicit input signature.

    `model.export()` builds its own serving function with a dynamic batch
    dimension and ignores the Input layer's `batch_size`, so the signature has
    to be spelled out here for a concrete batch to survive into the graph.
    """
    input_signature = [tf.TensorSpec((batch_size, AUDIO_N_SAMPLES, 1), tf.float32, name="audio")]
    model.export(output_dir, input_signature=input_signature)


def convert(model_dir, output_dir, n_harmonics=DEFAULT_N_HARMONICS, batch_size=None,
            verify=False, weights=None):
    model = build_salience_model(n_harmonics=n_harmonics, batch_size=batch_size)
    if weights:
        # A fine-tune from finetune.py. The architecture is unchanged, so only
        # the weights differ -- but that means the reference check cannot apply:
        # these weights are meant to disagree with the pretrained model.
        model.load_weights(weights)
        print("Loaded fine-tuned weights from %s" % weights)
        if verify:
            print("--verify skipped: fine-tuned weights differ from the reference "
                  "by design")
            verify = False
    else:
        load_pretrained_weights(model, model_dir)

    if verify:
        verify_against_full_model(model, model_dir)

    export_saved_model(model, output_dir, batch_size)
    print("Saved {}".format(output_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "model_dir",
        nargs="?",
        default=DEFAULT_MODEL_DIR,
        help="Basic Pitch SavedModel directory to take weights from (default: {})".format(DEFAULT_MODEL_DIR),
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help="Path to output SavedModel directory (default: ./<model_dir name>_salience)",
    )
    parser.add_argument(
        "--harmonics",
        dest="n_harmonics",
        type=int,
        default=DEFAULT_N_HARMONICS,
        help="Number of harmonics in the stacking layer (default: {})".format(DEFAULT_N_HARMONICS),
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        help="Pin the number of audio windows the exported model accepts (default: dynamic)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Export a .weights.h5 from finetune.py instead of the pretrained weights",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare the exported salience map against the untouched full model before saving",
    )
    args = parser.parse_args()

    output_path = args.output_path
    if output_path is None:
        # Into the working directory, not next to the source model -- that one
        # lives inside the installed package.
        base = (os.path.basename(args.weights).split('.')[0] if args.weights
                else os.path.basename(args.model_dir.rstrip(os.sep)))
        output_path = base + "_salience"

    convert(args.model_dir, output_path, args.n_harmonics, args.batch_size,
            args.verify, args.weights)


if __name__ == "__main__":
    main()
