"""Export Basic Pitch's salience (contour) head to a TensorFlow.js graph model.

Same model as `convert_model_to_savedmodel.py` -- the contour path only, with
the note and onset outputs dropped -- routed through the tfjs converter instead
of being left as a SavedModel. See that module for what the salience path is
and why the weights are read from the variables checkpoint.

Unlike model3, Basic Pitch's input shape is already concrete: the model takes a
fixed-length audio window (AUDIO_N_SAMPLES samples) and computes its own CQT, so
the only placeholder dimension is the batch. That is pinned here, because tfjs
rejects a -1 wherever a concrete tensor has to be built from the input shape
("dimensions ... should be positive integers"). One window in, one salience
frame block out; feed longer audio by chunking it the way
`basic_pitch.inference` does.

Requires `tensorflowjs`, which is not part of Basic Pitch's dependencies:

    pip install tensorflowjs

Be aware that it pins `tensorflow<2.20` and pulls in tensorflow_decision_forests,
jax and flax, so installing it alongside Basic Pitch downgrades TensorFlow. If
that matters, install it into a throwaway environment instead and convert the
SavedModel that `convert_model_to_savedmodel.py` writes:

    tensorflowjs_converter --input_format=tf_saved_model --strip_debug_ops=True \
        nmp_salience nmp_salience_tfjs

Usage:
    python convert_model_to_tfjs.py [model_dir] [output_dir] [--batch-size N]

Defaults to the bundled ICASSP 2022 model, resolved through the installed
`basic_pitch` package so it does not depend on the working directory, and writes
to ./nmp_salience_tfjs.
"""

import argparse
import os
import shutil
import sys
import tempfile
import types

try:
    import tensorflow_decision_forests  # noqa: F401
except Exception:
    # tensorflowjs imports tensorflow_decision_forests unconditionally, but only
    # uses it for decision-forest models. It is the usual casualty of a protobuf
    # gencode/runtime mismatch, so stub it rather than let an unrelated
    # dependency block a CNN conversion.
    sys.modules["tensorflow_decision_forests"] = types.ModuleType("tensorflow_decision_forests")

from tensorflowjs.converters import tf_saved_model_conversion_v2

from convert_model_to_savedmodel import (
    DEFAULT_MODEL_DIR,
    DEFAULT_N_HARMONICS,
    build_salience_model,
    export_saved_model,
    load_pretrained_weights,
    verify_against_full_model,
)

DEFAULT_BATCH_SIZE = 1


def convert(model_dir, output_dir, n_harmonics=DEFAULT_N_HARMONICS,
            batch_size=DEFAULT_BATCH_SIZE, verify=False, weights=None):
    model = build_salience_model(n_harmonics=n_harmonics, batch_size=batch_size)
    if weights:
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

    saved_model_dir = tempfile.mkdtemp(prefix="basic_pitch_salience_")
    try:
        export_saved_model(model, saved_model_dir, batch_size)
        # strip_debug_ops drops the Assert that HarmonicStacking's shape check
        # emits. tfjs has no kernel for it and rejects the graph otherwise; it
        # is a debug-only node, so removing it does not change the output.
        tf_saved_model_conversion_v2.convert_tf_saved_model(
            saved_model_dir, output_dir, strip_debug_ops=True
        )
    finally:
        shutil.rmtree(saved_model_dir, ignore_errors=True)

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
        help="Path to output tfjs model directory (default: ./<model_dir name>_salience_tfjs)",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of audio windows the exported model accepts (default: {})".format(DEFAULT_BATCH_SIZE),
    )
    parser.add_argument(
        "--harmonics",
        dest="n_harmonics",
        type=int,
        default=DEFAULT_N_HARMONICS,
        help="Number of harmonics in the stacking layer (default: {})".format(DEFAULT_N_HARMONICS),
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Export a .weights.h5 from finetune.py instead of the pretrained weights",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare the salience map against the untouched full model before converting",
    )
    args = parser.parse_args()

    output_path = args.output_path
    if output_path is None:
        # Into the working directory, not next to the source model -- that one
        # lives inside the installed package.
        base = (os.path.basename(args.weights).split('.')[0] if args.weights
                else os.path.basename(args.model_dir.rstrip(os.sep)))
        output_path = base + "_salience_tfjs"

    convert(args.model_dir, output_path, args.n_harmonics, args.batch_size,
            args.verify, args.weights)


if __name__ == "__main__":
    main()
