"""Predict one independent webpage screenshot with finalized Model 2.

This script performs inference only. It does not train, fine-tune, or modify
the model and it never changes the locked validation-selected threshold.

Examples
--------
python predict_model2_single_external_image.py "C:\\path\\page.png"
python predict_model2_single_external_image.py "C:\\path\\page.png" --true-label LEGITIMATE
python predict_model2_single_external_image.py "C:\\path\\page.png" --true-label PHISHING
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "model2_effnetv2b0"
    / "stage2"
    / "model2_effnetv2b0_stage2_best.keras"
)

THRESHOLD_FILE = (
    PROJECT_ROOT
    / "reports"
    / "model2_effnetv2b0"
    / "threshold_selection"
    / "selected_threshold.json"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "reports"
    / "model2_effnetv2b0"
    / "external_evaluation"
    / "single_external_predictions.csv"
)

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 384
CHANNELS = 3

EXPECTED_THRESHOLD = 0.392
ALLOWED_TRUE_LABELS = {"LEGITIMATE", "PHISHING"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict one external webpage screenshot with the finalized "
            "Model 2 model and locked threshold."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to an external webpage screenshot.",
    )
    parser.add_argument(
        "--true-label",
        choices=sorted(ALLOWED_TRUE_LABELS),
        help=(
            "Optional known ground-truth label. Use only when the image's "
            "label is independently known."
        ),
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print the prediction without appending it to the CSV log.",
    )
    return parser.parse_args()


def resolve_image_path(value: str | None) -> Path:
    if value is None:
        value = input("Enter the full screenshot path: ").strip().strip('"')

    image_path = Path(value).expanduser().resolve()

    if not image_path.is_file():
        raise SystemExit(
            "\nERROR: Screenshot was not found:\n"
            f"{image_path}"
        )

    return image_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_locked_threshold() -> tuple[float, str]:
    if not THRESHOLD_FILE.is_file():
        raise SystemExit(
            "\nERROR: Locked threshold file was not found:\n"
            f"{THRESHOLD_FILE}\n"
            "Run optimize_model2_validation_threshold.py first."
        )

    with THRESHOLD_FILE.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    if "selected_threshold" not in data:
        raise SystemExit(
            "\nERROR: selected_threshold is missing from:\n"
            f"{THRESHOLD_FILE}"
        )

    threshold = float(data["selected_threshold"])
    selection_rule = str(
        data.get("selection_rule", "validation-selected threshold")
    )

    if not 0.0 < threshold < 1.0:
        raise SystemExit("\nERROR: Locked threshold must be between 0 and 1.")

    if not np.isclose(threshold, EXPECTED_THRESHOLD, atol=0.0005):
        raise SystemExit(
            "\nERROR: The threshold file does not contain the finalized "
            f"threshold {EXPECTED_THRESHOLD:.3f}.\n"
            f"Found: {threshold:.6f}\n"
            "Do not change the threshold after fixed-test evaluation."
        )

    return threshold, selection_rule


def resize_with_white_padding(image: tf.Tensor) -> tf.Tensor:
    """Match the exact preprocessing used during training and evaluation."""
    image = tf.cast(image, tf.float32)

    original_shape = tf.cast(tf.shape(image)[:2], tf.float32)
    original_height = original_shape[0]
    original_width = original_shape[1]

    scale = tf.minimum(
        IMAGE_HEIGHT / original_height,
        IMAGE_WIDTH / original_width,
    )

    new_height = tf.maximum(
        1,
        tf.cast(tf.round(original_height * scale), tf.int32),
    )
    new_width = tf.maximum(
        1,
        tf.cast(tf.round(original_width * scale), tf.int32),
    )

    image = tf.image.resize(
        image,
        [new_height, new_width],
        method="bilinear",
        antialias=True,
    )

    height_padding = IMAGE_HEIGHT - new_height
    width_padding = IMAGE_WIDTH - new_width

    top_padding = height_padding // 2
    bottom_padding = height_padding - top_padding
    left_padding = width_padding // 2
    right_padding = width_padding - left_padding

    image = tf.pad(
        image,
        paddings=[
            [top_padding, bottom_padding],
            [left_padding, right_padding],
            [0, 0],
        ],
        mode="CONSTANT",
        constant_values=255.0,
    )

    image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])
    return image


def load_and_prepare_image(image_path: Path) -> tf.Tensor:
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(
        image_bytes,
        channels=CHANNELS,
        expand_animations=False,
    )
    image.set_shape([None, None, CHANNELS])
    image = resize_with_white_padding(image)
    return tf.expand_dims(image, axis=0)


def append_result(row: dict[str, str]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = OUTPUT_CSV.is_file()

    fieldnames = [
        "prediction_time",
        "image_name",
        "image_path",
        "image_sha256",
        "true_label",
        "predicted_label",
        "phishing_probability",
        "legitimate_probability",
        "locked_threshold",
        "correct_if_label_known",
        "model_path",
    ]

    with OUTPUT_CSV.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_arguments()
    image_path = resolve_image_path(args.image)

    if not MODEL_PATH.is_file():
        raise SystemExit(
            "\nERROR: Finalized Model 2 file was not found:\n"
            f"{MODEL_PATH}"
        )

    threshold, selection_rule = load_locked_threshold()
    image_batch = load_and_prepare_image(image_path)

    print("=" * 80)
    print("MODEL 2 - SINGLE EXTERNAL SCREENSHOT PREDICTION")
    print("=" * 80)
    print("This script performs inference only.")
    print("It does not train, tune or modify the finalized model.")
    print()
    print(f"Image: {image_path}")
    print(f"Model: {MODEL_PATH}")
    print(f"Locked threshold: {threshold:.3f}")
    print(f"Threshold source: {selection_rule}")
    print()

    model = tf.keras.models.load_model(MODEL_PATH)
    phishing_probability = float(
        model.predict(image_batch, verbose=0).reshape(-1)[0]
    )
    legitimate_probability = 1.0 - phishing_probability

    predicted_label = (
        "PHISHING"
        if phishing_probability >= threshold
        else "LEGITIMATE"
    )

    correctness = "NOT_PROVIDED"
    if args.true_label is not None:
        correctness = (
            "CORRECT"
            if predicted_label == args.true_label
            else "INCORRECT"
        )

    print("PREDICTION RESULT")
    print("-" * 80)
    print(f"Predicted label       : {predicted_label}")
    print(f"Phishing probability  : {phishing_probability:.6f}")
    print(f"Legitimate probability: {legitimate_probability:.6f}")
    print(f"Locked threshold      : {threshold:.3f}")

    if args.true_label is not None:
        print(f"Known true label      : {args.true_label}")
        print(f"Prediction result     : {correctness}")

    if not args.no_save:
        append_result(
            {
                "prediction_time": datetime.now().astimezone().isoformat(),
                "image_name": image_path.name,
                "image_path": str(image_path),
                "image_sha256": sha256_file(image_path),
                "true_label": args.true_label or "",
                "predicted_label": predicted_label,
                "phishing_probability": f"{phishing_probability:.8f}",
                "legitimate_probability": f"{legitimate_probability:.8f}",
                "locked_threshold": f"{threshold:.3f}",
                "correct_if_label_known": correctness,
                "model_path": str(MODEL_PATH),
            }
        )
        print()
        print(f"Prediction log: {OUTPUT_CSV}")

    print()
    print("IMPORTANT")
    print("-" * 80)
    print("Do not change the model or threshold based on individual screenshots.")
    print("Use a labelled independent dataset for formal external evaluation.")
    print("=" * 80)


if __name__ == "__main__":
    main()
