"""
18_evaluate_efficientnetv2b0_diagnostic_test.py

One-time diagnostic TEST evaluation for the retained EfficientNetV2B0 model.

This script:
- does not train or modify the model;
- keeps the validation-selected threshold fixed at 0.4660;
- does not tune anything using TEST;
- saves predictions, metrics, error CSV files and plots;
- blocks accidental reruns after successful completion.

The TEST result is diagnostic because this split has already been inspected.
A separate external dataset is still required for final evaluation.
"""

import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(
    r"C:\Users\oshan\Desktop\model 1 login pp2"
)

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "prepared_login_uniform_top_viewport_384x160"
)

MANIFEST_PATH = (
    PROJECT_DIR
    / "reports"
    / "13_uniform_top_viewport_dataset"
    / "complete_uniform_top_viewport_manifest.csv"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "model1_efficientnetv2b0_top_viewport"
    / "model1_efficientnetv2b0_selected_best.keras"
)

VALIDATION_METRICS_PATH = (
    PROJECT_DIR
    / "reports"
    / "15_efficientnetv2b0_top_viewport_training"
    / "validation_metrics.json"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "reports"
    / "18_efficientnetv2b0_diagnostic_test"
)

COMPLETION_MARKER = (
    OUTPUT_DIR / "diagnostic_evaluation_complete.json"
)

EXPECTED_MANIFEST_SHA256 = (
    "c83a0c78069b61d24c0362d79d253394fc425a5ed21dd48b1084625278381ab9"
)

EXPECTED_MODEL_SHA256 = (
    "6be68e8492246210f323b99a89fdc6107d7abfef30db7bb06820b5005497f1af"
)

DECISION_THRESHOLD = 0.4660

IMAGE_WIDTH = 384
IMAGE_HEIGHT = 160
IMAGE_CHANNELS = 3
BATCH_SIZE = 16

EXPECTED_TEST_IMAGES = 428
EXPECTED_TEST_PER_CLASS = 214

EXPECTED_SPLIT_COUNTS = {
    "TRAIN": 1992,
    "VALIDATION": 426,
    "TEST": 428,
}

EXPECTED_SPLIT_CLASS_COUNTS = {
    ("TRAIN", "NOT_MALICIOUS"): 996,
    ("TRAIN", "MALICIOUS"): 996,
    ("VALIDATION", "NOT_MALICIOUS"): 213,
    ("VALIDATION", "MALICIOUS"): 213,
    ("TEST", "NOT_MALICIOUS"): 214,
    ("TEST", "MALICIOUS"): 214,
}

EXPECTED_ACTION = "UNIFORM_WIDTH_SCALE_TOP_VIEWPORT"

CLASS_TO_NUMERIC = {
    "NOT_MALICIOUS": 0,
    "MALICIOUS": 1,
}

NUMERIC_TO_CLASS = {
    0: "NOT_MALICIOUS",
    1: "MALICIOUS",
}

REQUIRED_COLUMNS = {
    "split",
    "class_label",
    "numeric_label",
    "prepared_relative_path",
    "original_relative_path",
    "processing_action",
    "output_width",
    "output_height",
    "output_colour_mode",
    "output_format",
    "output_sha256",
}


# =============================================================================
# GENERAL FUNCTIONS
# =============================================================================

def sha256(file_path):
    """Calculate SHA256 without changing the file."""

    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def safe_divide(numerator, denominator):
    """Divide safely."""

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def read_json(file_path):
    """Read a JSON object."""

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Expected a JSON object in:\n{file_path}"
        )

    return data


def write_json(file_path, data):
    """Write formatted JSON."""

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def normalise_relative_path(value, row_number):
    """Create a safe relative path from either slash style."""

    cleaned = str(value).strip().replace("\\", "/")
    parts = [part for part in cleaned.split("/") if part]

    if (
        not parts
        or any(part in {".", ".."} for part in parts)
        or ":" in parts[0]
    ):
        raise RuntimeError(
            f"Unsafe prepared path at manifest row {row_number}: "
            f"{value!r}"
        )

    relative_path = Path(*parts)

    if relative_path.is_absolute():
        raise RuntimeError(
            f"Expected a relative path at manifest row "
            f"{row_number}."
        )

    return relative_path


def format_elapsed(seconds):
    """Format seconds as HH:MM:SS."""

    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# =============================================================================
# LOCKED INPUT CHECKS
# =============================================================================

def check_inputs():
    """Validate files, hashes, threshold and one-time status."""

    if COMPLETION_MARKER.is_file():
        raise RuntimeError(
            "This one-time diagnostic TEST evaluation already "
            "completed. Read the saved summary instead and do not "
            "rerun or tune using TEST."
        )

    required_files = [
        MANIFEST_PATH,
        MODEL_PATH,
        VALIDATION_METRICS_PATH,
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required file was not found:\n{file_path}"
            )

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory was not found:\n{DATASET_DIR}"
        )

    manifest_hash = sha256(MANIFEST_PATH)
    model_hash = sha256(MODEL_PATH)

    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "Manifest SHA256 mismatch.\n"
            f"Expected: {EXPECTED_MANIFEST_SHA256}\n"
            f"Actual  : {manifest_hash}"
        )

    if model_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "Retained model SHA256 mismatch.\n"
            f"Expected: {EXPECTED_MODEL_SHA256}\n"
            f"Actual  : {model_hash}"
        )

    validation_metrics = read_json(
        VALIDATION_METRICS_PATH
    )

    saved_threshold = float(validation_metrics["threshold"])
    saved_model_hash = str(
        validation_metrics["selected_model_sha256"]
    ).strip().lower()

    if not math.isclose(
        saved_threshold,
        DECISION_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "Saved validation threshold does not match "
            f"{DECISION_THRESHOLD:.4f}."
        )

    if saved_model_hash != model_hash:
        raise RuntimeError(
            "Validation report refers to a different model."
        )

    return manifest_hash, model_hash, validation_metrics


# =============================================================================
# MANIFEST AND IMAGE CHECKS
# =============================================================================

def read_test_records():
    """Validate the complete manifest and return TEST records only."""

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])

        missing = REQUIRED_COLUMNS - columns

        if missing:
            raise RuntimeError(
                "Manifest is missing columns:\n"
                f"{sorted(missing)}"
            )

        rows = list(reader)

    split_counts = Counter()
    split_class_counts = Counter()
    seen_paths = set()
    test_records = []

    for row_number, row in enumerate(rows, start=2):
        split = row["split"].strip().upper()
        class_name = row["class_label"].strip().upper()

        if split not in EXPECTED_SPLIT_COUNTS:
            raise RuntimeError(
                f"Invalid split at row {row_number}: {split!r}"
            )

        if class_name not in CLASS_TO_NUMERIC:
            raise RuntimeError(
                f"Invalid class at row {row_number}: "
                f"{class_name!r}"
            )

        try:
            numeric_label = int(row["numeric_label"])
            width = int(row["output_width"])
            height = int(row["output_height"])
        except ValueError as error:
            raise RuntimeError(
                f"Invalid numeric value at manifest row "
                f"{row_number}."
            ) from error

        if numeric_label != CLASS_TO_NUMERIC[class_name]:
            raise RuntimeError(
                f"Class/label mismatch at row {row_number}."
            )

        if row["processing_action"].strip().upper() != EXPECTED_ACTION:
            raise RuntimeError(
                f"Unexpected processing action at row {row_number}."
            )

        if (width, height) != (IMAGE_WIDTH, IMAGE_HEIGHT):
            raise RuntimeError(
                f"Unexpected dimensions at row {row_number}."
            )

        if row["output_colour_mode"].strip().upper() != "RGB":
            raise RuntimeError(
                f"Unexpected colour mode at row {row_number}."
            )

        if row["output_format"].strip().upper() != "PNG":
            raise RuntimeError(
                f"Unexpected format at row {row_number}."
            )

        relative_path = normalise_relative_path(
            row["prepared_relative_path"],
            row_number,
        )

        path_key = str(relative_path).casefold()

        if path_key in seen_paths:
            raise RuntimeError(
                f"Duplicate prepared path:\n{relative_path}"
            )

        seen_paths.add(path_key)

        expected_hash = row["output_sha256"].strip().lower()

        if (
            len(expected_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_hash
            )
        ):
            raise RuntimeError(
                f"Invalid SHA256 at row {row_number}."
            )

        split_counts[split] += 1
        split_class_counts[(split, class_name)] += 1

        if split == "TEST":
            test_records.append(
                {
                    "test_index": len(test_records) + 1,
                    "class_name": class_name,
                    "numeric_label": numeric_label,
                    "prepared_relative_path": relative_path,
                    "original_relative_path": row[
                        "original_relative_path"
                    ].strip(),
                    "expected_sha256": expected_hash,
                    "image_path": DATASET_DIR / relative_path,
                }
            )

    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(
            "Unexpected split counts.\n"
            f"Expected: {EXPECTED_SPLIT_COUNTS}\n"
            f"Actual  : {dict(split_counts)}"
        )

    for key, expected in EXPECTED_SPLIT_CLASS_COUNTS.items():
        actual = split_class_counts[key]

        if actual != expected:
            raise RuntimeError(
                f"Unexpected count for {key}: expected "
                f"{expected}, found {actual}."
            )

    return test_records


def preflight_images(records):
    """Check TEST files, hashes, dimensions and modes."""

    print("Checking TEST image files and hashes...")

    for index, record in enumerate(records, start=1):
        image_path = record["image_path"]

        if not image_path.is_file():
            raise FileNotFoundError(
                f"TEST image was not found:\n{image_path}"
            )

        actual_hash = sha256(image_path)

        if actual_hash != record["expected_sha256"]:
            raise RuntimeError(
                f"TEST image SHA256 mismatch:\n{image_path}"
            )

        with Image.open(image_path) as image:
            image.load()

            if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                raise RuntimeError(
                    f"Unexpected dimensions for:\n{image_path}"
                )

            if image.mode != "RGB":
                raise RuntimeError(
                    f"Unexpected colour mode for:\n{image_path}"
                )

        if index % 100 == 0 or index == len(records):
            print(f"Checked {index}/{len(records)} images")


# =============================================================================
# TENSORFLOW DATASET
# =============================================================================

def decode_image(image_path):
    """Decode one prepared image with values in [0, 255]."""

    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    return tf.cast(image, tf.float32)


def create_dataset(records):
    """Create deterministic TEST batches."""

    paths = [str(record["image_path"]) for record in records]
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(
        decode_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def extract_probabilities(raw_predictions):
    """Extract malicious-class probabilities."""

    raw_predictions = np.asarray(raw_predictions)

    if raw_predictions.ndim == 1:
        probabilities = raw_predictions
    elif (
        raw_predictions.ndim == 2
        and raw_predictions.shape[1] == 1
    ):
        probabilities = raw_predictions[:, 0]
    elif (
        raw_predictions.ndim == 2
        and raw_predictions.shape[1] == 2
    ):
        probabilities = raw_predictions[:, 1]
    else:
        raise RuntimeError(
            f"Unexpected model output shape: "
            f"{raw_predictions.shape}"
        )

    probabilities = np.asarray(probabilities, dtype=np.float64)

    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("Predictions contain non-finite values.")

    return np.clip(probabilities, 0.0, 1.0)


# =============================================================================
# METRICS AND REPORTS
# =============================================================================

def calculate_metrics(true_labels, probabilities):
    """Calculate fixed-threshold diagnostic metrics."""

    predictions = (
        probabilities >= DECISION_THRESHOLD
    ).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "evaluation_type": "DIAGNOSTIC_TEST",
        "architecture": "EfficientNetV2B0",
        "decision_threshold": DECISION_THRESHOLD,
        "threshold_source": "STEP_15_VALIDATION_ONLY",
        "test_images": int(len(true_labels)),
        "accuracy": float(
            accuracy_score(true_labels, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_labels, predictions)
        ),
        "precision": float(
            precision_score(
                true_labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall_sensitivity": float(
            recall_score(
                true_labels,
                predictions,
                zero_division=0,
            )
        ),
        "specificity": safe_divide(tn, tn + fp),
        "f1_score": float(
            f1_score(
                true_labels,
                predictions,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(true_labels, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(true_labels, probabilities)
        ),
        "average_precision": float(
            average_precision_score(true_labels, probabilities)
        ),
        "binary_log_loss": float(
            log_loss(
                true_labels,
                probabilities,
                labels=[0, 1],
            )
        ),
        "brier_score": float(
            brier_score_loss(true_labels, probabilities)
        ),
        "false_positive_rate": safe_divide(fp, fp + tn),
        "false_negative_rate": safe_divide(fn, fn + tp),
        "negative_predictive_value": safe_divide(tn, tn + fn),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "threshold_tuned_on_test": False,
        "model_trained_by_this_script": False,
        "diagnostic_only": True,
        "external_dataset_required": True,
    }

    return metrics, predictions


def save_predictions(records, probabilities, predictions):
    """Save all predictions plus separate FP and FN CSV files."""

    fields = [
        "test_index",
        "prepared_relative_path",
        "prepared_image_path",
        "original_relative_path",
        "output_sha256",
        "actual_numeric_label",
        "actual_class",
        "malicious_probability",
        "decision_threshold",
        "predicted_numeric_label",
        "predicted_class",
        "correct_prediction",
        "result_type",
    ]

    rows = []

    for record, probability, prediction in zip(
        records,
        probabilities,
        predictions,
    ):
        actual = record["numeric_label"]

        if actual == 0 and prediction == 0:
            result_type = "TRUE_NEGATIVE"
        elif actual == 0 and prediction == 1:
            result_type = "FALSE_POSITIVE"
        elif actual == 1 and prediction == 0:
            result_type = "FALSE_NEGATIVE"
        else:
            result_type = "TRUE_POSITIVE"

        rows.append(
            {
                "test_index": record["test_index"],
                "prepared_relative_path": str(
                    record["prepared_relative_path"]
                ),
                "prepared_image_path": str(record["image_path"]),
                "original_relative_path": record[
                    "original_relative_path"
                ],
                "output_sha256": record["expected_sha256"],
                "actual_numeric_label": actual,
                "actual_class": NUMERIC_TO_CLASS[actual],
                "malicious_probability": f"{probability:.10f}",
                "decision_threshold": f"{DECISION_THRESHOLD:.4f}",
                "predicted_numeric_label": int(prediction),
                "predicted_class": NUMERIC_TO_CLASS[
                    int(prediction)
                ],
                "correct_prediction": (
                    "YES" if actual == prediction else "NO"
                ),
                "result_type": result_type,
            }
        )

    all_path = OUTPUT_DIR / "diagnostic_test_predictions.csv"

    with open(
        all_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    false_positives = [
        row for row in rows
        if row["result_type"] == "FALSE_POSITIVE"
    ]
    false_positives.sort(
        key=lambda row: float(row["malicious_probability"]),
        reverse=True,
    )

    false_negatives = [
        row for row in rows
        if row["result_type"] == "FALSE_NEGATIVE"
    ]
    false_negatives.sort(
        key=lambda row: float(row["malicious_probability"])
    )

    for name, error_rows in [
        ("false_positives.csv", false_positives),
        ("false_negatives.csv", false_negatives),
    ]:
        with open(
            OUTPUT_DIR / name,
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(error_rows)


# =============================================================================
# PLOTS
# =============================================================================

def save_confusion_plot(metrics):
    """Save the confusion matrix."""

    matrix = np.array(
        [
            [metrics["true_negatives"], metrics["false_positives"]],
            [metrics["false_negatives"], metrics["true_positives"]],
        ]
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_title("EfficientNetV2B0 Diagnostic TEST")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(
        ["NOT_MALICIOUS", "MALICIOUS"],
        rotation=15,
    )
    axis.set_yticklabels(
        ["NOT_MALICIOUS", "MALICIOUS"]
    )

    for row_index in range(2):
        for column_index in range(2):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=14,
                color=(
                    "white"
                    if value > matrix.max() / 2
                    else "black"
                ),
            )

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_curves(true_labels, probabilities, metrics):
    """Save ROC, precision-recall and probability plots."""

    false_positive_rates, true_positive_rates, _ = roc_curve(
        true_labels,
        probabilities,
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(
        false_positive_rates,
        true_positive_rates,
        linewidth=2,
        label=f"ROC-AUC = {metrics['roc_auc']:.4f}",
    )
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set_title("Diagnostic TEST ROC Curve")
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "roc_curve.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    precision_values, recall_values, _ = precision_recall_curve(
        true_labels,
        probabilities,
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(
        recall_values,
        precision_values,
        linewidth=2,
        label=f"AP = {metrics['average_precision']:.4f}",
    )
    axis.axhline(0.5, linestyle="--", color="gray")
    axis.set_title("Diagnostic TEST Precision-Recall Curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim([0, 1])
    axis.set_ylim([0, 1.05])
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "precision_recall_curve.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0, 1, 31)
    axis.hist(
        probabilities[true_labels == 0],
        bins=bins,
        alpha=0.65,
        color="green",
        label="Actual NOT_MALICIOUS",
    )
    axis.hist(
        probabilities[true_labels == 1],
        bins=bins,
        alpha=0.65,
        color="red",
        label="Actual MALICIOUS",
    )
    axis.axvline(
        DECISION_THRESHOLD,
        linestyle="--",
        color="black",
        linewidth=2,
        label=f"Threshold = {DECISION_THRESHOLD:.4f}",
    )
    axis.set_title("Diagnostic TEST Probability Distribution")
    axis.set_xlabel("Predicted malicious probability")
    axis.set_ylabel("Number of images")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "probability_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = time.time()

    print("=" * 80)
    print("EFFICIENTNETV2B0 ONE-TIME DIAGNOSTIC TEST")
    print("=" * 80)
    print(f"TensorFlow          : {tf.__version__}")
    print(f"Model               : {MODEL_PATH}")
    print(f"Dataset             : {DATASET_DIR}")
    print(f"Manifest            : {MANIFEST_PATH}")
    print(f"Output directory    : {OUTPUT_DIR}")
    print(f"Fixed threshold     : {DECISION_THRESHOLD:.4f}")
    print(f"Batch size          : {BATCH_SIZE}")
    print("Evaluation          : DIAGNOSTIC ONLY")
    print()

    manifest_hash, model_hash, validation_metrics = (
        check_inputs()
    )

    print(f"Model SHA256        : {model_hash}")
    print(f"Manifest SHA256     : {manifest_hash}")
    print()

    print("Reading TEST records...")
    test_records = read_test_records()

    true_labels = np.asarray(
        [record["numeric_label"] for record in test_records],
        dtype=np.int32,
    )

    not_malicious_count = int(np.sum(true_labels == 0))
    malicious_count = int(np.sum(true_labels == 1))

    if len(test_records) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_IMAGES} TEST images, "
            f"but found {len(test_records)}."
        )

    if (
        not_malicious_count != EXPECTED_TEST_PER_CLASS
        or malicious_count != EXPECTED_TEST_PER_CLASS
    ):
        raise RuntimeError("Unexpected TEST class counts.")

    print(f"TEST images         : {len(test_records)}")
    print(f"NOT_MALICIOUS       : {not_malicious_count}")
    print(f"MALICIOUS           : {malicious_count}")
    print()

    preflight_images(test_records)

    print()
    print("Loading retained model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    expected_shape = (
        None,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
    )

    if model.input_shape != expected_shape:
        raise RuntimeError(
            f"Unexpected model input shape: {model.input_shape}"
        )

    test_dataset = create_dataset(test_records)

    print()
    print("=" * 80)
    print("RUNNING DIAGNOSTIC TEST PREDICTIONS")
    print("=" * 80)
    print()

    raw_predictions = model.predict(
        test_dataset,
        verbose=1,
    )

    probabilities = extract_probabilities(raw_predictions)

    if len(probabilities) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            "Prediction count does not match TEST image count."
        )

    metrics, predictions = calculate_metrics(
        true_labels,
        probabilities,
    )

    metrics.update(
        {
            "model_path": str(MODEL_PATH),
            "model_sha256": model_hash,
            "manifest_path": str(MANIFEST_PATH),
            "manifest_sha256": manifest_hash,
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_predictions(test_records, probabilities, predictions)

    write_json(
        OUTPUT_DIR / "diagnostic_test_metrics.json",
        metrics,
    )

    report = classification_report(
        true_labels,
        predictions,
        labels=[0, 1],
        target_names=["NOT_MALICIOUS", "MALICIOUS"],
        digits=6,
        zero_division=0,
    )

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    save_confusion_plot(metrics)
    save_curves(true_labels, probabilities, metrics)

    elapsed = time.time() - start_time

    validation_accuracy = float(
        validation_metrics["accuracy"]
    )
    validation_f1 = float(
        validation_metrics["f1_score"]
    )
    validation_mcc = float(validation_metrics["mcc"])
    validation_roc_auc = float(
        validation_metrics["roc_auc"]
    )
    validation_ap = float(
        validation_metrics["average_precision_pr_auc"]
    )

    summary_lines = [
        "=" * 80,
        "EFFICIENTNETV2B0 DIAGNOSTIC TEST COMPLETE",
        "=" * 80,
        "",
        f"Model                  : {MODEL_PATH}",
        f"Model SHA256           : {model_hash}",
        f"Manifest SHA256        : {manifest_hash}",
        f"Decision threshold     : {DECISION_THRESHOLD:.4f}",
        "Threshold source       : VALIDATION ONLY",
        f"Elapsed time           : {format_elapsed(elapsed)}",
        "",
        "DIAGNOSTIC TEST RESULTS",
        "-" * 80,
        f"Test images            : {metrics['test_images']}",
        f"Accuracy               : {metrics['accuracy']:.6f}",
        f"Balanced accuracy      : {metrics['balanced_accuracy']:.6f}",
        f"Precision              : {metrics['precision']:.6f}",
        f"Recall / sensitivity   : {metrics['recall_sensitivity']:.6f}",
        f"Specificity            : {metrics['specificity']:.6f}",
        f"F1 score               : {metrics['f1_score']:.6f}",
        f"MCC                    : {metrics['mcc']:.6f}",
        f"ROC-AUC                : {metrics['roc_auc']:.6f}",
        f"Average precision      : {metrics['average_precision']:.6f}",
        f"Binary log loss        : {metrics['binary_log_loss']:.6f}",
        f"Brier score            : {metrics['brier_score']:.6f}",
        f"False-positive rate    : {metrics['false_positive_rate']:.6f}",
        f"False-negative rate    : {metrics['false_negative_rate']:.6f}",
        f"Negative predictive val: {metrics['negative_predictive_value']:.6f}",
        "",
        "CONFUSION MATRIX",
        "-" * 80,
        f"TN: {metrics['true_negatives']}",
        f"FP: {metrics['false_positives']}",
        f"FN: {metrics['false_negatives']}",
        f"TP: {metrics['true_positives']}",
        "",
        "TEST MINUS VALIDATION",
        "-" * 80,
        f"Accuracy difference    : {metrics['accuracy'] - validation_accuracy:+.6f}",
        f"F1 difference          : {metrics['f1_score'] - validation_f1:+.6f}",
        f"MCC difference         : {metrics['mcc'] - validation_mcc:+.6f}",
        f"ROC-AUC difference     : {metrics['roc_auc'] - validation_roc_auc:+.6f}",
        f"Average precision diff : {metrics['average_precision'] - validation_ap:+.6f}",
        "",
        "IMPORTANT",
        "-" * 80,
        "These results are diagnostic, not final external results.",
        "The model was not trained or modified.",
        "The threshold was not tuned or changed using TEST.",
        "Do not use these TEST results for further tuning.",
        "A separate external dataset is required for final evaluation.",
        "No original or prepared screenshot was modified.",
        "=" * 80,
    ]

    summary = "\n".join(summary_lines)
    summary_path = OUTPUT_DIR / "diagnostic_test_summary.txt"

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(summary)

    # Write this last so a failed or interrupted run may be restarted.
    write_json(
        COMPLETION_MARKER,
        {
            "status": "COMPLETE",
            "summary_path": str(summary_path),
            "model_sha256": model_hash,
            "manifest_sha256": manifest_hash,
            "decision_threshold": DECISION_THRESHOLD,
            "threshold_tuned_on_test": False,
            "external_dataset_required": True,
        },
    )

    print()
    print(summary)
    print()
    print(f"Reports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print()
        print("=" * 80)
        print("DIAGNOSTIC TEST FAILED OR BLOCKED")
        print("=" * 80)
        print(error)
        print()
        print("No model training or threshold tuning was performed.")
        print("No screenshot was modified.")

        sys.exit(1)
