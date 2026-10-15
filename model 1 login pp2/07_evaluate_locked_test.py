"""
07_evaluate_locked_test.py

Locked TEST evaluation for Model 1 MobileNetV3Large.

Actual test manifest columns:
- split
- class_label
- numeric_label
- prepared_relative_path
- original_relative_path
- viewport_action
- output_sha256

Important:
- This script does not train the model.
- The threshold remains fixed at 0.1900.
- Test data is not used for model selection.
- Original or prepared images are not modified.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
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

PREPARED_DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "prepared_login_384x160"
)

TEST_MANIFEST_PATH = (
    PROJECT_DIR
    / "reports"
    / "05_dataset_split"
    / "test_manifest.csv"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "model1_mobilenetv3large"
    / "model1_selected_best.keras"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "reports"
    / "07_locked_test_evaluation"
)

DECISION_THRESHOLD = 0.1900

EXPECTED_TEST_IMAGES = 428
EXPECTED_MALICIOUS_IMAGES = 214
EXPECTED_NOT_MALICIOUS_IMAGES = 214

BATCH_SIZE = 16

CLASS_NAMES = {
    0: "NOT_MALICIOUS",
    1: "MALICIOUS",
}


# =============================================================================
# GENERAL FUNCTIONS
# =============================================================================

def calculate_sha256(file_path):
    """Calculate SHA256 without modifying the file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


def safe_divide(numerator, denominator):
    """Safely divide two numbers."""

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def read_test_manifest():
    """Read and validate the locked test manifest."""

    if not TEST_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Test manifest was not found:\n"
            f"{TEST_MANIFEST_PATH}"
        )

    with open(
        TEST_MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        fieldnames = reader.fieldnames or []
        rows = list(reader)

    required_columns = {
        "split",
        "class_label",
        "numeric_label",
        "prepared_relative_path",
        "original_relative_path",
        "viewport_action",
        "output_sha256",
    }

    missing_columns = required_columns.difference(
        fieldnames
    )

    if missing_columns:
        raise RuntimeError(
            "The test manifest is missing these columns:\n"
            f"{sorted(missing_columns)}"
        )

    test_rows = [
        row
        for row in rows
        if row["split"].strip().upper() == "TEST"
    ]

    if len(test_rows) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_IMAGES} TEST rows, "
            f"but found {len(test_rows)}."
        )

    return test_rows


def resolve_prepared_path(relative_path):
    """Resolve a prepared-relative path on Windows."""

    normalized_relative_path = (
        str(relative_path)
        .strip()
        .replace("\\", os.sep)
        .replace("/", os.sep)
    )

    full_path = (
        PREPARED_DATASET_DIR
        / normalized_relative_path
    )

    return full_path


# =============================================================================
# IMAGE AND HASH VALIDATION
# =============================================================================

def prepare_test_records(test_rows):
    """Resolve paths and validate labels and image hashes."""

    records = []

    print("Checking test image paths and hashes...")

    for index, row in enumerate(
        test_rows,
        start=1,
    ):
        try:
            numeric_label = int(
                row["numeric_label"].strip()
            )
        except ValueError as error:
            raise RuntimeError(
                f"Invalid numeric label at row {index}: "
                f"{row['numeric_label']!r}"
            ) from error

        if numeric_label not in {0, 1}:
            raise RuntimeError(
                f"Numeric label must be 0 or 1 at row "
                f"{index}."
            )

        expected_class = CLASS_NAMES[numeric_label]

        manifest_class = (
            row["class_label"]
            .strip()
            .upper()
        )

        if manifest_class != expected_class:
            raise RuntimeError(
                f"Class-label mismatch at row {index}.\n"
                f"Class label: {manifest_class}\n"
                f"Numeric label: {numeric_label}"
            )

        image_path = resolve_prepared_path(
            row["prepared_relative_path"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Prepared test image was not found:\n"
                f"{image_path}"
            )

        expected_hash = (
            row["output_sha256"]
            .strip()
            .lower()
        )

        if expected_hash:
            actual_hash = calculate_sha256(
                image_path
            )

            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"SHA256 mismatch for test image:\n"
                    f"{image_path}\n"
                    f"Expected: {expected_hash}\n"
                    f"Actual  : {actual_hash}"
                )

        records.append(
            {
                "test_index": index,
                "image_path": image_path,
                "prepared_relative_path": (
                    row["prepared_relative_path"]
                ),
                "original_relative_path": (
                    row["original_relative_path"]
                ),
                "viewport_action": (
                    row["viewport_action"]
                ),
                "class_label": manifest_class,
                "numeric_label": numeric_label,
                "output_sha256": expected_hash,
            }
        )

        if (
            index % 100 == 0
            or index == len(test_rows)
        ):
            print(
                f"Checked {index}/{len(test_rows)} images"
            )

    return records


# =============================================================================
# TENSORFLOW DATASET
# =============================================================================

def decode_image(
    image_path,
    image_height,
    image_width,
):
    """
    Decode the prepared image.

    Pixel values remain between 0 and 255 because the saved
    MobileNetV3Large contains its own preprocessing layer.
    """

    image_bytes = tf.io.read_file(
        image_path
    )

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        size=[
            image_height,
            image_width,
        ],
        method="bilinear",
        antialias=True,
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    return image


def create_test_dataset(
    image_paths,
    image_height,
    image_width,
):
    """Create deterministic test batches."""

    dataset = tf.data.Dataset.from_tensor_slices(
        [
            str(path)
            for path in image_paths
        ]
    )

    dataset = dataset.map(
        lambda path: decode_image(
            path,
            image_height,
            image_width,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# =============================================================================
# CSV REPORTS
# =============================================================================

def save_prediction_reports(
    records,
    probabilities,
    predicted_labels,
):
    """Save all predictions and separate FP/FN reports."""

    fieldnames = [
        "test_index",
        "prepared_relative_path",
        "prepared_image_path",
        "original_relative_path",
        "viewport_action",
        "output_sha256",
        "actual_numeric_label",
        "actual_class",
        "malicious_probability",
        "decision_threshold",
        "predicted_numeric_label",
        "predicted_class",
        "correct_prediction",
        "error_type",
        "prediction_confidence",
    ]

    output_rows = []

    for record, probability, prediction in zip(
        records,
        probabilities,
        predicted_labels,
    ):
        actual_label = record["numeric_label"]

        if actual_label == 0 and prediction == 0:
            error_type = "TRUE_NEGATIVE"

        elif actual_label == 0 and prediction == 1:
            error_type = "FALSE_POSITIVE"

        elif actual_label == 1 and prediction == 0:
            error_type = "FALSE_NEGATIVE"

        else:
            error_type = "TRUE_POSITIVE"

        if prediction == 1:
            confidence = probability
        else:
            confidence = 1.0 - probability

        output_rows.append(
            {
                "test_index": record["test_index"],
                "prepared_relative_path": (
                    record["prepared_relative_path"]
                ),
                "prepared_image_path": str(
                    record["image_path"]
                ),
                "original_relative_path": (
                    record["original_relative_path"]
                ),
                "viewport_action": (
                    record["viewport_action"]
                ),
                "output_sha256": (
                    record["output_sha256"]
                ),
                "actual_numeric_label": (
                    actual_label
                ),
                "actual_class": (
                    CLASS_NAMES[actual_label]
                ),
                "malicious_probability": (
                    f"{float(probability):.10f}"
                ),
                "decision_threshold": (
                    f"{DECISION_THRESHOLD:.4f}"
                ),
                "predicted_numeric_label": (
                    int(prediction)
                ),
                "predicted_class": (
                    CLASS_NAMES[int(prediction)]
                ),
                "correct_prediction": (
                    "YES"
                    if actual_label == prediction
                    else "NO"
                ),
                "error_type": error_type,
                "prediction_confidence": (
                    f"{float(confidence):.10f}"
                ),
            }
        )

    all_predictions_path = (
        OUTPUT_DIR
        / "test_predictions.csv"
    )

    with open(
        all_predictions_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(output_rows)

    false_positive_rows = [
        row
        for row in output_rows
        if row["error_type"] == "FALSE_POSITIVE"
    ]

    false_positive_rows.sort(
        key=lambda row: float(
            row["malicious_probability"]
        ),
        reverse=True,
    )

    false_negative_rows = [
        row
        for row in output_rows
        if row["error_type"] == "FALSE_NEGATIVE"
    ]

    false_negative_rows.sort(
        key=lambda row: float(
            row["malicious_probability"]
        ),
    )

    with open(
        OUTPUT_DIR / "false_positives.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            false_positive_rows
        )

    with open(
        OUTPUT_DIR / "false_negatives.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            false_negative_rows
        )


# =============================================================================
# PLOTS
# =============================================================================

def save_confusion_matrix_plot(
    tn,
    fp,
    fn,
    tp,
):
    """Save the test confusion matrix."""

    matrix = np.array(
        [
            [tn, fp],
            [fn, tp],
        ]
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    image = axis.imshow(
        matrix,
        cmap="Blues",
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set_title(
        "Model 1 Locked Test Confusion Matrix"
    )

    axis.set_xlabel(
        "Predicted class"
    )

    axis.set_ylabel(
        "Actual class"
    )

    class_names = [
        "NOT_MALICIOUS",
        "MALICIOUS",
    ]

    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])

    axis.set_xticklabels(
        class_names,
        rotation=15,
    )

    axis.set_yticklabels(
        class_names
    )

    row_totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(
            matrix,
            dtype=float,
        ),
        where=row_totals != 0,
    )

    colour_threshold = (
        matrix.max() / 2
    )

    for row_index in range(2):
        for column_index in range(2):
            count = matrix[
                row_index,
                column_index,
            ]

            percentage = (
                percentages[
                    row_index,
                    column_index,
                ]
                * 100
            )

            axis.text(
                column_index,
                row_index,
                f"{count}\n({percentage:.1f}%)",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12,
                color=(
                    "white"
                    if count > colour_threshold
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


def save_roc_plot(
    true_labels,
    probabilities,
    roc_auc,
):
    """Save the ROC curve."""

    false_positive_rates, true_positive_rates, _ = (
        roc_curve(
            true_labels,
            probabilities,
        )
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    axis.plot(
        false_positive_rates,
        true_positive_rates,
        linewidth=2,
        label=(
            f"MobileNetV3Large "
            f"(ROC-AUC = {roc_auc:.4f})"
        ),
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random classifier",
    )

    axis.set_title(
        "Model 1 Locked Test ROC Curve"
    )

    axis.set_xlabel(
        "False-positive rate"
    )

    axis.set_ylabel(
        "True-positive rate"
    )

    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR / "roc_curve.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_precision_recall_plot(
    true_labels,
    probabilities,
    average_precision,
):
    """Save the precision-recall curve."""

    precision_values, recall_values, _ = (
        precision_recall_curve(
            true_labels,
            probabilities,
        )
    )

    positive_rate = float(
        np.mean(true_labels)
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    axis.plot(
        recall_values,
        precision_values,
        linewidth=2,
        label=(
            f"MobileNetV3Large "
            f"(AP = {average_precision:.4f})"
        ),
    )

    axis.axhline(
        positive_rate,
        linestyle="--",
        color="gray",
        label=(
            f"Random baseline = "
            f"{positive_rate:.4f}"
        ),
    )

    axis.set_title(
        "Model 1 Locked Test Precision-Recall Curve"
    )

    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")

    axis.set_xlim([0.0, 1.0])
    axis.set_ylim([0.0, 1.05])

    axis.grid(alpha=0.25)
    axis.legend(loc="lower left")

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR / "precision_recall_curve.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_probability_distribution(
    true_labels,
    probabilities,
):
    """Save predicted probability distributions."""

    legitimate_probabilities = probabilities[
        true_labels == 0
    ]

    malicious_probabilities = probabilities[
        true_labels == 1
    ]

    bins = np.linspace(
        0.0,
        1.0,
        31,
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.hist(
        legitimate_probabilities,
        bins=bins,
        alpha=0.65,
        color="green",
        label="Actual NOT_MALICIOUS",
    )

    axis.hist(
        malicious_probabilities,
        bins=bins,
        alpha=0.65,
        color="red",
        label="Actual MALICIOUS",
    )

    axis.axvline(
        DECISION_THRESHOLD,
        linestyle="--",
        linewidth=2,
        color="black",
        label=(
            f"Locked threshold = "
            f"{DECISION_THRESHOLD:.4f}"
        ),
    )

    axis.set_title(
        "Locked Test Probability Distribution"
    )

    axis.set_xlabel(
        "Predicted malicious probability"
    )

    axis.set_ylabel(
        "Number of images"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR
        / "probability_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("MODEL 1 LOCKED TEST EVALUATION")
    print("=" * 80)

    print(
        f"TensorFlow          : "
        f"{tf.__version__}"
    )

    print(
        f"Model               : "
        f"{MODEL_PATH}"
    )

    print(
        f"Test manifest       : "
        f"{TEST_MANIFEST_PATH}"
    )

    print(
        f"Prepared dataset    : "
        f"{PREPARED_DATASET_DIR}"
    )

    print(
        f"Decision threshold  : "
        f"{DECISION_THRESHOLD:.4f}"
    )

    print(
        f"Batch size          : "
        f"{BATCH_SIZE}"
    )

    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Selected model was not found:\n"
            f"{MODEL_PATH}"
        )

    if not PREPARED_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Prepared dataset was not found:\n"
            f"{PREPARED_DATASET_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Reading locked test manifest...")

    test_rows = read_test_manifest()

    print(
        f"Test manifest rows  : "
        f"{len(test_rows)}"
    )

    print()

    test_records = prepare_test_records(
        test_rows
    )

    true_labels = np.asarray(
        [
            record["numeric_label"]
            for record in test_records
        ],
        dtype=np.int32,
    )

    image_paths = [
        record["image_path"]
        for record in test_records
    ]

    malicious_count = int(
        np.sum(true_labels == 1)
    )

    not_malicious_count = int(
        np.sum(true_labels == 0)
    )

    print()
    print("LOCKED TEST CLASS COUNTS")
    print("-" * 80)

    print(
        f"Total images        : "
        f"{len(test_records)}"
    )

    print(
        f"MALICIOUS           : "
        f"{malicious_count}"
    )

    print(
        f"NOT_MALICIOUS       : "
        f"{not_malicious_count}"
    )

    if (
        malicious_count
        != EXPECTED_MALICIOUS_IMAGES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_MALICIOUS_IMAGES} "
            f"MALICIOUS images, but found "
            f"{malicious_count}."
        )

    if (
        not_malicious_count
        != EXPECTED_NOT_MALICIOUS_IMAGES
    ):
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_NOT_MALICIOUS_IMAGES} "
            f"NOT_MALICIOUS images, but found "
            f"{not_malicious_count}."
        )

    if len(set(image_paths)) != len(image_paths):
        raise RuntimeError(
            "Duplicate prepared image paths were found "
            "inside the test manifest."
        )

    print()
    print("Calculating reproducibility hashes...")

    model_sha256 = calculate_sha256(
        MODEL_PATH
    )

    manifest_sha256 = calculate_sha256(
        TEST_MANIFEST_PATH
    )

    print(
        f"Model SHA256        : "
        f"{model_sha256}"
    )

    print(
        f"Manifest SHA256     : "
        f"{manifest_sha256}"
    )

    print()
    print("Loading selected model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    input_shape = model.input_shape

    if (
        not isinstance(input_shape, tuple)
        or len(input_shape) != 4
    ):
        raise RuntimeError(
            f"Unexpected model input shape: "
            f"{input_shape}"
        )

    image_height = input_shape[1]
    image_width = input_shape[2]
    image_channels = input_shape[3]

    if (
        image_height is None
        or image_width is None
        or image_channels != 3
    ):
        raise RuntimeError(
            f"Unsupported model input shape: "
            f"{input_shape}"
        )

    print(
        f"Model input         : "
        f"{image_height}x"
        f"{image_width}x"
        f"{image_channels}"
    )

    test_dataset = create_test_dataset(
        image_paths=image_paths,
        image_height=image_height,
        image_width=image_width,
    )

    print()
    print("=" * 80)
    print("RUNNING LOCKED TEST PREDICTIONS")
    print("=" * 80)
    print()

    raw_predictions = model.predict(
        test_dataset,
        verbose=1,
    )

    raw_predictions = np.asarray(
        raw_predictions
    )

    if (
        raw_predictions.ndim == 2
        and raw_predictions.shape[1] == 1
    ):
        probabilities = raw_predictions[:, 0]

    elif (
        raw_predictions.ndim == 2
        and raw_predictions.shape[1] == 2
    ):
        probabilities = raw_predictions[:, 1]

    elif raw_predictions.ndim == 1:
        probabilities = raw_predictions

    else:
        raise RuntimeError(
            f"Unexpected prediction output shape: "
            f"{raw_predictions.shape}"
        )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    if len(probabilities) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_IMAGES} "
            f"predictions, but received "
            f"{len(probabilities)}."
        )

    predicted_labels = (
        probabilities >= DECISION_THRESHOLD
    ).astype(np.int32)

    test_confusion_matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        test_confusion_matrix.ravel()
    )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    balanced_accuracy = balanced_accuracy_score(
        true_labels,
        predicted_labels,
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        zero_division=0,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        zero_division=0,
    )

    mcc = matthews_corrcoef(
        true_labels,
        predicted_labels,
    )

    roc_auc = roc_auc_score(
        true_labels,
        probabilities,
    )

    average_precision = average_precision_score(
        true_labels,
        probabilities,
    )

    false_positive_rate = safe_divide(
        fp,
        fp + tn,
    )

    false_negative_rate = safe_divide(
        fn,
        fn + tp,
    )

    negative_predictive_value = safe_divide(
        tn,
        tn + fn,
    )

    metrics = {
        "evaluation_type": "LOCKED_TEST",
        "model": "MobileNetV3Large",
        "selected_checkpoint": (
            "PHASE_2_FINETUNED"
        ),
        "decision_threshold": (
            DECISION_THRESHOLD
        ),
        "total_test_images": int(
            len(true_labels)
        ),
        "malicious_images": (
            malicious_count
        ),
        "not_malicious_images": (
            not_malicious_count
        ),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "mcc": float(mcc),
        "roc_auc": float(roc_auc),
        "average_precision": float(
            average_precision
        ),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "false_negative_rate": float(
            false_negative_rate
        ),
        "negative_predictive_value": float(
            negative_predictive_value
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "model_path": str(MODEL_PATH),
        "model_sha256": model_sha256,
        "test_manifest_path": str(
            TEST_MANIFEST_PATH
        ),
        "test_manifest_sha256": (
            manifest_sha256
        ),
    }

    save_prediction_reports(
        records=test_records,
        probabilities=probabilities,
        predicted_labels=predicted_labels,
    )

    with open(
        OUTPUT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    report_text = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=[
            "NOT_MALICIOUS",
            "MALICIOUS",
        ],
        digits=6,
        zero_division=0,
    )

    with open(
        OUTPUT_DIR
        / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report_text)

    save_confusion_matrix_plot(
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )

    save_roc_plot(
        true_labels=true_labels,
        probabilities=probabilities,
        roc_auc=roc_auc,
    )

    save_precision_recall_plot(
        true_labels=true_labels,
        probabilities=probabilities,
        average_precision=average_precision,
    )

    save_probability_distribution(
        true_labels=true_labels,
        probabilities=probabilities,
    )

    summary_lines = [
        "=" * 80,
        "MODEL 1 LOCKED TEST EVALUATION COMPLETE",
        "=" * 80,
        "",
        f"Model                  : {MODEL_PATH}",
        f"Decision threshold     : {DECISION_THRESHOLD:.4f}",
        f"Test images            : {len(true_labels)}",
        "",
        "TEST RESULTS",
        "-" * 80,
        f"Accuracy               : {accuracy:.6f}",
        f"Balanced accuracy      : {balanced_accuracy:.6f}",
        f"Precision              : {precision:.6f}",
        f"Recall / sensitivity   : {recall:.6f}",
        f"Specificity            : {specificity:.6f}",
        f"F1 score               : {f1:.6f}",
        f"MCC                    : {mcc:.6f}",
        f"ROC-AUC                : {roc_auc:.6f}",
        f"Average precision      : {average_precision:.6f}",
        f"False-positive rate    : {false_positive_rate:.6f}",
        f"False-negative rate    : {false_negative_rate:.6f}",
        f"Negative predictive val: {negative_predictive_value:.6f}",
        "",
        "CONFUSION MATRIX",
        "-" * 80,
        f"TN: {tn}",
        f"FP: {fp}",
        f"FN: {fn}",
        f"TP: {tp}",
        "",
        "IMPORTANT",
        "-" * 80,
        "This was the locked test evaluation.",
        "The model was not retrained.",
        "The threshold was not changed.",
        "No screenshot was modified.",
        "Do not tune the model using these test results.",
        "=" * 80,
    ]

    summary_text = "\n".join(
        summary_lines
    )

    with open(
        OUTPUT_DIR
        / "test_evaluation_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(summary_text)

    print()
    print(summary_text)

    print()
    print(
        f"Reports saved to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print()
        print("=" * 80)
        print("LOCKED TEST EVALUATION FAILED")
        print("=" * 80)
        print(error)
        print()
        print(
            "No training or threshold tuning "
            "was performed."
        )

        sys.exit(1)