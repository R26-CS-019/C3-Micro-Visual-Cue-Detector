"""
17_continue_efficientnetv2b0_finetuning.py

Run one controlled continuation experiment from the selected Step 15
EfficientNetV2B0 model.

Scientific safeguards
---------------------
- The existing selected model is read-only and is never overwritten.
- Only TRAIN is used for gradient updates.
- VALIDATION is used for checkpointing and candidate comparison.
- TEST images are never opened, decoded, placed in a TensorFlow dataset,
  predicted, evaluated, or used for selection.
- The Step 15 threshold remains fixed at 0.4660 for the comparison.
- The candidate is selected only if validation AP improves by at least 0.001
  while validation accuracy and MCC do not decrease.
- Original and prepared screenshots are never modified.
"""

import csv
import hashlib
import json
import math
import os
import random
import shutil
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
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
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

BASELINE_MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "model1_efficientnetv2b0_top_viewport"
    / "model1_efficientnetv2b0_selected_best.keras"
)

BASELINE_REPORT_DIR = (
    PROJECT_DIR
    / "reports"
    / "15_efficientnetv2b0_top_viewport_training"
)

BASELINE_METRICS_PATH = (
    BASELINE_REPORT_DIR
    / "validation_metrics.json"
)

BASELINE_PREDICTIONS_PATH = (
    BASELINE_REPORT_DIR
    / "validation_predictions.csv"
)

CANDIDATE_MODEL_DIR = (
    PROJECT_DIR
    / "models"
    / "model1_efficientnetv2b0_top_viewport_continuation"
)

CANDIDATE_CHECKPOINT_PATH = (
    CANDIDATE_MODEL_DIR
    / "continued_finetuning_best.keras"
)

SELECTED_CANDIDATE_PATH = (
    CANDIDATE_MODEL_DIR
    / "model1_efficientnetv2b0_continued_selected.keras"
)

REPORT_DIR = (
    PROJECT_DIR
    / "reports"
    / "17_efficientnetv2b0_continued_finetuning"
)

EXPECTED_MANIFEST_SHA256 = (
    "c83a0c78069b61d24c0362d79d253394fc425a5ed21dd48b1084625278381ab9"
)

IMAGE_WIDTH = 384
IMAGE_HEIGHT = 160
IMAGE_CHANNELS = 3
BATCH_SIZE = 16

RANDOM_SEED = 20260826

CONTINUATION_MAX_EPOCHS = 15
CONTINUATION_LEARNING_RATE = 2.5e-6
CONTINUATION_EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5
MINIMUM_LEARNING_RATE = 1e-7

FINE_TUNE_LAST_LAYERS = 40
FIXED_THRESHOLD = 0.4660
MINIMUM_AP_GAIN = 0.0010

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

EXPECTED_PROCESSING_ACTION = (
    "UNIFORM_WIDTH_SCALE_TOP_VIEWPORT"
)

CLASS_TO_NUMERIC = {
    "NOT_MALICIOUS": 0,
    "MALICIOUS": 1,
}

NUMERIC_TO_CLASS = {
    0: "NOT_MALICIOUS",
    1: "MALICIOUS",
}

REQUIRED_MANIFEST_COLUMNS = {
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

REQUIRED_BASELINE_PREDICTION_COLUMNS = {
    "validation_index",
    "prepared_relative_path",
    "actual_numeric_label",
    "actual_class",
    "malicious_probability",
    "selected_validation_threshold",
    "predicted_numeric_label",
    "predicted_class",
    "correct_prediction",
    "result_type",
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def calculate_sha256(file_path):
    """Calculate SHA256 without modifying the file."""

    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def safe_divide(numerator, denominator):
    """Divide safely and return zero when the denominator is zero."""

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def write_json(output_path, data):
    """Write formatted JSON."""

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_json(input_path):
    """Read one JSON object."""

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Expected a JSON object in:\n{input_path}"
        )

    return data


def format_elapsed(seconds):
    """Format elapsed seconds as HH:MM:SS."""

    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def normalise_relative_path(value, context):
    """Return a safe platform-specific relative path."""

    cleaned = str(value).strip().replace("\\", "/")

    if not cleaned:
        raise RuntimeError(
            f"Empty relative path in {context}."
        )

    parts = [part for part in cleaned.split("/") if part]

    if (
        not parts
        or any(part in {".", ".."} for part in parts)
        or ":" in parts[0]
    ):
        raise RuntimeError(
            f"Unsafe relative path in {context}: {value!r}"
        )

    relative_path = Path(*parts)

    if relative_path.is_absolute():
        raise RuntimeError(
            f"Expected a relative path in {context}: {value!r}"
        )

    return relative_path


def configure_reproducibility():
    """Set deterministic random seeds where TensorFlow supports them."""

    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    try:
        tf.config.experimental.enable_op_determinism()
        determinism_status = "ENABLED"
    except Exception:
        determinism_status = "BEST_EFFORT"

    return determinism_status


def ensure_required_inputs():
    """Validate all required inputs before creating output folders."""

    required_files = [
        MANIFEST_PATH,
        BASELINE_MODEL_PATH,
        BASELINE_METRICS_PATH,
        BASELINE_PREDICTIONS_PATH,
    ]

    missing_files = [
        path for path in required_files if not path.is_file()
    ]

    if missing_files:
        formatted = "\n".join(
            f"  - {path}" for path in missing_files
        )

        raise FileNotFoundError(
            "Required input files were not found:\n"
            f"{formatted}"
        )

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            "Prepared dataset directory was not found:\n"
            f"{DATASET_DIR}"
        )


# =============================================================================
# MANIFEST AND IMAGE VALIDATION
# =============================================================================

def read_manifest():
    """Read the manifest while retaining only TRAIN/VALIDATION images."""

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])

        missing_columns = REQUIRED_MANIFEST_COLUMNS - fieldnames

        if missing_columns:
            raise RuntimeError(
                "Manifest is missing required columns:\n"
                f"{sorted(missing_columns)}"
            )

        rows = list(reader)

    split_counts = Counter()
    split_class_counts = Counter()
    seen_paths = set()
    train_records = []
    validation_records = []

    for row_number, row in enumerate(rows, start=2):
        split = row["split"].strip().upper()
        class_label = row["class_label"].strip().upper()

        if split not in EXPECTED_SPLIT_COUNTS:
            raise RuntimeError(
                f"Unexpected split at manifest row {row_number}: "
                f"{split!r}"
            )

        if class_label not in CLASS_TO_NUMERIC:
            raise RuntimeError(
                f"Unexpected class at manifest row {row_number}: "
                f"{class_label!r}"
            )

        try:
            numeric_label = int(row["numeric_label"])
            output_width = int(row["output_width"])
            output_height = int(row["output_height"])
        except ValueError as error:
            raise RuntimeError(
                f"Invalid numeric field at manifest row "
                f"{row_number}."
            ) from error

        if numeric_label != CLASS_TO_NUMERIC[class_label]:
            raise RuntimeError(
                "Class/numeric-label mismatch at manifest row "
                f"{row_number}."
            )

        if (
            row["processing_action"].strip().upper()
            != EXPECTED_PROCESSING_ACTION
        ):
            raise RuntimeError(
                "Unexpected processing action at manifest row "
                f"{row_number}."
            )

        if (
            output_width != IMAGE_WIDTH
            or output_height != IMAGE_HEIGHT
        ):
            raise RuntimeError(
                "Unexpected output dimensions at manifest row "
                f"{row_number}."
            )

        if row["output_colour_mode"].strip().upper() != "RGB":
            raise RuntimeError(
                "Unexpected colour mode at manifest row "
                f"{row_number}."
            )

        if row["output_format"].strip().upper() != "PNG":
            raise RuntimeError(
                "Unexpected image format at manifest row "
                f"{row_number}."
            )

        relative_path = normalise_relative_path(
            row["prepared_relative_path"],
            f"manifest row {row_number}",
        )

        path_key = str(relative_path).casefold()

        if path_key in seen_paths:
            raise RuntimeError(
                "Duplicate prepared path in manifest:\n"
                f"{relative_path}"
            )

        seen_paths.add(path_key)

        output_sha256 = row["output_sha256"].strip().lower()

        if (
            len(output_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in output_sha256
            )
        ):
            raise RuntimeError(
                f"Invalid SHA256 at manifest row {row_number}."
            )

        split_counts[split] += 1
        split_class_counts[(split, class_label)] += 1

        if split == "TEST":
            # Deliberately do not construct or inspect a TEST image path.
            continue

        record = {
            "split": split,
            "class_label": class_label,
            "numeric_label": numeric_label,
            "prepared_relative_path": relative_path,
            "original_relative_path": row[
                "original_relative_path"
            ].strip(),
            "expected_sha256": output_sha256,
            "image_path": DATASET_DIR / relative_path,
        }

        if split == "TRAIN":
            train_records.append(record)
        else:
            validation_records.append(record)

    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(
            "Unexpected split counts.\n"
            f"Expected: {EXPECTED_SPLIT_COUNTS}\n"
            f"Actual  : {dict(split_counts)}"
        )

    for key, expected_count in (
        EXPECTED_SPLIT_CLASS_COUNTS.items()
    ):
        actual_count = split_class_counts[key]

        if actual_count != expected_count:
            raise RuntimeError(
                f"Unexpected count for {key}: expected "
                f"{expected_count}, found {actual_count}."
            )

    return train_records, validation_records


def preflight_records(records, label):
    """Validate file, SHA256, dimensions and RGB mode for selected records."""

    print(f"Preflighting {label} images...")

    for index, record in enumerate(records, start=1):
        image_path = record["image_path"]

        if not image_path.is_file():
            raise FileNotFoundError(
                f"{label} image was not found:\n{image_path}"
            )

        actual_sha256 = calculate_sha256(image_path)

        if actual_sha256 != record["expected_sha256"]:
            raise RuntimeError(
                f"SHA256 mismatch for {label} image:\n"
                f"{image_path}\n"
                f"Expected: {record['expected_sha256']}\n"
                f"Actual  : {actual_sha256}"
            )

        try:
            with Image.open(image_path) as image:
                image.load()

                if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                    raise RuntimeError(
                        f"Unexpected dimensions for:\n{image_path}\n"
                        f"Found: {image.size}"
                    )

                if image.mode != "RGB":
                    raise RuntimeError(
                        f"Unexpected colour mode for:\n{image_path}\n"
                        f"Found: {image.mode}"
                    )

        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError(
                f"Could not read image:\n{image_path}\n{error}"
            ) from error

        if index % 250 == 0 or index == len(records):
            print(f"  Checked {index}/{len(records)}")


# =============================================================================
# BASELINE VALIDATION
# =============================================================================

def read_baseline_metrics():
    """Read and validate the Step 15 baseline metadata."""

    metrics = load_json(BASELINE_METRICS_PATH)

    required_keys = {
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "mcc",
        "roc_auc",
        "average_precision_pr_auc",
        "false_positive_rate",
        "false_negative_rate",
        "negative_predictive_value",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "validation_images",
        "selected_model_sha256",
    }

    missing_keys = required_keys - set(metrics)

    if missing_keys:
        raise RuntimeError(
            "Baseline validation metrics are missing keys:\n"
            f"{sorted(missing_keys)}"
        )

    if not math.isclose(
        float(metrics["threshold"]),
        FIXED_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "The Step 15 threshold is not the expected fixed "
            f"threshold {FIXED_THRESHOLD:.4f}."
        )

    if int(metrics["validation_images"]) != 426:
        raise RuntimeError(
            "Unexpected validation image count in baseline metrics."
        )

    actual_model_sha256 = calculate_sha256(
        BASELINE_MODEL_PATH
    )

    saved_model_sha256 = str(
        metrics["selected_model_sha256"]
    ).strip().lower()

    if actual_model_sha256 != saved_model_sha256:
        raise RuntimeError(
            "Baseline model SHA256 does not match the Step 15 "
            "validation report.\n"
            f"Expected: {saved_model_sha256}\n"
            f"Actual  : {actual_model_sha256}"
        )

    return metrics, actual_model_sha256


def expected_result_type(actual, predicted):
    """Return the confusion-matrix category for one prediction."""

    if actual == 0 and predicted == 0:
        return "TRUE_NEGATIVE"

    if actual == 0 and predicted == 1:
        return "FALSE_POSITIVE"

    if actual == 1 and predicted == 0:
        return "FALSE_NEGATIVE"

    return "TRUE_POSITIVE"


def read_baseline_probabilities(validation_records):
    """Read Step 15 probabilities in manifest validation order."""

    with open(
        BASELINE_PREDICTIONS_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])

        missing_columns = (
            REQUIRED_BASELINE_PREDICTION_COLUMNS
            - fieldnames
        )

        if missing_columns:
            raise RuntimeError(
                "Baseline predictions are missing columns:\n"
                f"{sorted(missing_columns)}"
            )

        rows = list(reader)

    if len(rows) != len(validation_records):
        raise RuntimeError(
            "Baseline prediction count does not match the "
            "VALIDATION manifest count."
        )

    probabilities = []

    for index, (row, record) in enumerate(
        zip(rows, validation_records),
        start=1,
    ):
        try:
            saved_index = int(row["validation_index"])
            actual_label = int(row["actual_numeric_label"])
            predicted_label = int(
                row["predicted_numeric_label"]
            )
            probability = float(
                row["malicious_probability"]
            )
            threshold = float(
                row["selected_validation_threshold"]
            )
        except ValueError as error:
            raise RuntimeError(
                f"Invalid baseline prediction row {index + 1}."
            ) from error

        expected_path = str(
            record["prepared_relative_path"]
        ).replace("\\", "/").casefold()

        saved_path = str(
            row["prepared_relative_path"]
        ).strip().replace("\\", "/").casefold()

        if saved_index != index:
            raise RuntimeError(
                "Unexpected validation index in baseline "
                f"prediction row {index + 1}."
            )

        if saved_path != expected_path:
            raise RuntimeError(
                "Baseline prediction order/path does not match "
                f"VALIDATION manifest at index {index}."
            )

        if actual_label != record["numeric_label"]:
            raise RuntimeError(
                "Baseline prediction label mismatch at "
                f"validation index {index}."
            )

        if not 0.0 <= probability <= 1.0:
            raise RuntimeError(
                "Invalid probability at baseline validation "
                f"index {index}."
            )

        if not math.isclose(
            threshold,
            FIXED_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "Baseline prediction threshold mismatch at "
                f"validation index {index}."
            )

        expected_prediction = int(
            probability >= FIXED_THRESHOLD
        )

        if predicted_label != expected_prediction:
            raise RuntimeError(
                "Baseline saved prediction is inconsistent at "
                f"validation index {index}."
            )

        expected_result = expected_result_type(
            actual_label,
            predicted_label,
        )

        if row["result_type"].strip().upper() != expected_result:
            raise RuntimeError(
                "Baseline result type is inconsistent at "
                f"validation index {index}."
            )

        probabilities.append(probability)

    return np.asarray(probabilities, dtype=np.float64)


# =============================================================================
# TENSORFLOW DATASETS
# =============================================================================

def decode_image(image_path, numeric_label):
    """Decode an RGB screenshot while keeping pixel values in [0, 255]."""

    image_bytes = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image_bytes,
        channels=IMAGE_CHANNELS,
        expand_animations=False,
    )

    image.set_shape(
        [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
    )

    image = tf.cast(image, tf.float32)
    numeric_label = tf.cast(numeric_label, tf.float32)

    return image, numeric_label


def create_dataset(records, training):
    """Create a TRAIN or deterministic VALIDATION dataset."""

    image_paths = [
        str(record["image_path"]) for record in records
    ]

    labels = [
        record["numeric_label"] for record in records
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(records),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        decode_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
    )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# =============================================================================
# MODEL CONTINUATION
# =============================================================================

def locate_efficientnet_base(model):
    """Find the nested EfficientNetV2 base inside the saved classifier."""

    candidates = []

    for layer in model.layers:
        normalised_name = (
            layer.name.lower()
            .replace("-", "")
            .replace("_", "")
        )

        if (
            isinstance(layer, tf.keras.Model)
            and "efficientnetv2" in normalised_name
        ):
            candidates.append(layer)

    if len(candidates) != 1:
        candidate_names = [
            layer.name for layer in candidates
        ]

        raise RuntimeError(
            "Exactly one nested EfficientNetV2 base was expected.\n"
            f"Candidates found: {candidate_names}"
        )

    return candidates[0]


def configure_last_layers(base_model):
    """Keep only the final requested non-BatchNorm layers trainable."""

    base_model.trainable = True

    fine_tune_start = max(
        0,
        len(base_model.layers) - FINE_TUNE_LAST_LAYERS,
    )

    for layer_index, layer in enumerate(base_model.layers):
        if layer_index < fine_tune_start:
            layer.trainable = False
        elif isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        ):
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_base_layers = [
        layer.name
        for layer in base_model.layers
        if layer.trainable
    ]

    if not trainable_base_layers:
        raise RuntimeError(
            "No EfficientNet base layers were made trainable."
        )

    return fine_tune_start, trainable_base_layers


def create_metrics():
    """Create fresh metrics for the continuation compile."""

    return [
        tf.keras.metrics.BinaryAccuracy(
            threshold=0.5,
            name="accuracy",
        ),
        tf.keras.metrics.Precision(
            thresholds=0.5,
            name="precision",
        ),
        tf.keras.metrics.Recall(
            thresholds=0.5,
            name="recall",
        ),
        tf.keras.metrics.AUC(
            curve="ROC",
            num_thresholds=200,
            name="roc_auc",
        ),
        tf.keras.metrics.AUC(
            curve="PR",
            num_thresholds=200,
            name="pr_auc",
        ),
    ]


def compile_continuation_model(model):
    """Compile with a new low-learning-rate optimizer."""

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=CONTINUATION_LEARNING_RATE,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=create_metrics(),
    )


def create_callbacks():
    """Create validation-only continuation callbacks."""

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=CANDIDATE_CHECKPOINT_PATH,
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=CONTINUATION_EARLY_STOPPING_PATIENCE,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_delta=1e-4,
            min_lr=MINIMUM_LEARNING_RATE,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            REPORT_DIR / "continuation_training_log.csv",
            append=False,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def parameter_counts(model):
    """Return total, trainable and non-trainable parameter counts."""

    total = int(model.count_params())

    trainable = int(
        sum(
            np.prod(variable.shape)
            for variable in model.trainable_weights
        )
    )

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
    }


def extract_probabilities(raw_predictions):
    """Convert supported binary-model outputs into class-1 probabilities."""

    raw_predictions = np.asarray(raw_predictions)

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
            "Unexpected prediction output shape: "
            f"{raw_predictions.shape}"
        )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError(
            "Candidate predictions contain non-finite values."
        )

    return np.clip(probabilities, 0.0, 1.0)


# =============================================================================
# METRICS AND SELECTION
# =============================================================================

def calculate_metrics(true_labels, probabilities, threshold):
    """Calculate thresholded and threshold-free validation metrics."""

    predicted_labels = (
        probabilities >= threshold
    ).astype(np.int32)

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "threshold": float(threshold),
        "validation_images": int(len(true_labels)),
        "accuracy": float(
            accuracy_score(true_labels, predicted_labels)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
        "precision": float(
            precision_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        "recall_sensitivity": float(
            recall_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        "specificity": safe_divide(tn, tn + fp),
        "f1_score": float(
            f1_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                true_labels,
                predicted_labels,
            )
        ),
        "roc_auc": float(
            roc_auc_score(true_labels, probabilities)
        ),
        "average_precision_pr_auc": float(
            average_precision_score(
                true_labels,
                probabilities,
            )
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
        "negative_predictive_value": safe_divide(
            tn,
            tn + fn,
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def validate_baseline_recalculation(saved_metrics, recalculated):
    """Ensure Step 15 probabilities reproduce the reported baseline."""

    float_keys = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "mcc",
        "roc_auc",
        "average_precision_pr_auc",
    ]

    integer_keys = [
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    ]

    for key in float_keys:
        if not math.isclose(
            float(saved_metrics[key]),
            float(recalculated[key]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                "Recalculated baseline metric does not match "
                f"Step 15 for {key}.\n"
                f"Saved      : {saved_metrics[key]}\n"
                f"Recalculated: {recalculated[key]}"
            )

    for key in integer_keys:
        if int(saved_metrics[key]) != int(recalculated[key]):
            raise RuntimeError(
                "Recalculated baseline count does not match "
                f"Step 15 for {key}."
            )


def compare_candidate(baseline_metrics, candidate_metrics):
    """Apply the predeclared candidate-selection gates."""

    ap_gain = (
        candidate_metrics["average_precision_pr_auc"]
        - baseline_metrics["average_precision_pr_auc"]
    )

    accuracy_gain = (
        candidate_metrics["accuracy"]
        - baseline_metrics["accuracy"]
    )

    mcc_gain = (
        candidate_metrics["mcc"]
        - baseline_metrics["mcc"]
    )

    ap_gate = ap_gain >= MINIMUM_AP_GAIN
    accuracy_gate = accuracy_gain >= 0.0
    mcc_gate = mcc_gain >= 0.0

    candidate_selected = (
        ap_gate and accuracy_gate and mcc_gate
    )

    return {
        "selection_primary_metric": (
            "VALIDATION_AVERAGE_PRECISION_PR_AUC"
        ),
        "fixed_threshold": FIXED_THRESHOLD,
        "minimum_required_ap_gain": MINIMUM_AP_GAIN,
        "ap_gain": float(ap_gain),
        "accuracy_gain": float(accuracy_gain),
        "mcc_gain": float(mcc_gain),
        "ap_gate_passed": bool(ap_gate),
        "accuracy_non_decrease_gate_passed": bool(
            accuracy_gate
        ),
        "mcc_non_decrease_gate_passed": bool(mcc_gate),
        "candidate_selected": bool(candidate_selected),
        "selection_result": (
            "CONTINUED_CANDIDATE_SELECTED"
            if candidate_selected
            else "BASELINE_RETAINED"
        ),
    }


# =============================================================================
# REPORTS
# =============================================================================

def save_candidate_predictions(
    validation_records,
    probabilities,
):
    """Save candidate predictions at the fixed Step 15 threshold."""

    fieldnames = [
        "validation_index",
        "prepared_relative_path",
        "original_relative_path",
        "actual_numeric_label",
        "actual_class",
        "malicious_probability",
        "fixed_decision_threshold",
        "predicted_numeric_label",
        "predicted_class",
        "correct_prediction",
        "result_type",
    ]

    predicted_labels = (
        probabilities >= FIXED_THRESHOLD
    ).astype(np.int32)

    rows = []

    for index, (record, probability, prediction) in enumerate(
        zip(
            validation_records,
            probabilities,
            predicted_labels,
        ),
        start=1,
    ):
        actual = record["numeric_label"]

        rows.append(
            {
                "validation_index": index,
                "prepared_relative_path": str(
                    record["prepared_relative_path"]
                ),
                "original_relative_path": record[
                    "original_relative_path"
                ],
                "actual_numeric_label": actual,
                "actual_class": NUMERIC_TO_CLASS[actual],
                "malicious_probability": (
                    f"{float(probability):.10f}"
                ),
                "fixed_decision_threshold": (
                    f"{FIXED_THRESHOLD:.4f}"
                ),
                "predicted_numeric_label": int(prediction),
                "predicted_class": NUMERIC_TO_CLASS[
                    int(prediction)
                ],
                "correct_prediction": (
                    "YES" if actual == prediction else "NO"
                ),
                "result_type": expected_result_type(
                    actual,
                    int(prediction),
                ),
            }
        )

    with open(
        REPORT_DIR / "candidate_validation_predictions.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def save_model_summary(model):
    """Save the loaded continuation model architecture."""

    lines = []
    model.summary(print_fn=lines.append)

    with open(
        REPORT_DIR / "continued_model_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(lines))


def save_history_plot(history):
    """Save continuation TRAIN/VALIDATION curves."""

    history_data = history.history
    epochs = range(1, len(history_data.get("loss", [])) + 1)

    metric_pairs = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("pr_auc", "PR-AUC"),
        ("roc_auc", "ROC-AUC"),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    for axis, (metric_name, label) in zip(
        axes.ravel(),
        metric_pairs,
    ):
        validation_name = f"val_{metric_name}"

        if metric_name in history_data:
            axis.plot(
                epochs,
                history_data[metric_name],
                linewidth=2,
                label="TRAIN",
            )

        if validation_name in history_data:
            axis.plot(
                epochs,
                history_data[validation_name],
                linewidth=2,
                label="VALIDATION",
            )

        axis.set_title(f"Continuation {label}")
        axis.set_xlabel("Continuation epoch")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle(
        "EfficientNetV2B0 Controlled Continued Fine-Tuning",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(
        REPORT_DIR / "continuation_training_history.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_metric_comparison_plot(
    baseline_metrics,
    candidate_metrics,
):
    """Save baseline-vs-candidate validation metric bars."""

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "f1_score",
        "mcc",
        "roc_auc",
        "average_precision_pr_auc",
    ]

    labels = [
        "Accuracy",
        "Balanced\naccuracy",
        "F1",
        "MCC",
        "ROC-AUC",
        "AP / PR-AUC",
    ]

    baseline_values = [
        baseline_metrics[name] for name in metric_names
    ]

    candidate_values = [
        candidate_metrics[name] for name in metric_names
    ]

    positions = np.arange(len(metric_names))
    width = 0.36

    figure, axis = plt.subplots(figsize=(13, 7))

    baseline_bars = axis.bar(
        positions - width / 2,
        baseline_values,
        width,
        label="Step 15 baseline",
    )

    candidate_bars = axis.bar(
        positions + width / 2,
        candidate_values,
        width,
        label="Step 17 candidate",
    )

    axis.set_title(
        "VALIDATION Comparison at Fixed Threshold 0.4660"
    )
    axis.set_ylabel("Metric value")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim([0.0, 1.05])
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    axis.bar_label(
        baseline_bars,
        fmt="%.4f",
        padding=3,
        fontsize=8,
    )

    axis.bar_label(
        candidate_bars,
        fmt="%.4f",
        padding=3,
        fontsize=8,
    )

    figure.tight_layout()
    figure.savefig(
        REPORT_DIR / "baseline_vs_candidate_metrics.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_confusion_matrix_comparison(
    baseline_metrics,
    candidate_metrics,
):
    """Save side-by-side fixed-threshold confusion matrices."""

    matrices = [
        np.asarray(
            [
                [
                    baseline_metrics["true_negatives"],
                    baseline_metrics["false_positives"],
                ],
                [
                    baseline_metrics["false_negatives"],
                    baseline_metrics["true_positives"],
                ],
            ]
        ),
        np.asarray(
            [
                [
                    candidate_metrics["true_negatives"],
                    candidate_metrics["false_positives"],
                ],
                [
                    candidate_metrics["false_negatives"],
                    candidate_metrics["true_positives"],
                ],
            ]
        ),
    ]

    titles = [
        "Step 15 baseline",
        "Step 17 candidate",
    ]

    figure, axes = plt.subplots(1, 2, figsize=(13, 6))

    maximum = max(matrix.max() for matrix in matrices)

    for axis, matrix, title in zip(axes, matrices, titles):
        image = axis.imshow(
            matrix,
            cmap="Blues",
            vmin=0,
            vmax=maximum,
        )

        axis.set_title(title)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
        axis.set_xticklabels(
            ["NOT_MALICIOUS", "MALICIOUS"],
            rotation=12,
        )
        axis.set_yticklabels(
            ["NOT_MALICIOUS", "MALICIOUS"]
        )

        for row_index in range(2):
            for column_index in range(2):
                count = matrix[row_index, column_index]

                axis.text(
                    column_index,
                    row_index,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=14,
                    color=(
                        "white"
                        if count > maximum / 2
                        else "black"
                    ),
                )

    figure.colorbar(image, ax=axes.ravel().tolist())
    figure.suptitle(
        "VALIDATION Confusion Matrices at Threshold 0.4660",
        fontsize=16,
    )
    figure.savefig(
        REPORT_DIR / "baseline_vs_candidate_confusion_matrices.png",
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
    print("CONTINUE EFFICIENTNETV2B0 FINE-TUNING")
    print("=" * 80)
    print(f"TensorFlow          : {tf.__version__}")
    print(f"Dataset             : {DATASET_DIR}")
    print(f"Manifest            : {MANIFEST_PATH}")
    print(f"Baseline model      : {BASELINE_MODEL_PATH}")
    print(f"Candidate directory : {CANDIDATE_MODEL_DIR}")
    print(f"Report directory    : {REPORT_DIR}")
    print(
        f"Fixed threshold     : {FIXED_THRESHOLD:.4f}"
    )
    print(
        f"Learning rate       : "
        f"{CONTINUATION_LEARNING_RATE:.8f}"
    )
    print(
        f"Maximum epochs      : {CONTINUATION_MAX_EPOCHS}"
    )
    print(
        f"Fine-tune last      : {FINE_TUNE_LAST_LAYERS} layers"
    )
    print(f"TEST image usage    : NONE")
    print()

    ensure_required_inputs()

    determinism_status = configure_reproducibility()
    print(f"Determinism         : {determinism_status}")

    print("Calculating manifest SHA256...")
    manifest_sha256 = calculate_sha256(MANIFEST_PATH)

    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "Manifest SHA256 does not match the audited Step 13 "
            "manifest.\n"
            f"Expected: {EXPECTED_MANIFEST_SHA256}\n"
            f"Actual  : {manifest_sha256}"
        )

    print(f"Manifest SHA256     : {manifest_sha256}")
    print()

    print("Reading and validating manifest...")
    train_records, validation_records = read_manifest()

    print(f"TRAIN records       : {len(train_records)}")
    print(
        f"VALIDATION records  : {len(validation_records)}"
    )
    print("TEST image records  : 0 retained")
    print()

    preflight_records(train_records, "TRAIN")
    preflight_records(validation_records, "VALIDATION")

    print()
    print("TEST images opened  : 0")
    print()

    print("Validating Step 15 baseline reports...")
    saved_baseline_metrics, baseline_model_sha256 = (
        read_baseline_metrics()
    )

    baseline_probabilities = read_baseline_probabilities(
        validation_records
    )

    validation_true_labels = np.asarray(
        [
            record["numeric_label"]
            for record in validation_records
        ],
        dtype=np.int32,
    )

    baseline_metrics = calculate_metrics(
        validation_true_labels,
        baseline_probabilities,
        FIXED_THRESHOLD,
    )

    validate_baseline_recalculation(
        saved_baseline_metrics,
        baseline_metrics,
    )

    print(
        f"Baseline model SHA256: {baseline_model_sha256}"
    )
    print(
        "Baseline AP / PR-AUC: "
        f"{baseline_metrics['average_precision_pr_auc']:.6f}"
    )
    print(
        f"Baseline accuracy   : "
        f"{baseline_metrics['accuracy']:.6f}"
    )
    print(f"Baseline MCC        : {baseline_metrics['mcc']:.6f}")
    print()

    print("Creating TensorFlow TRAIN and VALIDATION datasets...")
    train_dataset = create_dataset(
        train_records,
        training=True,
    )

    validation_dataset = create_dataset(
        validation_records,
        training=False,
    )

    print("Loading the read-only Step 15 selected model...")
    model = tf.keras.models.load_model(
        BASELINE_MODEL_PATH,
        compile=False,
    )

    if model.input_shape != (
        None,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
    ):
        raise RuntimeError(
            "Unexpected baseline model input shape:\n"
            f"{model.input_shape}"
        )

    efficientnet_base = locate_efficientnet_base(model)

    fine_tune_start, trainable_base_layers = (
        configure_last_layers(efficientnet_base)
    )

    compile_continuation_model(model)
    counts = parameter_counts(model)

    print(f"EfficientNet base   : {efficientnet_base.name}")
    print(f"Base layers         : {len(efficientnet_base.layers)}")
    print(f"Fine-tune start     : {fine_tune_start}")
    print(
        f"Trainable base layers: {len(trainable_base_layers)}"
    )
    print("BatchNorm layers    : Frozen")
    print(
        f"Trainable parameters: "
        f"{counts['trainable_parameters']:,}"
    )

    CANDIDATE_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_model_summary(model)

    print()
    print("=" * 80)
    print("RUNNING CONTROLLED CONTINUATION")
    print("=" * 80)
    print()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=CONTINUATION_MAX_EPOCHS,
        callbacks=create_callbacks(),
        verbose=1,
    )

    if not CANDIDATE_CHECKPOINT_PATH.is_file():
        raise RuntimeError(
            "The continuation checkpoint was not created."
        )

    completed_epochs = len(history.history.get("loss", []))

    print()
    print("Loading best continued candidate checkpoint...")

    candidate_model = tf.keras.models.load_model(
        CANDIDATE_CHECKPOINT_PATH,
        compile=False,
    )

    print("Predicting VALIDATION with the candidate...")
    raw_candidate_predictions = candidate_model.predict(
        validation_dataset,
        verbose=1,
    )

    candidate_probabilities = extract_probabilities(
        raw_candidate_predictions
    )

    if len(candidate_probabilities) != len(validation_records):
        raise RuntimeError(
            "Candidate prediction count does not match "
            "VALIDATION records."
        )

    candidate_metrics = calculate_metrics(
        validation_true_labels,
        candidate_probabilities,
        FIXED_THRESHOLD,
    )

    comparison = compare_candidate(
        baseline_metrics,
        candidate_metrics,
    )

    candidate_checkpoint_sha256 = calculate_sha256(
        CANDIDATE_CHECKPOINT_PATH
    )

    selected_model_path = BASELINE_MODEL_PATH
    selected_model_sha256 = baseline_model_sha256

    if comparison["candidate_selected"]:
        shutil.copy2(
            CANDIDATE_CHECKPOINT_PATH,
            SELECTED_CANDIDATE_PATH,
        )

        selected_model_path = SELECTED_CANDIDATE_PATH
        selected_model_sha256 = calculate_sha256(
            SELECTED_CANDIDATE_PATH
        )

    save_candidate_predictions(
        validation_records,
        candidate_probabilities,
    )

    save_history_plot(history)

    save_metric_comparison_plot(
        baseline_metrics,
        candidate_metrics,
    )

    save_confusion_matrix_comparison(
        baseline_metrics,
        candidate_metrics,
    )

    elapsed_seconds = time.time() - start_time

    report = {
        "experiment": (
            "CONTROLLED_EFFICIENTNETV2B0_CONTINUED_FINETUNING"
        ),
        "dataset_path": str(DATASET_DIR),
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": manifest_sha256,
        "baseline_model_path": str(BASELINE_MODEL_PATH),
        "baseline_model_sha256": baseline_model_sha256,
        "candidate_checkpoint_path": str(
            CANDIDATE_CHECKPOINT_PATH
        ),
        "candidate_checkpoint_sha256": (
            candidate_checkpoint_sha256
        ),
        "selected_model_path": str(selected_model_path),
        "selected_model_sha256": selected_model_sha256,
        "selected_model_source": (
            "STEP_17_CONTINUED_CANDIDATE"
            if comparison["candidate_selected"]
            else "STEP_15_BASELINE"
        ),
        "fixed_threshold": FIXED_THRESHOLD,
        "train_images": len(train_records),
        "validation_images": len(validation_records),
        "test_images_used": 0,
        "continuation_max_epochs": (
            CONTINUATION_MAX_EPOCHS
        ),
        "continuation_epochs_completed": completed_epochs,
        "continuation_learning_rate": (
            CONTINUATION_LEARNING_RATE
        ),
        "fine_tune_last_layers_requested": (
            FINE_TUNE_LAST_LAYERS
        ),
        "fine_tune_start_index": fine_tune_start,
        "trainable_base_layers": trainable_base_layers,
        "batch_normalization_frozen": True,
        "parameter_counts": counts,
        "baseline_validation_metrics": baseline_metrics,
        "candidate_validation_metrics": candidate_metrics,
        "selection_comparison": comparison,
        "elapsed_seconds": float(elapsed_seconds),
        "determinism_status": determinism_status,
        "baseline_model_overwritten": False,
        "diagnostic_test_loaded": False,
        "threshold_changed": False,
        "screenshots_modified": False,
    }

    write_json(
        REPORT_DIR / "continuation_comparison.json",
        report,
    )

    summary_lines = [
        "=" * 80,
        "EFFICIENTNETV2B0 CONTINUED FINE-TUNING COMPLETE",
        "=" * 80,
        "",
        f"Baseline model        : {BASELINE_MODEL_PATH}",
        f"Baseline SHA256       : {baseline_model_sha256}",
        f"Candidate checkpoint  : {CANDIDATE_CHECKPOINT_PATH}",
        f"Candidate SHA256      : {candidate_checkpoint_sha256}",
        f"Fixed threshold       : {FIXED_THRESHOLD:.4f}",
        f"Continuation epochs   : {completed_epochs}",
        f"Elapsed time          : {format_elapsed(elapsed_seconds)}",
        "",
        "DATA USAGE",
        "-" * 80,
        f"TRAIN images          : {len(train_records)}",
        f"VALIDATION images     : {len(validation_records)}",
        "TEST images used     : 0",
        "",
        "BASELINE VALIDATION",
        "-" * 80,
        f"Accuracy              : {baseline_metrics['accuracy']:.6f}",
        f"Balanced accuracy     : {baseline_metrics['balanced_accuracy']:.6f}",
        f"Precision             : {baseline_metrics['precision']:.6f}",
        f"Recall / sensitivity : {baseline_metrics['recall_sensitivity']:.6f}",
        f"Specificity           : {baseline_metrics['specificity']:.6f}",
        f"F1 score              : {baseline_metrics['f1_score']:.6f}",
        f"MCC                   : {baseline_metrics['mcc']:.6f}",
        f"ROC-AUC               : {baseline_metrics['roc_auc']:.6f}",
        f"Average precision     : {baseline_metrics['average_precision_pr_auc']:.6f}",
        f"TN / FP / FN / TP     : {baseline_metrics['true_negatives']} / {baseline_metrics['false_positives']} / {baseline_metrics['false_negatives']} / {baseline_metrics['true_positives']}",
        "",
        "CONTINUED CANDIDATE VALIDATION",
        "-" * 80,
        f"Accuracy              : {candidate_metrics['accuracy']:.6f}",
        f"Balanced accuracy     : {candidate_metrics['balanced_accuracy']:.6f}",
        f"Precision             : {candidate_metrics['precision']:.6f}",
        f"Recall / sensitivity : {candidate_metrics['recall_sensitivity']:.6f}",
        f"Specificity           : {candidate_metrics['specificity']:.6f}",
        f"F1 score              : {candidate_metrics['f1_score']:.6f}",
        f"MCC                   : {candidate_metrics['mcc']:.6f}",
        f"ROC-AUC               : {candidate_metrics['roc_auc']:.6f}",
        f"Average precision     : {candidate_metrics['average_precision_pr_auc']:.6f}",
        f"TN / FP / FN / TP     : {candidate_metrics['true_negatives']} / {candidate_metrics['false_positives']} / {candidate_metrics['false_negatives']} / {candidate_metrics['true_positives']}",
        "",
        "PREDECLARED SELECTION GATES",
        "-" * 80,
        f"Minimum AP gain       : {MINIMUM_AP_GAIN:.6f}",
        f"Actual AP gain        : {comparison['ap_gain']:+.6f}",
        f"Accuracy gain         : {comparison['accuracy_gain']:+.6f}",
        f"MCC gain              : {comparison['mcc_gain']:+.6f}",
        f"AP gate passed        : {comparison['ap_gate_passed']}",
        f"Accuracy gate passed  : {comparison['accuracy_non_decrease_gate_passed']}",
        f"MCC gate passed       : {comparison['mcc_non_decrease_gate_passed']}",
        "",
        "SELECTION RESULT",
        "-" * 80,
        f"Result                : {comparison['selection_result']}",
        f"Selected model        : {selected_model_path}",
        f"Selected model SHA256 : {selected_model_sha256}",
        "",
        "IMPORTANT",
        "-" * 80,
        "The Step 15 baseline model was not overwritten.",
        "The decision threshold was not changed.",
        "The diagnostic TEST split was not loaded or evaluated.",
        "No original or prepared screenshot was modified.",
        "This is one validation-controlled continuation experiment.",
        "Do not tune again using diagnostic TEST results.",
        "=" * 80,
    ]

    summary_text = "\n".join(summary_lines)

    with open(
        REPORT_DIR / "continuation_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(summary_text)

    print()
    print(summary_text)
    print()
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print()
        print("=" * 80)
        print("CONTINUED FINE-TUNING FAILED")
        print("=" * 80)
        print(error)
        print()
        print("The Step 15 baseline model was not overwritten.")
        print("The diagnostic TEST split was not evaluated.")
        print("No screenshot was modified.")

        sys.exit(1)
