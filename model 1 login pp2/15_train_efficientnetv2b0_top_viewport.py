"""
15_train_efficientnetv2b0_top_viewport.py

Train Model 1 using EfficientNetV2B0 and the audited uniform top-viewport
dataset.

Scientific rules enforced by this script:

- TRAIN is used for gradient updates.
- VALIDATION is used for early stopping, checkpoint selection and threshold
  selection.
- TEST is diagnostic only and is never placed in a TensorFlow dataset or used
  by fit(), evaluate(), predict(), checkpoint selection or threshold tuning.
- ImageNet EfficientNetV2B0 is trained in two phases:
    1. frozen convolutional base;
    2. careful fine-tuning with a very low learning rate.
- Mild augmentation is identical for both classes.
- No horizontal flipping or strong rotation is used because these are webpage
  screenshots containing text and interface elements.
- EfficientNetV2 built-in preprocessing is enabled, so decoded RGB values stay
  in the 0-255 range.

The existing diagnostic TEST split is not a final evaluation dataset. A
separate external dataset is still required for final evaluation.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import csv
import hashlib
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

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

MODEL_DIR = (
    PROJECT_DIR
    / "models"
    / "model1_efficientnetv2b0_top_viewport"
)

REPORT_DIR = (
    PROJECT_DIR
    / "reports"
    / "15_efficientnetv2b0_top_viewport_training"
)

PHASE_1_BEST_PATH = (
    MODEL_DIR
    / "phase1_frozen_best.keras"
)

PHASE_2_BEST_PATH = (
    MODEL_DIR
    / "phase2_finetuned_best.keras"
)

SELECTED_MODEL_PATH = (
    MODEL_DIR
    / "model1_efficientnetv2b0_selected_best.keras"
)

IMAGE_WIDTH = 384
IMAGE_HEIGHT = 160
IMAGE_CHANNELS = 3

EXPECTED_PROCESSING_ACTION = (
    "UNIFORM_WIDTH_SCALE_TOP_VIEWPORT"
)

EXPECTED_TOTAL_IMAGES = 2846

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

CLASS_TO_NUMERIC_LABEL = {
    "NOT_MALICIOUS": 0,
    "MALICIOUS": 1,
}

NUMERIC_TO_CLASS_LABEL = {
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

RANDOM_SEED = 20260826
BATCH_SIZE = 16

PHASE_1_MAX_EPOCHS = 15
PHASE_1_LEARNING_RATE = 1e-3
PHASE_1_EARLY_STOPPING_PATIENCE = 4

PHASE_2_MAX_EPOCHS = 20
PHASE_2_LEARNING_RATE = 1e-5
PHASE_2_EARLY_STOPPING_PATIENCE = 6
FINE_TUNE_LAST_LAYERS = 40

REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 2

HEAD_DENSE_UNITS = 128
HEAD_DROPOUT_1 = 0.35
HEAD_DROPOUT_2 = 0.20
HEAD_L2 = 1e-5

THRESHOLD_MINIMUM = 0.010
THRESHOLD_MAXIMUM = 0.990
THRESHOLD_STEP = 0.001


# =============================================================================
# REPRODUCIBILITY AND GENERAL HELPERS
# =============================================================================

def configure_reproducibility():
    """Set seeds and request deterministic TensorFlow operations."""

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def calculate_sha256(file_path):
    """Calculate SHA256 without modifying a file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


def safe_divide(numerator, denominator):
    """Safely divide two values."""

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def normalise_relative_path(value, row_number):
    """Convert a manifest path into a safe relative Path."""

    text = str(value).strip().replace("\\", "/")

    if not text:
        raise RuntimeError(
            f"Empty prepared path at manifest row {row_number}."
        )

    relative_path = Path(text)

    if relative_path.is_absolute():
        raise RuntimeError(
            "Prepared path must be relative at manifest row "
            f"{row_number}: {text}"
        )

    if any(part in {"", ".", ".."} for part in relative_path.parts):
        raise RuntimeError(
            f"Unsafe prepared path at manifest row "
            f"{row_number}: {text}"
        )

    return relative_path


def ensure_new_output_locations():
    """Refuse to overwrite an earlier or partial training run."""

    existing_paths = [
        path
        for path in [MODEL_DIR, REPORT_DIR]
        if path.exists()
    ]

    if existing_paths:
        formatted_paths = "\n".join(
            f"  - {path}"
            for path in existing_paths
        )

        raise RuntimeError(
            "A training output location already exists. "
            "Nothing was overwritten.\n\n"
            f"{formatted_paths}\n\n"
            "Inspect the existing run first. If it is an incomplete "
            "attempt, move or remove only that specific output folder "
            "before rerunning."
        )


def json_safe(value):
    """Convert NumPy values into JSON-compatible Python values."""

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


def write_json(output_path, data):
    """Write formatted JSON."""

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(data),
            file,
            indent=4,
        )


def format_elapsed(seconds):
    """Format elapsed seconds as HH:MM:SS."""

    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# =============================================================================
# MANIFEST AND IMAGE PREFLIGHT
# =============================================================================

def read_and_validate_manifest():
    """Read the complete top-viewport manifest and validate its splits."""

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "Top-viewport manifest was not found:\n"
            f"{MANIFEST_PATH}"
        )

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)

    missing_columns = (
        REQUIRED_MANIFEST_COLUMNS.difference(fieldnames)
    )

    if missing_columns:
        raise RuntimeError(
            "Manifest is missing columns:\n"
            f"{sorted(missing_columns)}"
        )

    if len(rows) != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_IMAGES} manifest rows, "
            f"but found {len(rows)}."
        )

    split_counts = Counter()
    split_class_counts = Counter()
    seen_paths = set()
    records = []

    for row_number, row in enumerate(rows, start=2):
        split = row["split"].strip().upper()
        class_label = row["class_label"].strip().upper()
        action = row["processing_action"].strip().upper()

        if split not in EXPECTED_SPLIT_COUNTS:
            raise RuntimeError(
                f"Invalid split at manifest row "
                f"{row_number}: {split!r}"
            )

        if class_label not in CLASS_TO_NUMERIC_LABEL:
            raise RuntimeError(
                f"Invalid class at manifest row "
                f"{row_number}: {class_label!r}"
            )

        if action != EXPECTED_PROCESSING_ACTION:
            raise RuntimeError(
                f"Unexpected processing action at manifest "
                f"row {row_number}: {action!r}"
            )

        try:
            numeric_label = int(
                row["numeric_label"].strip()
            )
        except ValueError as error:
            raise RuntimeError(
                f"Invalid numeric label at manifest row "
                f"{row_number}: {row['numeric_label']!r}"
            ) from error

        if (
            numeric_label
            != CLASS_TO_NUMERIC_LABEL[class_label]
        ):
            raise RuntimeError(
                "Class/numeric-label mismatch at manifest "
                f"row {row_number}."
            )

        try:
            declared_width = int(row["output_width"])
            declared_height = int(row["output_height"])
        except ValueError as error:
            raise RuntimeError(
                "Invalid output dimensions at manifest row "
                f"{row_number}."
            ) from error

        if (
            declared_width != IMAGE_WIDTH
            or declared_height != IMAGE_HEIGHT
        ):
            raise RuntimeError(
                "Unexpected declared output dimensions at "
                f"manifest row {row_number}."
            )

        if row["output_colour_mode"].strip().upper() != "RGB":
            raise RuntimeError(
                f"Unexpected colour mode at manifest row "
                f"{row_number}."
            )

        if row["output_format"].strip().upper() != "PNG":
            raise RuntimeError(
                f"Unexpected format at manifest row "
                f"{row_number}."
            )

        relative_path = normalise_relative_path(
            row["prepared_relative_path"],
            row_number,
        )

        path_key = str(relative_path).casefold()

        if path_key in seen_paths:
            raise RuntimeError(
                "Duplicate prepared path in manifest:\n"
                f"{relative_path}"
            )

        seen_paths.add(path_key)

        records.append(
            {
                "manifest_index": len(records) + 1,
                "split": split,
                "class_label": class_label,
                "numeric_label": numeric_label,
                "prepared_relative_path": relative_path,
                "original_relative_path": row[
                    "original_relative_path"
                ],
                "expected_sha256": row[
                    "output_sha256"
                ].strip().lower(),
                "image_path": DATASET_DIR / relative_path,
            }
        )

        split_counts[split] += 1
        split_class_counts[(split, class_label)] += 1

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

    return records


def preflight_images(records):
    """Validate every file and manifest hash before training starts."""

    for index, record in enumerate(records, start=1):
        image_path = record["image_path"]

        if not image_path.is_file():
            raise FileNotFoundError(
                "Prepared screenshot was not found:\n"
                f"{image_path}"
            )

        actual_sha256 = calculate_sha256(image_path)

        if not record["expected_sha256"]:
            raise RuntimeError(
                "Manifest SHA256 is empty for:\n"
                f"{image_path}"
            )

        if actual_sha256 != record["expected_sha256"]:
            raise RuntimeError(
                "Prepared screenshot SHA256 mismatch:\n"
                f"{image_path}\n"
                f"Expected: {record['expected_sha256']}\n"
                f"Actual  : {actual_sha256}"
            )

        try:
            with Image.open(image_path) as image:
                actual_size = image.size
                actual_mode = image.mode
                actual_format = image.format
        except Exception as error:
            raise RuntimeError(
                "Could not read prepared screenshot:\n"
                f"{image_path}\n{error}"
            ) from error

        if actual_size != (IMAGE_WIDTH, IMAGE_HEIGHT):
            raise RuntimeError(
                "Prepared screenshot has wrong dimensions:\n"
                f"{image_path}\n"
                f"Expected: {IMAGE_WIDTH}x{IMAGE_HEIGHT}\n"
                f"Actual  : {actual_size[0]}x{actual_size[1]}"
            )

        if actual_mode != "RGB":
            raise RuntimeError(
                "Prepared screenshot is not RGB:\n"
                f"{image_path}\nMode: {actual_mode}"
            )

        if actual_format != "PNG":
            raise RuntimeError(
                "Prepared screenshot is not PNG:\n"
                f"{image_path}\nFormat: {actual_format}"
            )

        record["actual_sha256"] = actual_sha256

        if index % 250 == 0 or index == len(records):
            print(
                f"Preflight checked {index}/{len(records)}"
            )


def split_records(records):
    """Return TRAIN and VALIDATION only; TEST is intentionally excluded."""

    train_records = [
        record
        for record in records
        if record["split"] == "TRAIN"
    ]

    validation_records = [
        record
        for record in records
        if record["split"] == "VALIDATION"
    ]

    diagnostic_test_count = sum(
        record["split"] == "TEST"
        for record in records
    )

    if len(train_records) != EXPECTED_SPLIT_COUNTS["TRAIN"]:
        raise RuntimeError("Unexpected TRAIN record count.")

    if (
        len(validation_records)
        != EXPECTED_SPLIT_COUNTS["VALIDATION"]
    ):
        raise RuntimeError("Unexpected VALIDATION record count.")

    if diagnostic_test_count != EXPECTED_SPLIT_COUNTS["TEST"]:
        raise RuntimeError("Unexpected diagnostic TEST record count.")

    return train_records, validation_records


# =============================================================================
# TENSORFLOW DATA PIPELINE
# =============================================================================

def decode_png(image_path, numeric_label):
    """Decode an audited 384x160 RGB PNG without additional resizing."""

    image_bytes = tf.io.read_file(image_path)

    image = tf.io.decode_png(
        image_bytes,
        channels=IMAGE_CHANNELS,
    )

    image = tf.ensure_shape(
        image,
        [
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IMAGE_CHANNELS,
        ],
    )

    # EfficientNetV2 built-in preprocessing expects float pixels in [0, 255].
    image = tf.cast(image, tf.float32)
    numeric_label = tf.cast(numeric_label, tf.float32)

    return image, numeric_label


def create_dataset(records, training):
    """Create a deterministic TRAIN or VALIDATION tf.data pipeline."""

    image_paths = [
        str(record["image_path"])
        for record in records
    ]

    labels = [
        record["numeric_label"]
        for record in records
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(records),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        decode_png,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# =============================================================================
# MODEL
# =============================================================================

def build_augmentation():
    """Create mild, class-independent screenshot augmentation."""

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomTranslation(
                height_factor=0.02,
                width_factor=0.02,
                fill_mode="constant",
                fill_value=255.0,
                interpolation="bilinear",
                seed=RANDOM_SEED + 1,
                name="small_translation",
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.03, 0.03),
                width_factor=(-0.03, 0.03),
                fill_mode="constant",
                fill_value=255.0,
                interpolation="bilinear",
                seed=RANDOM_SEED + 2,
                name="small_zoom",
            ),
            tf.keras.layers.RandomContrast(
                factor=0.08,
                seed=RANDOM_SEED + 3,
                name="small_contrast",
            ),
        ],
        name="identical_class_augmentation",
    )


def build_model():
    """Build an ImageNet EfficientNetV2B0 binary classifier."""

    try:
        base_model = (
            tf.keras.applications.EfficientNetV2B0(
                include_top=False,
                weights="imagenet",
                input_shape=(
                    IMAGE_HEIGHT,
                    IMAGE_WIDTH,
                    IMAGE_CHANNELS,
                ),
                pooling="avg",
                include_preprocessing=True,
            )
        )
    except Exception as error:
        raise RuntimeError(
            "EfficientNetV2B0 ImageNet weights could not be loaded. "
            "The first run may need internet access so Keras can "
            "download the official weights. Random weights were not "
            "used as a fallback.\n\n"
            f"Original error: {error}"
        ) from error

    base_model.trainable = False

    inputs = tf.keras.Input(
        shape=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IMAGE_CHANNELS,
        ),
        dtype=tf.float32,
        name="screenshot_rgb_0_255",
    )

    augmentation = build_augmentation()
    x = augmentation(inputs)

    # training=False keeps EfficientNet BatchNormalization statistics stable
    # during both frozen training and later fine-tuning.
    x = base_model(x, training=False)

    x = tf.keras.layers.Dropout(
        HEAD_DROPOUT_1,
        name="head_dropout_1",
    )(x)

    x = tf.keras.layers.Dense(
        HEAD_DENSE_UNITS,
        activation="swish",
        kernel_regularizer=tf.keras.regularizers.L2(
            HEAD_L2
        ),
        name="head_dense",
    )(x)

    x = tf.keras.layers.Dropout(
        HEAD_DROPOUT_2,
        name="head_dropout_2",
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="malicious_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="model1_efficientnetv2b0_top_viewport",
    )

    return model, base_model


def create_metrics():
    """Return new metric objects for each compile call."""

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


def compile_model(model, learning_rate):
    """Compile the binary classifier."""

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=create_metrics(),
    )


def configure_fine_tuning(base_model):
    """Unfreeze only the last non-BatchNorm base layers."""

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

    trainable_base_layers = sum(
        bool(layer.trainable)
        for layer in base_model.layers
    )

    return fine_tune_start, trainable_base_layers


def parameter_counts(model):
    """Count total, trainable and non-trainable parameters."""

    trainable = int(
        sum(
            np.prod(variable.shape)
            for variable in model.trainable_weights
        )
    )

    non_trainable = int(
        sum(
            np.prod(variable.shape)
            for variable in model.non_trainable_weights
        )
    )

    return {
        "total_parameters": trainable + non_trainable,
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
    }


# =============================================================================
# CALLBACKS AND HISTORY
# =============================================================================

def create_callbacks(
    checkpoint_path,
    csv_log_path,
    early_stopping_patience,
    minimum_learning_rate,
):
    """Create phase-specific validation callbacks."""

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=early_stopping_patience,
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
            min_lr=minimum_learning_rate,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=csv_log_path,
            separator=",",
            append=False,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def history_to_rows(history, phase_name):
    """Convert a Keras History into explicit per-epoch rows."""

    metric_names = list(history.history.keys())
    rows = []

    for position, epoch_number in enumerate(history.epoch):
        row = {
            "phase": phase_name,
            "epoch": int(epoch_number) + 1,
        }

        for metric_name in metric_names:
            row[metric_name] = float(
                history.history[metric_name][position]
            )

        rows.append(row)

    return rows


def write_history_csv(output_path, rows):
    """Write combined training history with all available columns."""

    if not rows:
        return

    fieldnames = ["phase", "epoch"]

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(
        output_path,
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


# =============================================================================
# CHECKPOINT EVALUATION AND SELECTION
# =============================================================================

def extract_probabilities(raw_predictions):
    """Convert supported binary model output shapes to probabilities."""

    raw_predictions = np.asarray(raw_predictions)

    if (
        raw_predictions.ndim == 2
        and raw_predictions.shape[1] == 1
    ):
        probabilities = raw_predictions[:, 0]

    elif raw_predictions.ndim == 1:
        probabilities = raw_predictions

    else:
        raise RuntimeError(
            "Unexpected model prediction shape: "
            f"{raw_predictions.shape}"
        )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError(
            "Model predictions contain NaN or infinite values."
        )

    return np.clip(probabilities, 0.0, 1.0)


def predict_checkpoint(checkpoint_path, validation_dataset):
    """Load one checkpoint and predict VALIDATION only."""

    model = tf.keras.models.load_model(
        checkpoint_path,
        compile=False,
    )

    raw_predictions = model.predict(
        validation_dataset,
        verbose=1,
    )

    return extract_probabilities(raw_predictions)


def checkpoint_validation_metrics(true_labels, probabilities):
    """Calculate validation-only checkpoint metrics."""

    predicted_labels = (
        probabilities >= 0.5
    ).astype(np.int32)

    return {
        "average_precision_pr_auc": float(
            average_precision_score(
                true_labels,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
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
            brier_score_loss(
                true_labels,
                probabilities,
            )
        ),
        "accuracy_at_0_5": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
        "f1_at_0_5": float(
            f1_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        "mcc_at_0_5": float(
            matthews_corrcoef(
                true_labels,
                predicted_labels,
            )
        ),
    }


def select_checkpoint(
    phase_1_metrics,
    phase_2_metrics,
):
    """Select a checkpoint using validation AP, ROC-AUC and log loss."""

    candidates = [
        (
            "PHASE_1_FROZEN",
            PHASE_1_BEST_PATH,
            phase_1_metrics,
        ),
        (
            "PHASE_2_FINETUNED",
            PHASE_2_BEST_PATH,
            phase_2_metrics,
        ),
    ]

    selected = max(
        candidates,
        key=lambda item: (
            item[2]["average_precision_pr_auc"],
            item[2]["roc_auc"],
            -item[2]["binary_log_loss"],
            -item[2]["brier_score"],
        ),
    )

    return selected


# =============================================================================
# VALIDATION THRESHOLD SELECTION
# =============================================================================

def metrics_at_threshold(
    true_labels,
    probabilities,
    threshold,
):
    """Calculate complete binary metrics for one validation threshold."""

    predicted_labels = (
        probabilities >= threshold
    ).astype(np.int32)

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

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

    specificity = safe_divide(tn, tn + fp)

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
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
        "false_positive_rate": safe_divide(
            fp,
            fp + tn,
        ),
        "false_negative_rate": safe_divide(
            fn,
            fn + tp,
        ),
        "negative_predictive_value": safe_divide(
            tn,
            tn + fn,
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def search_validation_thresholds(
    true_labels,
    probabilities,
):
    """Search thresholds using VALIDATION predictions only."""

    thresholds = np.arange(
        THRESHOLD_MINIMUM,
        THRESHOLD_MAXIMUM + THRESHOLD_STEP / 2,
        THRESHOLD_STEP,
    )

    threshold_rows = [
        metrics_at_threshold(
            true_labels,
            probabilities,
            float(threshold),
        )
        for threshold in thresholds
    ]

    selected_mcc = max(
        threshold_rows,
        key=lambda row: (
            row["mcc"],
            row["balanced_accuracy"],
            row["f1_score"],
            row["specificity"],
            -abs(row["threshold"] - 0.5),
        ),
    )

    selected_f1 = max(
        threshold_rows,
        key=lambda row: (
            row["f1_score"],
            row["mcc"],
            row["balanced_accuracy"],
            -abs(row["threshold"] - 0.5),
        ),
    )

    selected_balanced_accuracy = max(
        threshold_rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["mcc"],
            row["f1_score"],
            -abs(row["threshold"] - 0.5),
        ),
    )

    recall_90_candidates = [
        row
        for row in threshold_rows
        if row["recall_sensitivity"] >= 0.90
    ]

    selected_recall_90 = None

    if recall_90_candidates:
        selected_recall_90 = max(
            recall_90_candidates,
            key=lambda row: (
                row["specificity"],
                row["precision"],
                row["mcc"],
                row["threshold"],
            ),
        )

    return {
        "all_thresholds": threshold_rows,
        "selected_primary_mcc": selected_mcc,
        "best_f1": selected_f1,
        "best_balanced_accuracy": (
            selected_balanced_accuracy
        ),
        "best_specificity_with_recall_at_least_0_90": (
            selected_recall_90
        ),
    }


def write_threshold_csv(output_path, threshold_rows):
    """Save all validation threshold candidates."""

    fieldnames = list(threshold_rows[0].keys())

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(threshold_rows)


def save_validation_predictions(
    validation_records,
    probabilities,
    threshold,
):
    """Save validation predictions at the selected threshold."""

    fieldnames = [
        "validation_index",
        "prepared_relative_path",
        "original_relative_path",
        "actual_numeric_label",
        "actual_class",
        "malicious_probability",
        "selected_validation_threshold",
        "predicted_numeric_label",
        "predicted_class",
        "correct_prediction",
        "result_type",
    ]

    predicted_labels = (
        probabilities >= threshold
    ).astype(np.int32)

    rows = []

    for index, (
        record,
        probability,
        prediction,
    ) in enumerate(
        zip(
            validation_records,
            probabilities,
            predicted_labels,
        ),
        start=1,
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
                "validation_index": index,
                "prepared_relative_path": str(
                    record["prepared_relative_path"]
                ),
                "original_relative_path": record[
                    "original_relative_path"
                ],
                "actual_numeric_label": actual,
                "actual_class": NUMERIC_TO_CLASS_LABEL[
                    actual
                ],
                "malicious_probability": (
                    f"{float(probability):.10f}"
                ),
                "selected_validation_threshold": (
                    f"{float(threshold):.4f}"
                ),
                "predicted_numeric_label": int(prediction),
                "predicted_class": NUMERIC_TO_CLASS_LABEL[
                    int(prediction)
                ],
                "correct_prediction": (
                    "YES" if actual == prediction else "NO"
                ),
                "result_type": result_type,
            }
        )

    with open(
        REPORT_DIR / "validation_predictions.csv",
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


# =============================================================================
# PLOTS
# =============================================================================

def combine_histories(phase_1_history, phase_2_history):
    """Combine two Keras histories for plotting."""

    combined = {}
    all_keys = set(phase_1_history.history).union(
        phase_2_history.history
    )

    for key in all_keys:
        combined[key] = (
            list(phase_1_history.history.get(key, []))
            + list(phase_2_history.history.get(key, []))
        )

    phase_1_epochs = len(phase_1_history.epoch)

    return combined, phase_1_epochs


def save_training_curves(
    phase_1_history,
    phase_2_history,
):
    """Save loss, PR-AUC, ROC-AUC and accuracy histories."""

    combined, phase_1_epochs = combine_histories(
        phase_1_history,
        phase_2_history,
    )

    total_epochs = max(
        len(combined.get("loss", [])),
        1,
    )

    epochs = np.arange(1, total_epochs + 1)

    plot_specs = [
        ("loss", "Loss", "lower"),
        ("pr_auc", "PR-AUC", "higher"),
        ("roc_auc", "ROC-AUC", "higher"),
        ("accuracy", "Accuracy at 0.5", "higher"),
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    for axis, (metric, title, direction) in zip(
        axes.ravel(),
        plot_specs,
    ):
        training_values = combined.get(metric, [])
        validation_values = combined.get(
            f"val_{metric}",
            [],
        )

        if training_values:
            axis.plot(
                epochs[: len(training_values)],
                training_values,
                marker="o",
                markersize=3,
                label="TRAIN",
            )

        if validation_values:
            axis.plot(
                epochs[: len(validation_values)],
                validation_values,
                marker="o",
                markersize=3,
                label="VALIDATION",
            )

        if phase_1_epochs < total_epochs:
            axis.axvline(
                phase_1_epochs + 0.5,
                linestyle="--",
                color="gray",
                label="Fine-tuning begins",
            )

        axis.set_title(
            f"{title} ({direction} is better)"
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle(
        "EfficientNetV2B0 TRAIN and VALIDATION History",
        fontsize=15,
    )

    figure.tight_layout()

    figure.savefig(
        REPORT_DIR / "training_curves.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_validation_confusion_matrix(metrics):
    """Save the selected-threshold validation confusion matrix."""

    matrix = np.asarray(
        [
            [
                metrics["true_negatives"],
                metrics["false_positives"],
            ],
            [
                metrics["false_negatives"],
                metrics["true_positives"],
            ],
        ]
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set_title(
        "VALIDATION Confusion Matrix at Selected Threshold"
    )
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

    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    colour_threshold = matrix.max() / 2

    for row_index in range(2):
        for column_index in range(2):
            count = matrix[row_index, column_index]
            percentage = (
                percentages[row_index, column_index]
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
        REPORT_DIR / "validation_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_validation_roc_pr_curves(
    true_labels,
    probabilities,
):
    """Save VALIDATION ROC and precision-recall curves."""

    roc_auc = roc_auc_score(true_labels, probabilities)
    average_precision = average_precision_score(
        true_labels,
        probabilities,
    )

    false_positive_rates, true_positive_rates, _ = (
        roc_curve(true_labels, probabilities)
    )

    precision_values, recall_values, _ = (
        precision_recall_curve(
            true_labels,
            probabilities,
        )
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    axes[0].plot(
        false_positive_rates,
        true_positive_rates,
        linewidth=2,
        label=f"ROC-AUC = {roc_auc:.4f}",
    )
    axes[0].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random baseline",
    )
    axes[0].set_title("VALIDATION ROC Curve")
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower right")

    positive_rate = float(np.mean(true_labels))

    axes[1].plot(
        recall_values,
        precision_values,
        linewidth=2,
        label=f"Average precision = {average_precision:.4f}",
    )
    axes[1].axhline(
        positive_rate,
        linestyle="--",
        color="gray",
        label=f"Random baseline = {positive_rate:.4f}",
    )
    axes[1].set_title("VALIDATION Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower left")

    figure.tight_layout()
    figure.savefig(
        REPORT_DIR / "validation_roc_pr_curves.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_threshold_tradeoff_plot(
    threshold_rows,
    selected_threshold,
):
    """Save VALIDATION threshold/metric tradeoffs."""

    thresholds = [
        row["threshold"]
        for row in threshold_rows
    ]

    figure, axis = plt.subplots(figsize=(10, 7))

    for metric_name, label in [
        ("precision", "Precision"),
        ("recall_sensitivity", "Recall"),
        ("specificity", "Specificity"),
        ("f1_score", "F1"),
        ("mcc", "MCC"),
        ("balanced_accuracy", "Balanced accuracy"),
    ]:
        axis.plot(
            thresholds,
            [row[metric_name] for row in threshold_rows],
            label=label,
        )

    axis.axvline(
        selected_threshold,
        linestyle="--",
        linewidth=2,
        color="black",
        label=(
            f"Selected MCC threshold = "
            f"{selected_threshold:.4f}"
        ),
    )

    axis.set_title("VALIDATION Threshold Tradeoffs")
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Metric value")
    axis.set_xlim([0.0, 1.0])
    axis.set_ylim([-0.05, 1.05])
    axis.grid(alpha=0.25)
    axis.legend(loc="best")

    figure.tight_layout()
    figure.savefig(
        REPORT_DIR / "validation_threshold_tradeoffs.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_probability_distribution(
    true_labels,
    probabilities,
    selected_threshold,
):
    """Save VALIDATION probability distributions by class."""

    legitimate_probabilities = probabilities[
        true_labels == 0
    ]
    malicious_probabilities = probabilities[
        true_labels == 1
    ]

    bins = np.linspace(0.0, 1.0, 31)

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.hist(
        legitimate_probabilities,
        bins=bins,
        alpha=0.65,
        color="green",
        label="VALIDATION NOT_MALICIOUS",
    )

    axis.hist(
        malicious_probabilities,
        bins=bins,
        alpha=0.65,
        color="red",
        label="VALIDATION MALICIOUS",
    )

    axis.axvline(
        selected_threshold,
        linestyle="--",
        linewidth=2,
        color="black",
        label=(
            f"Selected threshold = "
            f"{selected_threshold:.4f}"
        ),
    )

    axis.set_title("VALIDATION Malicious-Probability Distribution")
    axis.set_xlabel("Predicted malicious probability")
    axis.set_ylabel("Number of images")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        REPORT_DIR / "validation_probability_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


# =============================================================================
# CONFIGURATION AND SUMMARY REPORTS
# =============================================================================

def save_model_summary(model):
    """Save Keras model architecture text."""

    summary_lines = []
    model.summary(
        print_fn=lambda line: summary_lines.append(line)
    )

    with open(
        REPORT_DIR / "model_architecture.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(summary_lines))


def save_configuration(
    manifest_sha256,
    train_records,
    validation_records,
    frozen_parameter_counts,
    finetuned_parameter_counts,
    fine_tune_start,
    trainable_base_layers,
):
    """Save reproducible training configuration."""

    configuration = {
        "script": "15_train_efficientnetv2b0_top_viewport.py",
        "tensorflow_version": tf.__version__,
        "model_architecture": "EfficientNetV2B0",
        "weights": "imagenet",
        "include_preprocessing": True,
        "input_shape": [
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IMAGE_CHANNELS,
        ],
        "input_pixel_range": [0.0, 255.0],
        "dataset_path": str(DATASET_DIR),
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": manifest_sha256,
        "training_images": len(train_records),
        "validation_images": len(validation_records),
        "diagnostic_test_images_used": 0,
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "class_weights": None,
        "classes_are_balanced": True,
        "augmentation": {
            "applied_to": "TRAIN_ONLY",
            "same_for_both_classes": True,
            "random_translation": 0.02,
            "random_zoom": [-0.03, 0.03],
            "random_contrast": 0.08,
            "horizontal_flip": False,
            "rotation": False,
        },
        "phase_1": {
            "base_frozen": True,
            "max_epochs": PHASE_1_MAX_EPOCHS,
            "learning_rate": PHASE_1_LEARNING_RATE,
            "early_stopping_patience": (
                PHASE_1_EARLY_STOPPING_PATIENCE
            ),
            "parameter_counts": frozen_parameter_counts,
        },
        "phase_2": {
            "fine_tune_last_layers_requested": (
                FINE_TUNE_LAST_LAYERS
            ),
            "fine_tune_start_layer_index": fine_tune_start,
            "trainable_base_layers": trainable_base_layers,
            "batch_normalization_frozen": True,
            "base_called_with_training_false": True,
            "max_epochs": PHASE_2_MAX_EPOCHS,
            "learning_rate": PHASE_2_LEARNING_RATE,
            "early_stopping_patience": (
                PHASE_2_EARLY_STOPPING_PATIENCE
            ),
            "parameter_counts": finetuned_parameter_counts,
        },
        "checkpoint_monitor": "val_pr_auc",
        "checkpoint_selection_data": "VALIDATION_ONLY",
        "checkpoint_primary_metric": (
            "average_precision_pr_auc"
        ),
        "threshold_selection_data": "VALIDATION_ONLY",
        "primary_threshold_objective": "MCC",
        "threshold_minimum": THRESHOLD_MINIMUM,
        "threshold_maximum": THRESHOLD_MAXIMUM,
        "threshold_step": THRESHOLD_STEP,
        "test_split_status": "DIAGNOSTIC_ONLY_NOT_USED",
        "external_dataset_required": True,
    }

    write_json(
        REPORT_DIR / "training_configuration.json",
        configuration,
    )


def save_final_summary(
    manifest_sha256,
    selected_phase,
    selected_source_path,
    selected_model_sha256,
    phase_1_metrics,
    phase_2_metrics,
    threshold_search,
    validation_metrics,
    phase_1_epochs_completed,
    phase_2_epochs_completed,
    elapsed_seconds,
):
    """Save and return the complete final training summary."""

    selected_threshold = validation_metrics["threshold"]

    summary_lines = [
        "=" * 80,
        "EFFICIENTNETV2B0 TOP-VIEWPORT TRAINING COMPLETE",
        "=" * 80,
        "",
        f"TensorFlow             : {tf.__version__}",
        "Architecture          : EfficientNetV2B0",
        "Pretrained weights    : ImageNet",
        (
            "Input shape           : "
            f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}x{IMAGE_CHANNELS}"
        ),
        "Built-in preprocessing: ENABLED (input pixels 0-255)",
        f"Dataset                : {DATASET_DIR}",
        f"Manifest               : {MANIFEST_PATH}",
        f"Manifest SHA256        : {manifest_sha256}",
        "",
        "DATA USAGE",
        "-" * 80,
        f"TRAIN images           : {EXPECTED_SPLIT_COUNTS['TRAIN']}",
        f"VALIDATION images      : {EXPECTED_SPLIT_COUNTS['VALIDATION']}",
        "TEST images used      : 0",
        "TEST status           : DIAGNOSTIC ONLY - NOT LOADED INTO MODEL",
        "",
        "TRAINING",
        "-" * 80,
        (
            "Phase 1 epochs       : "
            f"{phase_1_epochs_completed} (base frozen)"
        ),
        (
            "Phase 2 epochs       : "
            f"{phase_2_epochs_completed} (fine-tuning)"
        ),
        f"Elapsed time          : {format_elapsed(elapsed_seconds)}",
        "Checkpoint monitor    : VALIDATION PR-AUC",
        "",
        "VALIDATION CHECKPOINT COMPARISON",
        "-" * 80,
        (
            "Phase 1 AP / PR-AUC : "
            f"{phase_1_metrics['average_precision_pr_auc']:.6f}"
        ),
        (
            "Phase 1 ROC-AUC     : "
            f"{phase_1_metrics['roc_auc']:.6f}"
        ),
        (
            "Phase 1 log loss    : "
            f"{phase_1_metrics['binary_log_loss']:.6f}"
        ),
        (
            "Phase 2 AP / PR-AUC : "
            f"{phase_2_metrics['average_precision_pr_auc']:.6f}"
        ),
        (
            "Phase 2 ROC-AUC     : "
            f"{phase_2_metrics['roc_auc']:.6f}"
        ),
        (
            "Phase 2 log loss    : "
            f"{phase_2_metrics['binary_log_loss']:.6f}"
        ),
        "",
        f"Selected checkpoint   : {selected_phase}",
        f"Selected source       : {selected_source_path}",
        f"Selected model        : {SELECTED_MODEL_PATH}",
        f"Selected model SHA256 : {selected_model_sha256}",
        "",
        "VALIDATION-SELECTED DECISION THRESHOLD",
        "-" * 80,
        "Primary objective     : Maximum MCC",
        f"Selected threshold    : {selected_threshold:.4f}",
        (
            "Best-F1 threshold   : "
            f"{threshold_search['best_f1']['threshold']:.4f}"
        ),
        (
            "Best-balanced thresh: "
            f"{threshold_search['best_balanced_accuracy']['threshold']:.4f}"
        ),
        "",
        "SELECTED-THRESHOLD VALIDATION RESULTS",
        "-" * 80,
        f"Accuracy              : {validation_metrics['accuracy']:.6f}",
        (
            "Balanced accuracy     : "
            f"{validation_metrics['balanced_accuracy']:.6f}"
        ),
        f"Precision             : {validation_metrics['precision']:.6f}",
        (
            "Recall / sensitivity  : "
            f"{validation_metrics['recall_sensitivity']:.6f}"
        ),
        f"Specificity           : {validation_metrics['specificity']:.6f}",
        f"F1 score              : {validation_metrics['f1_score']:.6f}",
        f"MCC                   : {validation_metrics['mcc']:.6f}",
        (
            "ROC-AUC               : "
            f"{validation_metrics['roc_auc']:.6f}"
        ),
        (
            "Average precision     : "
            f"{validation_metrics['average_precision_pr_auc']:.6f}"
        ),
        "",
        "VALIDATION CONFUSION MATRIX",
        "-" * 80,
        f"TN: {validation_metrics['true_negatives']}",
        f"FP: {validation_metrics['false_positives']}",
        f"FN: {validation_metrics['false_negatives']}",
        f"TP: {validation_metrics['true_positives']}",
        "",
        "IMPORTANT",
        "-" * 80,
        "These are VALIDATION results, not final TEST results.",
        "The selected threshold was chosen using VALIDATION only.",
        "The previous TEST split was not used by the model in this script.",
        "Do not change the threshold using diagnostic TEST results.",
        "The screenshot sources still have known collection differences.",
        "A separate external dataset is required for final evaluation.",
        "=" * 80,
    ]

    summary_text = "\n".join(summary_lines)

    with open(
        REPORT_DIR / "training_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(summary_text)

    return summary_text


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = time.time()

    print("=" * 80)
    print("TRAIN EFFICIENTNETV2B0 ON UNIFORM TOP-VIEWPORT DATASET")
    print("=" * 80)
    print(f"TensorFlow          : {tf.__version__}")
    print(f"Dataset             : {DATASET_DIR}")
    print(f"Manifest            : {MANIFEST_PATH}")
    print(f"Model directory     : {MODEL_DIR}")
    print(f"Report directory    : {REPORT_DIR}")
    print(
        f"Input shape         : "
        f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}x{IMAGE_CHANNELS}"
    )
    print(f"Batch size          : {BATCH_SIZE}")
    print(f"Random seed         : {RANDOM_SEED}")
    print("TEST model usage    : NONE")
    print()

    configure_reproducibility()

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            "Top-viewport dataset was not found:\n"
            f"{DATASET_DIR}"
        )

    ensure_new_output_locations()

    print("Reading and validating manifest...")
    records = read_and_validate_manifest()
    print(f"Manifest records    : {len(records)}")
    print("Manifest counts and split assignments are valid.")
    print()

    print("Running image/hash preflight checks...")
    preflight_images(records)
    print()
    print("All prepared screenshots passed preflight checks.")
    print()

    train_records, validation_records = split_records(
        records
    )

    print("DATA USED BY MODEL")
    print("-" * 80)
    print(f"TRAIN              : {len(train_records)}")
    print(
        f"VALIDATION         : {len(validation_records)}"
    )
    print("TEST               : 0 (diagnostic split excluded)")
    print()

    manifest_sha256 = calculate_sha256(MANIFEST_PATH)

    print("Building ImageNet EfficientNetV2B0 model...")
    print(
        "The first run may download official ImageNet "
        "weights if they are not cached."
    )

    model, base_model = build_model()

    frozen_parameter_counts = parameter_counts(model)

    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=False)

    save_model_summary(model)

    train_dataset = create_dataset(
        train_records,
        training=True,
    )

    validation_dataset = create_dataset(
        validation_records,
        training=False,
    )

    validation_true_labels = np.asarray(
        [
            record["numeric_label"]
            for record in validation_records
        ],
        dtype=np.int32,
    )

    print()
    print("=" * 80)
    print("PHASE 1: TRAIN CLASSIFICATION HEAD WITH FROZEN BASE")
    print("=" * 80)
    print(
        f"Learning rate       : "
        f"{PHASE_1_LEARNING_RATE:.8f}"
    )
    print(f"Maximum epochs      : {PHASE_1_MAX_EPOCHS}")
    print(
        f"Trainable parameters: "
        f"{frozen_parameter_counts['trainable_parameters']:,}"
    )
    print()

    compile_model(
        model,
        learning_rate=PHASE_1_LEARNING_RATE,
    )

    phase_1_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=PHASE_1_MAX_EPOCHS,
        callbacks=create_callbacks(
            checkpoint_path=PHASE_1_BEST_PATH,
            csv_log_path=(
                REPORT_DIR / "phase1_training_log.csv"
            ),
            early_stopping_patience=(
                PHASE_1_EARLY_STOPPING_PATIENCE
            ),
            minimum_learning_rate=1e-5,
        ),
        verbose=1,
    )

    if not PHASE_1_BEST_PATH.is_file():
        raise RuntimeError(
            "Phase 1 best checkpoint was not created."
        )

    phase_1_epochs_completed = len(
        phase_1_history.epoch
    )

    print()
    print("=" * 80)
    print("PHASE 2: CAREFUL FINE-TUNING")
    print("=" * 80)

    fine_tune_start, trainable_base_layers = (
        configure_fine_tuning(base_model)
    )

    compile_model(
        model,
        learning_rate=PHASE_2_LEARNING_RATE,
    )

    finetuned_parameter_counts = parameter_counts(model)

    print(
        f"Fine-tune start     : Base layer "
        f"{fine_tune_start}/{len(base_model.layers)}"
    )
    print(
        f"Trainable base layers: {trainable_base_layers}"
    )
    print("BatchNorm layers    : Frozen / inference mode")
    print(
        f"Learning rate       : "
        f"{PHASE_2_LEARNING_RATE:.8f}"
    )
    print(f"Maximum epochs      : {PHASE_2_MAX_EPOCHS}")
    print(
        f"Trainable parameters: "
        f"{finetuned_parameter_counts['trainable_parameters']:,}"
    )
    print()

    phase_2_final_epoch = (
        phase_1_epochs_completed
        + PHASE_2_MAX_EPOCHS
    )

    phase_2_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        initial_epoch=phase_1_epochs_completed,
        epochs=phase_2_final_epoch,
        callbacks=create_callbacks(
            checkpoint_path=PHASE_2_BEST_PATH,
            csv_log_path=(
                REPORT_DIR / "phase2_training_log.csv"
            ),
            early_stopping_patience=(
                PHASE_2_EARLY_STOPPING_PATIENCE
            ),
            minimum_learning_rate=1e-7,
        ),
        verbose=1,
    )

    if not PHASE_2_BEST_PATH.is_file():
        raise RuntimeError(
            "Phase 2 best checkpoint was not created."
        )

    phase_2_epochs_completed = len(
        phase_2_history.epoch
    )

    print()
    print("=" * 80)
    print("VALIDATION-ONLY CHECKPOINT SELECTION")
    print("=" * 80)
    print()

    print("Predicting VALIDATION with Phase 1 checkpoint...")
    phase_1_probabilities = predict_checkpoint(
        PHASE_1_BEST_PATH,
        validation_dataset,
    )

    print("Predicting VALIDATION with Phase 2 checkpoint...")
    phase_2_probabilities = predict_checkpoint(
        PHASE_2_BEST_PATH,
        validation_dataset,
    )

    if (
        len(phase_1_probabilities)
        != len(validation_true_labels)
        or len(phase_2_probabilities)
        != len(validation_true_labels)
    ):
        raise RuntimeError(
            "Validation prediction count mismatch."
        )

    phase_1_metrics = checkpoint_validation_metrics(
        validation_true_labels,
        phase_1_probabilities,
    )

    phase_2_metrics = checkpoint_validation_metrics(
        validation_true_labels,
        phase_2_probabilities,
    )

    (
        selected_phase,
        selected_source_path,
        selected_checkpoint_metrics,
    ) = select_checkpoint(
        phase_1_metrics,
        phase_2_metrics,
    )

    if selected_phase == "PHASE_1_FROZEN":
        selected_probabilities = phase_1_probabilities
    else:
        selected_probabilities = phase_2_probabilities

    shutil.copy2(
        selected_source_path,
        SELECTED_MODEL_PATH,
    )

    selected_model_sha256 = calculate_sha256(
        SELECTED_MODEL_PATH
    )

    checkpoint_report = {
        "selection_data": "VALIDATION_ONLY",
        "primary_selection_metric": (
            "average_precision_pr_auc"
        ),
        "tie_breakers": [
            "roc_auc",
            "lower_binary_log_loss",
            "lower_brier_score",
        ],
        "phase_1": {
            "checkpoint": str(PHASE_1_BEST_PATH),
            "metrics": phase_1_metrics,
        },
        "phase_2": {
            "checkpoint": str(PHASE_2_BEST_PATH),
            "metrics": phase_2_metrics,
        },
        "selected_phase": selected_phase,
        "selected_source_path": str(
            selected_source_path
        ),
        "selected_model_path": str(
            SELECTED_MODEL_PATH
        ),
        "selected_model_sha256": (
            selected_model_sha256
        ),
        "selected_checkpoint_metrics": (
            selected_checkpoint_metrics
        ),
        "diagnostic_test_used": False,
    }

    write_json(
        REPORT_DIR / "checkpoint_selection.json",
        checkpoint_report,
    )

    print()
    print(f"Selected checkpoint: {selected_phase}")
    print(
        "Selected AP / PR-AUC: "
        f"{selected_checkpoint_metrics['average_precision_pr_auc']:.6f}"
    )
    print(
        "Selected ROC-AUC    : "
        f"{selected_checkpoint_metrics['roc_auc']:.6f}"
    )

    print()
    print("=" * 80)
    print("VALIDATION-ONLY THRESHOLD SELECTION")
    print("=" * 80)
    print()

    threshold_search = search_validation_thresholds(
        validation_true_labels,
        selected_probabilities,
    )

    selected_threshold_metrics = threshold_search[
        "selected_primary_mcc"
    ]

    selected_threshold = selected_threshold_metrics[
        "threshold"
    ]

    validation_metrics = dict(
        selected_threshold_metrics
    )

    validation_metrics.update(
        {
            "evaluation_type": (
                "VALIDATION_SELECTED_THRESHOLD"
            ),
            "selected_checkpoint": selected_phase,
            "roc_auc": float(
                roc_auc_score(
                    validation_true_labels,
                    selected_probabilities,
                )
            ),
            "average_precision_pr_auc": float(
                average_precision_score(
                    validation_true_labels,
                    selected_probabilities,
                )
            ),
            "binary_log_loss": float(
                log_loss(
                    validation_true_labels,
                    selected_probabilities,
                    labels=[0, 1],
                )
            ),
            "brier_score": float(
                brier_score_loss(
                    validation_true_labels,
                    selected_probabilities,
                )
            ),
            "validation_images": int(
                len(validation_true_labels)
            ),
            "selected_model_path": str(
                SELECTED_MODEL_PATH
            ),
            "selected_model_sha256": (
                selected_model_sha256
            ),
            "threshold_selection_objective": "MCC",
            "threshold_selection_data": "VALIDATION_ONLY",
            "diagnostic_test_used": False,
        }
    )

    write_threshold_csv(
        REPORT_DIR / "validation_threshold_search.csv",
        threshold_search["all_thresholds"],
    )

    write_json(
        REPORT_DIR / "validation_threshold_selection.json",
        {
            key: value
            for key, value in threshold_search.items()
            if key != "all_thresholds"
        },
    )

    write_json(
        REPORT_DIR / "validation_metrics.json",
        validation_metrics,
    )

    save_validation_predictions(
        validation_records=validation_records,
        probabilities=selected_probabilities,
        threshold=selected_threshold,
    )

    phase_1_history_rows = history_to_rows(
        phase_1_history,
        "PHASE_1_FROZEN",
    )

    phase_2_history_rows = history_to_rows(
        phase_2_history,
        "PHASE_2_FINETUNED",
    )

    write_history_csv(
        REPORT_DIR / "combined_training_history.csv",
        phase_1_history_rows + phase_2_history_rows,
    )

    save_configuration(
        manifest_sha256=manifest_sha256,
        train_records=train_records,
        validation_records=validation_records,
        frozen_parameter_counts=frozen_parameter_counts,
        finetuned_parameter_counts=(
            finetuned_parameter_counts
        ),
        fine_tune_start=fine_tune_start,
        trainable_base_layers=trainable_base_layers,
    )

    print()
    print("Creating training and validation plots...")

    save_training_curves(
        phase_1_history,
        phase_2_history,
    )

    save_validation_confusion_matrix(
        validation_metrics
    )

    save_validation_roc_pr_curves(
        validation_true_labels,
        selected_probabilities,
    )

    save_threshold_tradeoff_plot(
        threshold_search["all_thresholds"],
        selected_threshold,
    )

    save_probability_distribution(
        validation_true_labels,
        selected_probabilities,
        selected_threshold,
    )

    elapsed_seconds = time.time() - start_time

    summary_text = save_final_summary(
        manifest_sha256=manifest_sha256,
        selected_phase=selected_phase,
        selected_source_path=selected_source_path,
        selected_model_sha256=selected_model_sha256,
        phase_1_metrics=phase_1_metrics,
        phase_2_metrics=phase_2_metrics,
        threshold_search=threshold_search,
        validation_metrics=validation_metrics,
        phase_1_epochs_completed=phase_1_epochs_completed,
        phase_2_epochs_completed=phase_2_epochs_completed,
        elapsed_seconds=elapsed_seconds,
    )

    print()
    print(summary_text)
    print()
    print(f"Selected model saved to: {SELECTED_MODEL_PATH}")
    print(f"Reports saved to       : {REPORT_DIR}")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("TRAINING INTERRUPTED BY USER")
        print("=" * 80)
        print("Existing dataset images were not modified.")
        print("The diagnostic TEST split was not evaluated.")
        sys.exit(130)

    except Exception as error:
        print()
        print("=" * 80)
        print("EFFICIENTNETV2B0 TRAINING FAILED")
        print("=" * 80)
        print(error)
        print()
        print("Existing dataset images were not modified.")
        print("The diagnostic TEST split was not evaluated.")
        sys.exit(1)
