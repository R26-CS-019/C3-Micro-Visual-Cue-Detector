from __future__ import annotations

import csv
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\oshan\Desktop\model 1 login pp2"
)

PREPARED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "prepared_login_384x160"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "05_dataset_split"
)

TRAIN_MANIFEST = (
    SPLIT_DIR / "train_manifest.csv"
)

VALIDATION_MANIFEST = (
    SPLIT_DIR / "validation_manifest.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "model1_mobilenetv3large"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "06_mobilenetv3large_training"
)

HEAD_CHECKPOINT = (
    MODEL_DIR / "phase1_head_best.keras"
)

FINETUNE_CHECKPOINT = (
    MODEL_DIR / "phase2_finetuned_best.keras"
)

SELECTED_MODEL = (
    MODEL_DIR / "model1_selected_best.keras"
)

SEED = 20260826

HEIGHT = 160
WIDTH = 384
CHANNELS = 3

BATCH_SIZE = 16

HEAD_EPOCHS = 15
FINETUNE_EPOCHS = 20

HEAD_LEARNING_RATE = 1e-3
FINETUNE_LEARNING_RATE = 1e-5

DROPOUT_RATE = 0.30


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

def read_manifest(
    manifest_path: Path,
) -> list[dict[str, str]]:
    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        return list(
            csv.DictReader(csv_file)
        )


def get_paths_and_labels(
    rows: list[dict[str, str]],
) -> tuple[list[str], np.ndarray]:
    image_paths = [
        str(
            PREPARED_ROOT
            / row["prepared_relative_path"]
        )
        for row in rows
    ]

    labels = np.asarray(
        [
            int(row["numeric_label"])
            for row in rows
        ],
        dtype=np.float32,
    )

    missing_files = [
        image_path
        for image_path in image_paths
        if not Path(image_path).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Some prepared images are missing:\n"
            + "\n".join(missing_files[:20])
        )

    return image_paths, labels


def decode_image(
    image_path: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(
        image_path
    )

    image = tf.io.decode_png(
        image_bytes,
        channels=CHANNELS,
    )

    image = tf.ensure_shape(
        image,
        [HEIGHT, WIDTH, CHANNELS],
    )

    label = tf.cast(
        label,
        tf.float32,
    )

    return image, label


def make_dataset(
    image_paths: list[str],
    labels: np.ndarray,
    training: bool,
) -> tf.data.Dataset:
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (image_paths, labels)
        )
    )

    options = tf.data.Options()
    options.experimental_deterministic = True

    dataset = dataset.with_options(
        options
    )

    dataset = dataset.map(
        decode_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    # Images stay as uint8 inside the cache.
    dataset = dataset.cache()

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=SEED,
            reshuffle_each_iteration=True,
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
# MODEL FUNCTIONS
# =============================================================================

def create_metrics() -> list:
    return [
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        ),
        tf.keras.metrics.AUC(
            name="roc_auc",
            curve="ROC",
        ),
        tf.keras.metrics.AUC(
            name="pr_auc",
            curve="PR",
        ),
    ]


def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss=(
            tf.keras.losses
            .BinaryCrossentropy()
        ),
        metrics=create_metrics(),
    )


def create_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomTranslation(
                height_factor=0.06,
                width_factor=0.03,
                fill_mode="constant",
                fill_value=255.0,
                seed=SEED + 1,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.06, 0.06),
                width_factor=(-0.04, 0.04),
                fill_mode="constant",
                fill_value=255.0,
                seed=SEED + 2,
            ),
            tf.keras.layers.RandomContrast(
                factor=0.10,
                seed=SEED + 3,
            ),
        ],
        name="webpage_augmentation",
    )


def build_model() -> tuple[
    tf.keras.Model,
    tf.keras.Model,
]:
    augmentation = create_augmentation()

    backbone = (
        tf.keras.applications
        .MobileNetV3Large(
            input_shape=(
                HEIGHT,
                WIDTH,
                CHANNELS,
            ),
            include_top=False,
            weights="imagenet",
            include_preprocessing=True,
        )
    )

    backbone.trainable = False

    inputs = tf.keras.Input(
        shape=(
            HEIGHT,
            WIDTH,
            CHANNELS,
        ),
        name="login_screenshot",
    )

    x = augmentation(inputs)

    x = backbone(
        x,
        training=False,
    )

    x = (
        tf.keras.layers
        .GlobalAveragePooling2D(
            name="global_average_pooling"
        )(x)
    )

    x = tf.keras.layers.Dropout(
        DROPOUT_RATE,
        seed=SEED + 4,
        name="classifier_dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="phishing_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="model1_login_specialist",
    )

    return model, backbone


def create_callbacks(
    phase_name: str,
    checkpoint_path: Path,
    patience: int,
) -> list:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=patience,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.30,
            patience=2,
            min_delta=1e-4,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(
                REPORT_DIR
                / f"{phase_name}_history.csv"
            ),
            append=False,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


# =============================================================================
# REPORT FUNCTIONS
# =============================================================================

def history_to_rows(
    history,
    phase_name: str,
) -> list[dict[str, object]]:
    output_rows = []

    for position, epoch in enumerate(
        history.epoch
    ):
        row = {
            "phase": phase_name,
            "epoch": int(epoch) + 1,
        }

        for metric_name, values in (
            history.history.items()
        ):
            row[metric_name] = float(
                values[position]
            )

        output_rows.append(row)

    return output_rows


def write_dictionary_rows(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return

    fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def predict_probabilities(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> np.ndarray:
    probabilities = model.predict(
        dataset,
        verbose=1,
    )

    return (
        probabilities
        .reshape(-1)
        .astype(np.float64)
    )


def calculate_threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "specificity": float(
            specificity
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                labels,
                predictions,
            )
        ),
        "fpr": float(
            false_positive_rate
        ),
        "fnr": float(
            false_negative_rate
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_validation_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[
    list[dict[str, float | int]],
    dict[str, float | int],
]:
    threshold_rows = [
        calculate_threshold_metrics(
            labels,
            probabilities,
            float(threshold),
        )
        for threshold in np.arange(
            0.05,
            0.951,
            0.01,
        )
    ]

    recall_qualified = [
        row
        for row in threshold_rows
        if float(row["recall"]) >= 0.90
    ]

    candidates = (
        recall_qualified
        if recall_qualified
        else threshold_rows
    )

    selected = max(
        candidates,
        key=lambda row: (
            float(row["mcc"]),
            float(row["f1"]),
            float(row["specificity"]),
            -abs(
                float(row["threshold"])
                - 0.5
            ),
        ),
    )

    return threshold_rows, selected


def save_predictions(
    validation_rows: list[dict[str, str]],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    output_rows = []

    for index, row in enumerate(
        validation_rows
    ):
        predicted_label = int(
            probabilities[index] >= threshold
        )

        true_label = int(
            labels[index]
        )

        output_rows.append(
            {
                "prepared_relative_path": (
                    row[
                        "prepared_relative_path"
                    ]
                ),
                "original_relative_path": (
                    row[
                        "original_relative_path"
                    ]
                ),
                "class_label": (
                    row["class_label"]
                ),
                "true_label": true_label,
                "phishing_probability": (
                    f"{probabilities[index]:.10f}"
                ),
                "threshold": (
                    f"{threshold:.4f}"
                ),
                "predicted_label": (
                    predicted_label
                ),
                "predicted_class": (
                    "MALICIOUS"
                    if predicted_label == 1
                    else "NOT_MALICIOUS"
                ),
                "correct": (
                    predicted_label
                    == true_label
                ),
                "viewport_action": (
                    row["viewport_action"]
                ),
            }
        )

    write_dictionary_rows(
        REPORT_DIR
        / "validation_predictions.csv",
        output_rows,
    )


def save_training_curves(
    rows: list[dict[str, object]],
) -> None:
    epochs = [
        int(row["epoch"])
        for row in rows
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8),
    )

    plot_settings = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("pr_auc", "PR-AUC"),
        ("roc_auc", "ROC-AUC"),
    ]

    for axis, (
        metric_name,
        title,
    ) in zip(
        axes.flat,
        plot_settings,
    ):
        training_values = [
            row.get(metric_name, np.nan)
            for row in rows
        ]

        validation_values = [
            row.get(
                f"val_{metric_name}",
                np.nan,
            )
            for row in rows
        ]

        axis.plot(
            epochs,
            training_values,
            label="Train",
        )

        axis.plot(
            epochs,
            validation_values,
            label="Validation",
        )

        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()

    figure.tight_layout()

    figure.savefig(
        REPORT_DIR
        / "training_curves.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_confusion_matrix_plot(
    metrics: dict[str, float | int],
) -> None:
    matrix = np.asarray(
        [
            [
                metrics["tn"],
                metrics["fp"],
            ],
            [
                metrics["fn"],
                metrics["tp"],
            ],
        ],
        dtype=np.int64,
    )

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    axis.imshow(
        matrix,
        cmap="Blues",
    )

    axis.set_xticks(
        [0, 1],
        ["Not malicious", "Malicious"],
    )

    axis.set_yticks(
        [0, 1],
        ["Not malicious", "Malicious"],
    )

    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")

    axis.set_title(
        "Validation Confusion Matrix\n"
        f"Threshold = "
        f"{float(metrics['threshold']):.2f}"
    )

    for row_index in range(2):
        for column_index in range(2):
            axis.text(
                column_index,
                row_index,
                str(
                    matrix[
                        row_index,
                        column_index,
                    ]
                ),
                ha="center",
                va="center",
                fontsize=14,
            )

    figure.tight_layout()

    figure.savefig(
        REPORT_DIR
        / "validation_confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


# =============================================================================
# MAIN TRAINING WORKFLOW
# =============================================================================

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    tf.keras.utils.set_random_seed(
        SEED
    )

    try:
        (
            tf.config.experimental
            .enable_op_determinism()
        )
    except Exception as error:
        print(
            f"Determinism warning: {error}"
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "MODEL 1 MOBILENETV3LARGE TRAINING"
    )
    print("=" * 80)

    print(
        f"TensorFlow : {tf.__version__}"
    )

    print(
        f"Input      : "
        f"{HEIGHT}x{WIDTH}x{CHANNELS}"
    )

    print(
        f"Batch size : {BATCH_SIZE}"
    )

    print(
        "GPU count  : "
        f"{len(tf.config.list_physical_devices('GPU'))}"
    )

    print()

    if not TRAIN_MANIFEST.exists():
        raise FileNotFoundError(
            f"Train manifest not found:\n"
            f"{TRAIN_MANIFEST}"
        )

    if not VALIDATION_MANIFEST.exists():
        raise FileNotFoundError(
            "Validation manifest not found:\n"
            f"{VALIDATION_MANIFEST}"
        )

    train_rows = read_manifest(
        TRAIN_MANIFEST
    )

    validation_rows = read_manifest(
        VALIDATION_MANIFEST
    )

    train_paths, train_labels = (
        get_paths_and_labels(train_rows)
    )

    (
        validation_paths,
        validation_labels,
    ) = get_paths_and_labels(
        validation_rows
    )

    train_counts = Counter(
        train_labels.astype(int)
    )

    validation_counts = Counter(
        validation_labels.astype(int)
    )

    print(
        f"Train total          : "
        f"{len(train_rows)}"
    )

    print(
        f"Train legitimate     : "
        f"{train_counts[0]}"
    )

    print(
        f"Train malicious      : "
        f"{train_counts[1]}"
    )

    print(
        f"Validation total     : "
        f"{len(validation_rows)}"
    )

    print(
        f"Validation legitimate: "
        f"{validation_counts[0]}"
    )

    print(
        f"Validation malicious : "
        f"{validation_counts[1]}"
    )

    print(
        "TEST DATA            : NOT USED"
    )

    print()

    train_dataset = make_dataset(
        train_paths,
        train_labels,
        training=True,
    )

    validation_dataset = make_dataset(
        validation_paths,
        validation_labels,
        training=False,
    )

    model, backbone = build_model()

    with (
        REPORT_DIR / "model_summary.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        model.summary(
            print_fn=lambda line: (
                summary_file.write(
                    line + "\n"
                )
            ),
            expand_nested=True,
        )

    training_config = {
        "architecture": (
            "MobileNetV3Large"
        ),
        "seed": SEED,
        "input_shape": [
            HEIGHT,
            WIDTH,
            CHANNELS,
        ],
        "batch_size": BATCH_SIZE,
        "train_images": len(train_rows),
        "validation_images": (
            len(validation_rows)
        ),
        "head_epochs": HEAD_EPOCHS,
        "finetune_epochs": (
            FINETUNE_EPOCHS
        ),
        "head_learning_rate": (
            HEAD_LEARNING_RATE
        ),
        "finetune_learning_rate": (
            FINETUNE_LEARNING_RATE
        ),
        "test_data_used": False,
        "tensorflow_version": (
            tf.__version__
        ),
    }

    (
        REPORT_DIR / "training_config.json"
    ).write_text(
        json.dumps(
            training_config,
            indent=2,
        ),
        encoding="utf-8",
    )

    training_start = time.time()

    # =========================================================================
    # PHASE 1
    # =========================================================================

    print("=" * 80)
    print(
        "PHASE 1: TRAIN CLASSIFICATION HEAD"
    )
    print("=" * 80)

    compile_model(
        model,
        HEAD_LEARNING_RATE,
    )

    head_history = model.fit(
        train_dataset,
        validation_data=(
            validation_dataset
        ),
        epochs=HEAD_EPOCHS,
        callbacks=create_callbacks(
            phase_name="phase1_head",
            checkpoint_path=(
                HEAD_CHECKPOINT
            ),
            patience=5,
        ),
        verbose=1,
    )

    # =========================================================================
    # PHASE 2
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "PHASE 2: FINE-TUNE UPPER 25% "
        "OF BACKBONE"
    )
    print("=" * 80)

    backbone.trainable = True

    unfreeze_from = int(
        len(backbone.layers) * 0.75
    )

    trainable_backbone_layers = 0

    for layer_index, layer in enumerate(
        backbone.layers
    ):
        layer.trainable = (
            layer_index >= unfreeze_from
            and not isinstance(
                layer,
                tf.keras.layers
                .BatchNormalization,
            )
        )

        if layer.trainable:
            trainable_backbone_layers += 1

    print(
        f"Backbone layers       : "
        f"{len(backbone.layers)}"
    )

    print(
        f"Fine-tuning from layer: "
        f"{unfreeze_from}"
    )

    print(
        f"Trainable upper layers: "
        f"{trainable_backbone_layers}"
    )

    compile_model(
        model,
        FINETUNE_LEARNING_RATE,
    )

    completed_head_epochs = len(
        head_history.epoch
    )

    finetune_history = model.fit(
        train_dataset,
        validation_data=(
            validation_dataset
        ),
        initial_epoch=(
            completed_head_epochs
        ),
        epochs=(
            completed_head_epochs
            + FINETUNE_EPOCHS
        ),
        callbacks=create_callbacks(
            phase_name=(
                "phase2_finetune"
            ),
            checkpoint_path=(
                FINETUNE_CHECKPOINT
            ),
            patience=7,
        ),
        verbose=1,
    )

    training_time = (
        time.time() - training_start
    )

    # =========================================================================
    # COMBINE TRAINING HISTORY
    # =========================================================================

    combined_history = (
        history_to_rows(
            head_history,
            "PHASE_1_HEAD",
        )
        + history_to_rows(
            finetune_history,
            "PHASE_2_FINETUNE",
        )
    )

    write_dictionary_rows(
        REPORT_DIR
        / "combined_training_history.csv",
        combined_history,
    )

    save_training_curves(
        combined_history
    )

    # =========================================================================
    # SELECT BEST CHECKPOINT
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "SELECT BEST CHECKPOINT USING "
        "VALIDATION DATA"
    )
    print("=" * 80)

    if not HEAD_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Head checkpoint missing:\n"
            f"{HEAD_CHECKPOINT}"
        )

    if not FINETUNE_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Fine-tuned checkpoint missing:\n"
            f"{FINETUNE_CHECKPOINT}"
        )

    head_model = (
        tf.keras.models.load_model(
            HEAD_CHECKPOINT
        )
    )

    finetuned_model = (
        tf.keras.models.load_model(
            FINETUNE_CHECKPOINT
        )
    )

    head_probabilities = (
        predict_probabilities(
            head_model,
            validation_dataset,
        )
    )

    finetuned_probabilities = (
        predict_probabilities(
            finetuned_model,
            validation_dataset,
        )
    )

    head_average_precision = (
        average_precision_score(
            validation_labels,
            head_probabilities,
        )
    )

    finetuned_average_precision = (
        average_precision_score(
            validation_labels,
            finetuned_probabilities,
        )
    )

    print(
        "Phase 1 average precision: "
        f"{head_average_precision:.6f}"
    )

    print(
        "Phase 2 average precision: "
        f"{finetuned_average_precision:.6f}"
    )

    if (
        finetuned_average_precision
        >= head_average_precision
    ):
        selected_model = (
            finetuned_model
        )

        selected_probabilities = (
            finetuned_probabilities
        )

        selected_checkpoint_name = (
            "PHASE_2_FINETUNED"
        )

        selected_average_precision = (
            finetuned_average_precision
        )

    else:
        selected_model = head_model

        selected_probabilities = (
            head_probabilities
        )

        selected_checkpoint_name = (
            "PHASE_1_HEAD"
        )

        selected_average_precision = (
            head_average_precision
        )

    selected_roc_auc = roc_auc_score(
        validation_labels,
        selected_probabilities,
    )

    selected_model.save(
        SELECTED_MODEL
    )

    # =========================================================================
    # SELECT VALIDATION THRESHOLD
    # =========================================================================

    (
        threshold_rows,
        selected_threshold_metrics,
    ) = choose_validation_threshold(
        validation_labels,
        selected_probabilities,
    )

    selected_threshold = float(
        selected_threshold_metrics[
            "threshold"
        ]
    )

    write_dictionary_rows(
        REPORT_DIR
        / "validation_threshold_search.csv",
        threshold_rows,
    )

    save_predictions(
        validation_rows,
        validation_labels,
        selected_probabilities,
        selected_threshold,
    )

    save_confusion_matrix_plot(
        selected_threshold_metrics
    )

    threshold_report = {
        "selected_checkpoint": (
            selected_checkpoint_name
        ),
        "selected_threshold": (
            selected_threshold
        ),
        "selection_rule": (
            "Maximum MCC among thresholds "
            "with validation recall >= 0.90"
        ),
        "validation_metrics": (
            selected_threshold_metrics
        ),
        "validation_roc_auc": float(
            selected_roc_auc
        ),
        "validation_average_precision": (
            float(
                selected_average_precision
            )
        ),
        "test_data_used": False,
    }

    (
        REPORT_DIR
        / "selected_validation_threshold.json"
    ).write_text(
        json.dumps(
            threshold_report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    summary_lines = [
        "=" * 80,
        (
            "MODEL 1 MOBILENETV3LARGE "
            "TRAINING COMPLETE"
        ),
        "=" * 80,
        "",
        (
            "Selected checkpoint : "
            f"{selected_checkpoint_name}"
        ),
        (
            "Selected model      : "
            f"{SELECTED_MODEL}"
        ),
        (
            "Decision threshold  : "
            f"{selected_threshold:.4f}"
        ),
        "",
        "VALIDATION RESULTS",
        "-" * 80,
        (
            "Accuracy            : "
            f"{float(selected_threshold_metrics['accuracy']):.6f}"
        ),
        (
            "Precision           : "
            f"{float(selected_threshold_metrics['precision']):.6f}"
        ),
        (
            "Recall              : "
            f"{float(selected_threshold_metrics['recall']):.6f}"
        ),
        (
            "Specificity         : "
            f"{float(selected_threshold_metrics['specificity']):.6f}"
        ),
        (
            "F1                  : "
            f"{float(selected_threshold_metrics['f1']):.6f}"
        ),
        (
            "MCC                 : "
            f"{float(selected_threshold_metrics['mcc']):.6f}"
        ),
        (
            "ROC-AUC             : "
            f"{selected_roc_auc:.6f}"
        ),
        (
            "Average precision   : "
            f"{selected_average_precision:.6f}"
        ),
        "",
        "CONFUSION MATRIX",
        "-" * 80,
        (
            "TN: "
            f"{int(selected_threshold_metrics['tn'])}"
        ),
        (
            "FP: "
            f"{int(selected_threshold_metrics['fp'])}"
        ),
        (
            "FN: "
            f"{int(selected_threshold_metrics['fn'])}"
        ),
        (
            "TP: "
            f"{int(selected_threshold_metrics['tp'])}"
        ),
        "",
        (
            "Training time       : "
            f"{training_time:.2f} seconds"
        ),
        "",
        "TEST DATA WAS NOT USED.",
        (
            "The next script will evaluate "
            "the locked model and threshold "
            "on the untouched test set."
        ),
        "=" * 80,
    ]

    summary_text = "\n".join(
        summary_lines
    )

    (
        REPORT_DIR / "training_summary.txt"
    ).write_text(
        summary_text,
        encoding="utf-8",
    )

    print()
    print(summary_text)
    print()

    print(
        f"Reports saved to: "
        f"{REPORT_DIR}"
    )


if __name__ == "__main__":
    main()