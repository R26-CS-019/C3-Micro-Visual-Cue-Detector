from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# =============================================================================
# PROJECT SETTINGS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\oshan\Desktop\new model 2 pp2"
)

SPLIT_FOLDER = (
    PROJECT_ROOT
    / "reports"
    / "balanced_split"
)

TRAIN_MANIFEST = (
    SPLIT_FOLDER
    / "phish360_train_manifest.csv"
)

VALIDATION_MANIFEST = (
    SPLIT_FOLDER
    / "phish360_validation_manifest.csv"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "models"
    / "model2_effnetv2b0"
    / "stage1"
)

REPORT_FOLDER = (
    PROJECT_ROOT
    / "reports"
    / "model2_effnetv2b0"
    / "stage1"
)

BEST_MODEL_PATH = (
    OUTPUT_FOLDER
    / "model2_effnetv2b0_stage1_best.keras"
)

FINAL_MODEL_PATH = (
    OUTPUT_FOLDER
    / "model2_effnetv2b0_stage1_final.keras"
)

TRAINING_LOG = (
    REPORT_FOLDER
    / "stage1_training_log.csv"
)

TRAINING_PLOT = (
    REPORT_FOLDER
    / "stage1_training_history.png"
)

AUGMENTATION_SPOTCHECK = (
    REPORT_FOLDER
    / "stage1_augmentation_spotcheck.png"
)

MODEL_SUMMARY_FILE = (
    REPORT_FOLDER
    / "stage1_model_summary.txt"
)

RESULTS_FILE = (
    REPORT_FOLDER
    / "stage1_validation_results.json"
)


# The wide shape preserves webpage layout better than a square resize.
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 384
CHANNELS = 3

BATCH_SIZE = 16
MAX_EPOCHS = 20
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

EXPECTED_TRAIN_IMAGES = 4_868
EXPECTED_VALIDATION_IMAGES = 1_216
EXPECTED_IMAGES_PER_TRAIN_CLASS = 2_434
EXPECTED_IMAGES_PER_VALIDATION_CLASS = 608

LABEL_TO_NUMBER = {
    "LEGITIMATE": 0,
    "PHISHING": 1,
}


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# =============================================================================
# CSV HELPERS
# =============================================================================

def normalize_header(value: str) -> str:
    return "".join(
        character
        for character in str(value).lower()
        if character.isalnum()
    )


def find_column(
    fieldnames: list[str],
    possible_names: list[str],
    description: str,
) -> str:
    normalized = {
        normalize_header(name): name
        for name in fieldnames
    }

    for possible_name in possible_names:
        key = normalize_header(possible_name)

        if key in normalized:
            return normalized[key]

    raise SystemExit(
        f"\nERROR: Could not find the {description} column.\n"
        f"Available columns:\n{fieldnames}"
    )


def load_manifest(
    manifest_path: Path,
    expected_total: int,
    expected_per_class: int,
    manifest_name: str,
) -> tuple[list[str], list[int], dict[str, int]]:
    if not manifest_path.is_file():
        raise SystemExit(
            f"\nERROR: {manifest_name} manifest was not found:\n"
            f"{manifest_path}"
        )

    with manifest_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise SystemExit(
                f"\nERROR: {manifest_name} manifest has no header."
            )

        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    path_column = find_column(
        fieldnames,
        [
            "screenshot_path",
            "image_path",
            "file_path",
            "path",
        ],
        "screenshot path",
    )

    label_column = find_column(
        fieldnames,
        [
            "label",
            "class",
            "class_label",
            "dataset_label",
        ],
        "class label",
    )

    if len(rows) != expected_total:
        raise SystemExit(
            f"\nERROR: Unexpected {manifest_name} image count.\n"
            f"Expected: {expected_total:,}\n"
            f"Found: {len(rows):,}"
        )

    paths: list[str] = []
    labels: list[int] = []
    class_counts = {
        "LEGITIMATE": 0,
        "PHISHING": 0,
    }

    missing_files: list[str] = []
    seen_paths: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        image_path = row[path_column].strip()
        label_name = row[label_column].strip().upper()

        if label_name not in LABEL_TO_NUMBER:
            raise SystemExit(
                f"\nERROR: Invalid label on {manifest_name} "
                f"row {row_number}: {label_name}"
            )

        if not image_path:
            raise SystemExit(
                f"\nERROR: Empty image path on {manifest_name} "
                f"row {row_number}."
            )

        normalized_path = os.path.normcase(
            os.path.normpath(image_path)
        )

        if normalized_path in seen_paths:
            raise SystemExit(
                f"\nERROR: Duplicate path inside {manifest_name}:\n"
                f"{image_path}"
            )

        seen_paths.add(normalized_path)

        if not Path(image_path).is_file():
            missing_files.append(image_path)

            if len(missing_files) >= 20:
                break

        paths.append(image_path)
        labels.append(LABEL_TO_NUMBER[label_name])
        class_counts[label_name] += 1

    if missing_files:
        missing_text = "\n".join(missing_files)

        raise SystemExit(
            f"\nERROR: Missing {manifest_name} screenshots "
            f"(showing up to 20):\n{missing_text}"
        )

    for label_name in ["LEGITIMATE", "PHISHING"]:
        if class_counts[label_name] != expected_per_class:
            raise SystemExit(
                f"\nERROR: Unexpected {manifest_name} "
                f"{label_name} count.\n"
                f"Expected: {expected_per_class:,}\n"
                f"Found: {class_counts[label_name]:,}"
            )

    return paths, labels, class_counts


# =============================================================================
# IMAGE PIPELINE
# =============================================================================

def resize_with_white_padding(
    image: tf.Tensor,
) -> tf.Tensor:
    """Resize without distortion and use white webpage-style padding."""
    image = tf.cast(image, tf.float32)

    original_shape = tf.cast(
        tf.shape(image)[:2],
        tf.float32,
    )

    original_height = original_shape[0]
    original_width = original_shape[1]

    scale = tf.minimum(
        IMAGE_HEIGHT / original_height,
        IMAGE_WIDTH / original_width,
    )

    new_height = tf.maximum(
        1,
        tf.cast(
            tf.round(original_height * scale),
            tf.int32,
        ),
    )

    new_width = tf.maximum(
        1,
        tf.cast(
            tf.round(original_width * scale),
            tf.int32,
        ),
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

    image.set_shape(
        [IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS]
    )

    return image


def load_and_prepare_image(
    image_path: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image_bytes,
        channels=CHANNELS,
        expand_animations=False,
    )

    image.set_shape([None, None, CHANNELS])
    image = resize_with_white_padding(image)

    label = tf.cast(label, tf.float32)

    return image, label


def create_dataset(
    paths: list[str],
    labels: list[int],
    training: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 4_000),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_and_prepare_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
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
# MODEL
# =============================================================================

def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
        name="screenshot",
    )

    # Webpages must not be horizontally flipped because that would mirror
    # logos and text. Only small viewport and colour variations are used.
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomTranslation(
                height_factor=0.02,
                width_factor=0.02,
                fill_mode="constant",
                fill_value=255.0,
                seed=RANDOM_SEED,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.04, 0.04),
                width_factor=(-0.04, 0.04),
                fill_mode="constant",
                fill_value=255.0,
                seed=RANDOM_SEED + 1,
            ),
            tf.keras.layers.RandomContrast(
                factor=0.08,
                seed=RANDOM_SEED + 2,
            ),
        ],
        name="webpage_augmentation",
    )

    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
        include_preprocessing=True,
    )

    backbone.trainable = False

    x = augmentation(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)
    x = tf.keras.layers.Dropout(
        0.30,
        name="dropout"
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="phishing_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="model2_effnetv2b0_stage1",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
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
                name="auc_roc",
                curve="ROC",
            ),
            tf.keras.metrics.AUC(
                name="auc_pr",
                curve="PR",
            ),
        ],
    )

    return model


def save_training_plot(
    history: tf.keras.callbacks.History,
) -> None:
    history_data = history.history

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    axes[0].plot(
        history_data.get("loss", []),
        label="Training",
    )
    axes[0].plot(
        history_data.get("val_loss", []),
        label="Validation",
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        history_data.get("accuracy", []),
        label="Training",
    )
    axes[1].plot(
        history_data.get("val_accuracy", []),
        label="Validation",
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        history_data.get("auc_pr", []),
        label="Training",
    )
    axes[2].plot(
        history_data.get("val_auc_pr", []),
        label="Validation",
    )
    axes[2].set_title("PR-AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(
        TRAINING_PLOT,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_augmentation_spotcheck(
    model: tf.keras.Model,
    sample_path: str,
) -> None:
    """Save one original and five on-the-fly training variations."""
    original_image, _ = load_and_prepare_image(
        tf.constant(sample_path),
        tf.constant(0),
    )

    augmentation_layer = model.get_layer(
        "webpage_augmentation"
    )

    images = [original_image]
    titles = ["Original training image"]

    for variation_number in range(1, 6):
        augmented_image = augmentation_layer(
            tf.expand_dims(original_image, axis=0),
            training=True,
        )[0]

        augmented_image = tf.clip_by_value(
            augmented_image,
            0.0,
            255.0,
        )

        images.append(augmented_image)
        titles.append(
            f"Augmented variation {variation_number}"
        )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18, 8),
    )

    for axis, image, title in zip(
        axes.flat,
        images,
        titles,
    ):
        axis.imshow(
            tf.cast(image, tf.uint8).numpy()
        )
        axis.set_title(title)
        axis.axis("off")

    figure.suptitle(
        "Model 2 Training Augmentation Spot-Check\n"
        "Small translation, zoom and contrast changes; "
        "no flipping or rotation",
        fontsize=15,
    )

    figure.tight_layout()
    figure.savefig(
        AUGMENTATION_SPOTCHECK,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("MODEL 2 - EFFICIENTNETV2B0 STAGE 1 TRAINING")
    print("=" * 80)
    print("Backbone: EfficientNetV2B0 with ImageNet weights")
    print("Backbone status: FROZEN")
    print(f"Input size: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Maximum epochs: {MAX_EPOCHS}")
    print("Fixed test set used during training: NO")
    print()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

    train_paths, train_labels, train_counts = load_manifest(
        manifest_path=TRAIN_MANIFEST,
        expected_total=EXPECTED_TRAIN_IMAGES,
        expected_per_class=EXPECTED_IMAGES_PER_TRAIN_CLASS,
        manifest_name="training",
    )

    validation_paths, validation_labels, validation_counts = load_manifest(
        manifest_path=VALIDATION_MANIFEST,
        expected_total=EXPECTED_VALIDATION_IMAGES,
        expected_per_class=EXPECTED_IMAGES_PER_VALIDATION_CLASS,
        manifest_name="validation",
    )

    normalized_train_paths = {
        os.path.normcase(os.path.normpath(path))
        for path in train_paths
    }

    normalized_validation_paths = {
        os.path.normcase(os.path.normpath(path))
        for path in validation_paths
    }

    path_overlap = (
        normalized_train_paths
        & normalized_validation_paths
    )

    if path_overlap:
        raise SystemExit(
            "\nERROR: Training/validation path leakage detected."
        )

    print("MANIFEST VALIDATION")
    print("-" * 80)
    print(
        f"Training: {len(train_paths):,} "
        f"({train_counts['LEGITIMATE']:,} legitimate, "
        f"{train_counts['PHISHING']:,} phishing)"
    )
    print(
        f"Validation: {len(validation_paths):,} "
        f"({validation_counts['LEGITIMATE']:,} legitimate, "
        f"{validation_counts['PHISHING']:,} phishing)"
    )
    print("Training/validation path overlap: 0")
    print()

    train_dataset = create_dataset(
        train_paths,
        train_labels,
        training=True,
    )

    validation_dataset = create_dataset(
        validation_paths,
        validation_labels,
        training=False,
    )

    model = build_model()

    save_augmentation_spotcheck(
        model=model,
        sample_path=train_paths[0],
    )

    with MODEL_SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        model.summary(
            print_fn=lambda line: file.write(line + "\n")
        )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_auc_pr",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc_pr",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc_pr",
            mode="max",
            factor=0.25,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(TRAINING_LOG),
            append=False,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    print("Starting Stage 1 training...")
    print()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(FINAL_MODEL_PATH)
    save_training_plot(history)

    best_model = tf.keras.models.load_model(
        BEST_MODEL_PATH
    )

    validation_results = best_model.evaluate(
        validation_dataset,
        return_dict=True,
        verbose=1,
    )

    completed_epochs = len(history.history.get("loss", []))

    results = {
        "stage": "stage1_frozen_backbone",
        "backbone": "EfficientNetV2B0",
        "input_width": IMAGE_WIDTH,
        "input_height": IMAGE_HEIGHT,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "completed_epochs": completed_epochs,
        "training_images": len(train_paths),
        "validation_images": len(validation_paths),
        "fixed_test_images_used": 0,
        "augmentation": {
            "application": "on_the_fly_training_only",
            "random_translation": "2_percent",
            "random_zoom": "plus_or_minus_4_percent",
            "random_contrast": "8_percent",
            "horizontal_flip": False,
            "rotation": False,
            "validation_augmented": False,
            "test_augmented": False,
            "spotcheck": str(AUGMENTATION_SPOTCHECK),
        },
        "best_model": str(BEST_MODEL_PATH),
        "validation_metrics": {
            metric_name: float(metric_value)
            for metric_name, metric_value
            in validation_results.items()
        },
    }

    RESULTS_FILE.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("STAGE 1 TRAINING COMPLETE")
    print("=" * 80)
    print(f"Completed epochs: {completed_epochs}")
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Final model: {FINAL_MODEL_PATH}")
    print(f"Training log: {TRAINING_LOG}")
    print(f"Training plot: {TRAINING_PLOT}")
    print(f"Augmentation spot-check: {AUGMENTATION_SPOTCHECK}")
    print(f"Validation results: {RESULTS_FILE}")
    print()
    print("BEST-MODEL VALIDATION METRICS")
    print("-" * 80)

    for metric_name, metric_value in validation_results.items():
        print(f"{metric_name:12s}: {metric_value:.6f}")

    print()
    print("The fixed test set was not used.")
    print("Do not run final test evaluation yet.")


if __name__ == "__main__":
    main()
