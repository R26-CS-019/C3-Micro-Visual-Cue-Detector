from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

import train_model2_effnetv2b0_stage1 as stage1


# =============================================================================
# STAGE 2 SETTINGS
# =============================================================================

PROJECT_ROOT = stage1.PROJECT_ROOT

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "models"
    / "model2_effnetv2b0"
    / "stage2"
)

REPORT_FOLDER = (
    PROJECT_ROOT
    / "reports"
    / "model2_effnetv2b0"
    / "stage2"
)

BEST_MODEL_PATH = (
    OUTPUT_FOLDER
    / "model2_effnetv2b0_stage2_best.keras"
)

FINAL_MODEL_PATH = (
    OUTPUT_FOLDER
    / "model2_effnetv2b0_stage2_final.keras"
)

TRAINING_LOG = (
    REPORT_FOLDER
    / "stage2_training_log.csv"
)

TRAINING_PLOT = (
    REPORT_FOLDER
    / "stage2_training_history.png"
)

RESULTS_FILE = (
    REPORT_FOLDER
    / "stage2_validation_results.json"
)

TRAINABLE_LAYERS_FILE = (
    REPORT_FOLDER
    / "stage2_trainable_layers.txt"
)

FINE_TUNE_LAST_LAYERS = 40
FINE_TUNE_LEARNING_RATE = 1e-5
MAX_FINE_TUNE_EPOCHS = 15


def find_efficientnet_backbone(
    model: tf.keras.Model,
) -> tf.keras.Model:
    """Find the nested EfficientNetV2 model safely."""
    for layer in model.layers:
        if (
            isinstance(layer, tf.keras.Model)
            and "efficientnet" in layer.name.lower()
        ):
            return layer

    raise SystemExit(
        "\nERROR: EfficientNetV2 backbone was not found "
        "inside the Stage 1 model."
    )


def configure_fine_tuning(
    backbone: tf.keras.Model,
) -> list[str]:
    """
    Unfreeze only the last backbone layers. Batch-normalization layers remain
    frozen because the dataset and batch size are relatively small.
    """
    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    candidate_layers = backbone.layers[
        -FINE_TUNE_LAST_LAYERS:
    ]

    for layer in candidate_layers:
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        ):
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_layer_names = [
        layer.name
        for layer in backbone.layers
        if layer.trainable and layer.trainable_weights
    ]

    if not trainable_layer_names:
        raise SystemExit(
            "\nERROR: No backbone layers became trainable."
        )

    return trainable_layer_names


def compile_for_fine_tuning(
    model: tf.keras.Model,
) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=FINE_TUNE_LEARNING_RATE
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


def save_training_plot(
    history: tf.keras.callbacks.History,
) -> None:
    values = history.history

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    axes[0].plot(
        values.get("loss", []),
        label="Training",
    )
    axes[0].plot(
        values.get("val_loss", []),
        label="Validation",
    )
    axes[0].set_title("Stage 2 Loss")
    axes[0].set_xlabel("Fine-tuning epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        values.get("accuracy", []),
        label="Training",
    )
    axes[1].plot(
        values.get("val_accuracy", []),
        label="Validation",
    )
    axes[1].set_title("Stage 2 Accuracy")
    axes[1].set_xlabel("Fine-tuning epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        values.get("auc_pr", []),
        label="Training",
    )
    axes[2].plot(
        values.get("val_auc_pr", []),
        label="Validation",
    )
    axes[2].set_title("Stage 2 PR-AUC")
    axes[2].set_xlabel("Fine-tuning epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(
        TRAINING_PLOT,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    print("=" * 80)
    print("MODEL 2 - EFFICIENTNETV2B0 STAGE 2 FINE-TUNING")
    print("=" * 80)
    print(f"Stage 1 model: {stage1.BEST_MODEL_PATH}")
    print(
        "Requested top backbone layers: "
        f"{FINE_TUNE_LAST_LAYERS}"
    )
    print(
        "Fine-tuning learning rate: "
        f"{FINE_TUNE_LEARNING_RATE}"
    )
    print(f"Maximum epochs: {MAX_FINE_TUNE_EPOCHS}")
    print("Training augmentation retained: YES")
    print("Validation augmentation: NO")
    print("Fixed test set used: NO")
    print()

    if not stage1.BEST_MODEL_PATH.is_file():
        raise SystemExit(
            "\nERROR: The best Stage 1 model was not found:\n"
            f"{stage1.BEST_MODEL_PATH}"
        )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

    train_paths, train_labels, train_counts = stage1.load_manifest(
        manifest_path=stage1.TRAIN_MANIFEST,
        expected_total=stage1.EXPECTED_TRAIN_IMAGES,
        expected_per_class=(
            stage1.EXPECTED_IMAGES_PER_TRAIN_CLASS
        ),
        manifest_name="training",
    )

    (
        validation_paths,
        validation_labels,
        validation_counts,
    ) = stage1.load_manifest(
        manifest_path=stage1.VALIDATION_MANIFEST,
        expected_total=stage1.EXPECTED_VALIDATION_IMAGES,
        expected_per_class=(
            stage1.EXPECTED_IMAGES_PER_VALIDATION_CLASS
        ),
        manifest_name="validation",
    )

    train_dataset = stage1.create_dataset(
        train_paths,
        train_labels,
        training=True,
    )

    validation_dataset = stage1.create_dataset(
        validation_paths,
        validation_labels,
        training=False,
    )

    model = tf.keras.models.load_model(
        stage1.BEST_MODEL_PATH
    )

    backbone = find_efficientnet_backbone(model)

    trainable_layer_names = configure_fine_tuning(
        backbone
    )

    compile_for_fine_tuning(model)

    trainable_parameters = sum(
        int(tf.keras.backend.count_params(weight))
        for weight in model.trainable_weights
    )

    total_parameters = model.count_params()

    TRAINABLE_LAYERS_FILE.write_text(
        "\n".join(
            [
                "MODEL 2 STAGE 2 TRAINABLE BACKBONE LAYERS",
                "=" * 70,
                f"Backbone: {backbone.name}",
                (
                    "Requested final backbone layers: "
                    f"{FINE_TUNE_LAST_LAYERS}"
                ),
                (
                    "Trainable weighted backbone layers: "
                    f"{len(trainable_layer_names)}"
                ),
                f"Total model parameters: {total_parameters:,}",
                f"Trainable parameters: {trainable_parameters:,}",
                "",
                *trainable_layer_names,
            ]
        ),
        encoding="utf-8",
    )

    print("MANIFEST AND MODEL VALIDATION")
    print("-" * 80)
    print(
        f"Training images: {len(train_paths):,} "
        f"({train_counts['LEGITIMATE']:,} legitimate, "
        f"{train_counts['PHISHING']:,} phishing)"
    )
    print(
        f"Validation images: {len(validation_paths):,} "
        f"({validation_counts['LEGITIMATE']:,} legitimate, "
        f"{validation_counts['PHISHING']:,} phishing)"
    )
    print(f"Backbone found: {backbone.name}")
    print(
        "Trainable weighted backbone layers: "
        f"{len(trainable_layer_names)}"
    )
    print(f"Trainable parameters: {trainable_parameters:,}")
    print()

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
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc_pr",
            mode="max",
            factor=0.25,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(TRAINING_LOG),
            append=False,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    print("Starting Stage 2 fine-tuning...")
    print()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=MAX_FINE_TUNE_EPOCHS,
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

    precision = float(
        validation_results.get("precision", 0.0)
    )
    recall = float(
        validation_results.get("recall", 0.0)
    )

    if precision + recall > 0:
        f1_score = (
            2.0 * precision * recall
            / (precision + recall)
        )
    else:
        f1_score = 0.0

    completed_epochs = len(
        history.history.get("loss", [])
    )

    results = {
        "stage": "stage2_partial_fine_tuning",
        "source_stage1_model": str(stage1.BEST_MODEL_PATH),
        "backbone": backbone.name,
        "requested_final_backbone_layers": FINE_TUNE_LAST_LAYERS,
        "trainable_weighted_backbone_layers": len(
            trainable_layer_names
        ),
        "learning_rate": FINE_TUNE_LEARNING_RATE,
        "maximum_epochs": MAX_FINE_TUNE_EPOCHS,
        "completed_epochs": completed_epochs,
        "training_images": len(train_paths),
        "validation_images": len(validation_paths),
        "fixed_test_images_used": 0,
        "training_augmentation": True,
        "validation_augmentation": False,
        "best_model": str(BEST_MODEL_PATH),
        "validation_metrics": {
            **{
                metric_name: float(metric_value)
                for metric_name, metric_value
                in validation_results.items()
            },
            "f1_score": f1_score,
        },
    }

    RESULTS_FILE.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("STAGE 2 FINE-TUNING COMPLETE")
    print("=" * 80)
    print(f"Completed epochs: {completed_epochs}")
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Final model: {FINAL_MODEL_PATH}")
    print(f"Training log: {TRAINING_LOG}")
    print(f"Training plot: {TRAINING_PLOT}")
    print(f"Validation results: {RESULTS_FILE}")
    print(f"Trainable-layer report: {TRAINABLE_LAYERS_FILE}")
    print()
    print("BEST-MODEL VALIDATION METRICS")
    print("-" * 80)

    for metric_name, metric_value in validation_results.items():
        print(f"{metric_name:12s}: {metric_value:.6f}")

    print(f"{'f1_score':12s}: {f1_score:.6f}")
    print()
    print("The fixed test set was not used.")
    print("Do not run final test evaluation yet.")


if __name__ == "__main__":
    main()
