from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import train_model2_effnetv2b0_stage1 as stage1
import train_model2_effnetv2b0_stage2 as stage2


# =============================================================================
# PATHS
# =============================================================================

OUTPUT_FOLDER = (
    stage1.PROJECT_ROOT
    / "reports"
    / "model2_effnetv2b0"
    / "threshold_selection"
)

THRESHOLD_METRICS_CSV = (
    OUTPUT_FOLDER
    / "validation_threshold_metrics.csv"
)

VALIDATION_PREDICTIONS_CSV = (
    OUTPUT_FOLDER
    / "validation_predictions.csv"
)

SELECTED_THRESHOLD_JSON = (
    OUTPUT_FOLDER
    / "selected_threshold.json"
)

SUMMARY_FILE = (
    OUTPUT_FOLDER
    / "validation_threshold_summary.txt"
)

THRESHOLD_PLOT = (
    OUTPUT_FOLDER
    / "validation_threshold_curves.png"
)

CONFUSION_MATRIX_PLOT = (
    OUTPUT_FOLDER
    / "validation_confusion_matrix_selected_threshold.png"
)

DEFAULT_THRESHOLD = 0.50
MINIMUM_RECALL_TARGET = 0.90


def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def calculate_metrics(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    true_negative = int(
        np.sum((true_labels == 0) & (predictions == 0))
    )
    false_positive = int(
        np.sum((true_labels == 0) & (predictions == 1))
    )
    false_negative = int(
        np.sum((true_labels == 1) & (predictions == 0))
    )
    true_positive = int(
        np.sum((true_labels == 1) & (predictions == 1))
    )

    accuracy = safe_divide(
        true_positive + true_negative,
        len(true_labels),
    )

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    specificity = safe_divide(
        true_negative,
        true_negative + false_positive,
    )

    f1_score = safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2.0

    false_positive_rate = safe_divide(
        false_positive,
        false_positive + true_negative,
    )

    false_negative_rate = safe_divide(
        false_negative,
        false_negative + true_positive,
    )

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "balanced_accuracy": balanced_accuracy,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


def choose_best_f1(
    metric_rows: list[dict[str, float | int]],
) -> dict[str, float | int]:
    return max(
        metric_rows,
        key=lambda row: (
            float(row["f1_score"]),
            float(row["recall"]),
            -abs(float(row["threshold"]) - DEFAULT_THRESHOLD),
        ),
    )


def choose_best_balanced_accuracy(
    metric_rows: list[dict[str, float | int]],
) -> dict[str, float | int]:
    return max(
        metric_rows,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            float(row["f1_score"]),
            -abs(float(row["threshold"]) - DEFAULT_THRESHOLD),
        ),
    )


def choose_recall_target_candidate(
    metric_rows: list[dict[str, float | int]],
) -> dict[str, float | int] | None:
    candidates = [
        row
        for row in metric_rows
        if float(row["recall"]) >= MINIMUM_RECALL_TARGET
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda row: (
            float(row["precision"]),
            float(row["f1_score"]),
            float(row["threshold"]),
        ),
    )


def format_metric_block(
    title: str,
    row: dict[str, float | int],
) -> list[str]:
    return [
        title,
        "-" * 80,
        f"Threshold         : {float(row['threshold']):.3f}",
        f"Accuracy          : {float(row['accuracy']):.6f}",
        f"Precision         : {float(row['precision']):.6f}",
        f"Recall            : {float(row['recall']):.6f}",
        f"Specificity       : {float(row['specificity']):.6f}",
        f"F1-score          : {float(row['f1_score']):.6f}",
        (
            "Balanced accuracy  : "
            f"{float(row['balanced_accuracy']):.6f}"
        ),
        (
            "Confusion matrix   : "
            f"TN={int(row['true_negative'])}, "
            f"FP={int(row['false_positive'])}, "
            f"FN={int(row['false_negative'])}, "
            f"TP={int(row['true_positive'])}"
        ),
        "",
    ]


def save_threshold_plot(
    metric_rows: list[dict[str, float | int]],
    selected_threshold: float,
) -> None:
    thresholds = np.array(
        [float(row["threshold"]) for row in metric_rows]
    )

    precision = np.array(
        [float(row["precision"]) for row in metric_rows]
    )

    recall = np.array(
        [float(row["recall"]) for row in metric_rows]
    )

    f1_score = np.array(
        [float(row["f1_score"]) for row in metric_rows]
    )

    accuracy = np.array(
        [float(row["accuracy"]) for row in metric_rows]
    )

    balanced_accuracy = np.array(
        [
            float(row["balanced_accuracy"])
            for row in metric_rows
        ]
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 5),
    )

    axes[0].plot(thresholds, precision, label="Precision")
    axes[0].plot(thresholds, recall, label="Recall")
    axes[0].plot(thresholds, f1_score, label="F1-score")
    axes[0].axvline(
        selected_threshold,
        color="black",
        linestyle="--",
        label=f"Selected: {selected_threshold:.3f}",
    )
    axes[0].set_xlabel("Decision threshold")
    axes[0].set_ylabel("Metric")
    axes[0].set_title("Precision, Recall and F1-score")
    axes[0].set_xlim(0.05, 0.95)
    axes[0].set_ylim(0.50, 1.01)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(thresholds, accuracy, label="Accuracy")
    axes[1].plot(
        thresholds,
        balanced_accuracy,
        label="Balanced accuracy",
    )
    axes[1].axvline(
        selected_threshold,
        color="black",
        linestyle="--",
        label=f"Selected: {selected_threshold:.3f}",
    )
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("Accuracy by Threshold")
    axes[1].set_xlim(0.05, 0.95)
    axes[1].set_ylim(0.50, 1.01)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(
        THRESHOLD_PLOT,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_confusion_matrix_plot(
    selected_metrics: dict[str, float | int],
) -> None:
    matrix = np.array(
        [
            [
                int(selected_metrics["true_negative"]),
                int(selected_metrics["false_positive"]),
            ],
            [
                int(selected_metrics["false_negative"]),
                int(selected_metrics["true_positive"]),
            ],
        ]
    )

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(["Legitimate", "Phishing"])
    axis.set_yticklabels(["Legitimate", "Phishing"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(
        "Validation Confusion Matrix\n"
        f"Selected threshold = "
        f"{float(selected_metrics['threshold']):.3f}"
    )

    maximum_value = matrix.max()

    for row_index in range(2):
        for column_index in range(2):
            value = matrix[row_index, column_index]

            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > maximum_value / 2
                    else "black"
                ),
                fontsize=14,
            )

    figure.tight_layout()
    figure.savefig(
        CONFUSION_MATRIX_PLOT,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    print("=" * 80)
    print("MODEL 2 VALIDATION THRESHOLD OPTIMIZATION")
    print("=" * 80)
    print(f"Model: {stage2.BEST_MODEL_PATH}")
    print("Threshold selection data: VALIDATION ONLY")
    print("Fixed test set used: NO")
    print("Primary selection rule: maximum validation F1-score")
    print()

    if not stage2.BEST_MODEL_PATH.is_file():
        raise SystemExit(
            "\nERROR: The best Stage 2 model was not found:\n"
            f"{stage2.BEST_MODEL_PATH}"
        )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    validation_paths, validation_labels, class_counts = (
        stage1.load_manifest(
            manifest_path=stage1.VALIDATION_MANIFEST,
            expected_total=stage1.EXPECTED_VALIDATION_IMAGES,
            expected_per_class=(
                stage1.EXPECTED_IMAGES_PER_VALIDATION_CLASS
            ),
            manifest_name="validation",
        )
    )

    validation_dataset = stage1.create_dataset(
        validation_paths,
        validation_labels,
        training=False,
    )

    model = tf.keras.models.load_model(
        stage2.BEST_MODEL_PATH
    )

    print("Generating validation probabilities...")

    probabilities = model.predict(
        validation_dataset,
        verbose=1,
    ).reshape(-1)

    true_labels = np.asarray(
        validation_labels,
        dtype=np.int32,
    )

    if len(probabilities) != len(true_labels):
        raise SystemExit(
            "\nERROR: Prediction and label counts do not match."
        )

    threshold_values = np.round(
        np.arange(0.05, 0.951, 0.001),
        3,
    )

    metric_rows = [
        calculate_metrics(
            true_labels,
            probabilities,
            float(threshold),
        )
        for threshold in threshold_values
    ]

    default_metrics = calculate_metrics(
        true_labels,
        probabilities,
        DEFAULT_THRESHOLD,
    )

    best_f1_metrics = choose_best_f1(metric_rows)

    best_balanced_metrics = choose_best_balanced_accuracy(
        metric_rows
    )

    recall_target_metrics = choose_recall_target_candidate(
        metric_rows
    )

    # The threshold is locked using a predeclared validation-only rule.
    selected_metrics = best_f1_metrics
    selected_threshold = float(
        selected_metrics["threshold"]
    )

    metric_fieldnames = list(metric_rows[0].keys())

    with THRESHOLD_METRICS_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=metric_fieldnames,
        )

        writer.writeheader()
        writer.writerows(metric_rows)

    with stage1.VALIDATION_MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)
        validation_rows = list(reader)
        original_fieldnames = list(reader.fieldnames or [])

    if len(validation_rows) != len(probabilities):
        raise SystemExit(
            "\nERROR: Validation manifest order could not be "
            "matched to predictions."
        )

    prediction_fieldnames = original_fieldnames + [
        "true_numeric_label",
        "phishing_probability",
        "selected_threshold",
        "predicted_numeric_label",
        "predicted_label",
        "prediction_correct",
    ]

    prediction_rows: list[dict[str, str]] = []

    for row, true_label, probability in zip(
        validation_rows,
        true_labels,
        probabilities,
    ):
        predicted_label = int(
            probability >= selected_threshold
        )

        output_row = dict(row)
        output_row["true_numeric_label"] = str(int(true_label))
        output_row["phishing_probability"] = f"{float(probability):.8f}"
        output_row["selected_threshold"] = f"{selected_threshold:.3f}"
        output_row["predicted_numeric_label"] = str(predicted_label)
        output_row["predicted_label"] = (
            "PHISHING"
            if predicted_label == 1
            else "LEGITIMATE"
        )
        output_row["prediction_correct"] = (
            "TRUE"
            if predicted_label == int(true_label)
            else "FALSE"
        )

        prediction_rows.append(output_row)

    with VALIDATION_PREDICTIONS_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=prediction_fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(prediction_rows)

    selected_output = {
        "model": str(stage2.BEST_MODEL_PATH),
        "selection_dataset": "validation_only",
        "fixed_test_set_used": False,
        "selection_rule": "maximum_validation_f1_score",
        "selected_threshold": selected_threshold,
        "selected_validation_metrics": selected_metrics,
        "default_0_5_validation_metrics": default_metrics,
        "best_balanced_accuracy_candidate": best_balanced_metrics,
        "minimum_recall_target": MINIMUM_RECALL_TARGET,
        "recall_target_candidate": recall_target_metrics,
    }

    SELECTED_THRESHOLD_JSON.write_text(
        json.dumps(selected_output, indent=2),
        encoding="utf-8",
    )

    save_threshold_plot(
        metric_rows,
        selected_threshold,
    )

    save_confusion_matrix_plot(selected_metrics)

    summary_lines = [
        "=" * 80,
        "MODEL 2 VALIDATION THRESHOLD SELECTION SUMMARY",
        "=" * 80,
        "",
        f"Validation images: {len(true_labels):,}",
        (
            "Validation LEGITIMATE: "
            f"{class_counts['LEGITIMATE']:,}"
        ),
        (
            "Validation PHISHING: "
            f"{class_counts['PHISHING']:,}"
        ),
        "Fixed test images used: 0",
        "",
        *format_metric_block(
            "DEFAULT THRESHOLD RESULTS",
            default_metrics,
        ),
        *format_metric_block(
            "SELECTED MAXIMUM-F1 THRESHOLD",
            selected_metrics,
        ),
        *format_metric_block(
            "BEST BALANCED-ACCURACY CANDIDATE",
            best_balanced_metrics,
        ),
    ]

    if recall_target_metrics is not None:
        summary_lines.extend(
            format_metric_block(
                (
                    "BEST PRECISION WITH RECALL >= "
                    f"{MINIMUM_RECALL_TARGET:.0%}"
                ),
                recall_target_metrics,
            )
        )

    summary_lines.extend(
        [
            "LOCKED MODEL AND THRESHOLD",
            "-" * 80,
            f"Model: {stage2.BEST_MODEL_PATH}",
            f"Threshold: {selected_threshold:.3f}",
            "Selection rule: maximum validation F1-score",
            "",
            "REPORT FILES",
            "-" * 80,
            str(THRESHOLD_METRICS_CSV),
            str(VALIDATION_PREDICTIONS_CSV),
            str(SELECTED_THRESHOLD_JSON),
            str(THRESHOLD_PLOT),
            str(CONFUSION_MATRIX_PLOT),
            str(SUMMARY_FILE),
            "",
            "IMPORTANT",
            "-" * 80,
            "The fixed test set was not loaded or evaluated.",
            "Do not change the threshold after viewing test results.",
            "=" * 80,
        ]
    )

    summary_text = "\n".join(summary_lines)

    SUMMARY_FILE.write_text(
        summary_text,
        encoding="utf-8",
    )

    print()
    print(summary_text)


if __name__ == "__main__":
    main()
