from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, roc_curve

import optimize_model2_validation_threshold as threshold_selection
import train_model2_effnetv2b0_stage1 as stage1
import train_model2_effnetv2b0_stage2 as stage2


# =============================================================================
# FIXED TEST SETTINGS
# =============================================================================

TEST_MANIFEST = (
    stage1.SPLIT_FOLDER
    / "phish360_test_manifest.csv"
)

SELECTED_THRESHOLD_FILE = (
    threshold_selection.SELECTED_THRESHOLD_JSON
)

OUTPUT_FOLDER = (
    stage1.PROJECT_ROOT
    / "reports"
    / "model2_effnetv2b0"
    / "final_test"
)

TEST_PREDICTIONS_CSV = (
    OUTPUT_FOLDER
    / "final_test_predictions.csv"
)

FALSE_POSITIVES_CSV = (
    OUTPUT_FOLDER
    / "final_test_false_positives.csv"
)

FALSE_NEGATIVES_CSV = (
    OUTPUT_FOLDER
    / "final_test_false_negatives.csv"
)

RESULTS_JSON = (
    OUTPUT_FOLDER
    / "final_test_results.json"
)

SUMMARY_FILE = (
    OUTPUT_FOLDER
    / "final_test_summary.txt"
)

CONFUSION_MATRIX_PLOT = (
    OUTPUT_FOLDER
    / "final_test_confusion_matrix.png"
)

ROC_PR_PLOT = (
    OUTPUT_FOLDER
    / "final_test_roc_pr_curves.png"
)

EXPECTED_TEST_IMAGES = 2_765
EXPECTED_TEST_LEGITIMATE = 1_499
EXPECTED_TEST_PHISHING = 1_266


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_fixed_test_manifest():
    if not TEST_MANIFEST.is_file():
        raise SystemExit(
            "\nERROR: Fixed test manifest was not found:\n"
            f"{TEST_MANIFEST}"
        )

    with TEST_MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise SystemExit(
                "\nERROR: Fixed test manifest has no header."
            )

        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    path_column = stage1.find_column(
        fieldnames,
        [
            "screenshot_path",
            "image_path",
            "file_path",
            "path",
        ],
        "screenshot path",
    )

    label_column = stage1.find_column(
        fieldnames,
        [
            "label",
            "class",
            "class_label",
            "dataset_label",
        ],
        "class label",
    )

    if len(rows) != EXPECTED_TEST_IMAGES:
        raise SystemExit(
            "\nERROR: Fixed test image count changed.\n"
            f"Expected: {EXPECTED_TEST_IMAGES:,}\n"
            f"Found: {len(rows):,}"
        )

    paths: list[str] = []
    labels: list[int] = []
    class_counts = {
        "LEGITIMATE": 0,
        "PHISHING": 0,
    }

    seen_paths: set[str] = set()
    missing_files: list[str] = []

    for row_number, row in enumerate(rows, start=2):
        image_path = row[path_column].strip()
        label_name = row[label_column].strip().upper()

        if label_name not in stage1.LABEL_TO_NUMBER:
            raise SystemExit(
                f"\nERROR: Invalid test label on row "
                f"{row_number}: {label_name}"
            )

        normalized_path = os.path.normcase(
            os.path.normpath(image_path)
        )

        if normalized_path in seen_paths:
            raise SystemExit(
                "\nERROR: Duplicate path inside fixed test "
                f"manifest:\n{image_path}"
            )

        seen_paths.add(normalized_path)

        if not Path(image_path).is_file():
            missing_files.append(image_path)

            if len(missing_files) >= 20:
                break

        paths.append(image_path)
        labels.append(stage1.LABEL_TO_NUMBER[label_name])
        class_counts[label_name] += 1

    if missing_files:
        raise SystemExit(
            "\nERROR: Missing fixed-test screenshots "
            "(showing up to 20):\n"
            + "\n".join(missing_files)
        )

    if class_counts["LEGITIMATE"] != EXPECTED_TEST_LEGITIMATE:
        raise SystemExit(
            "\nERROR: Fixed-test LEGITIMATE count changed."
        )

    if class_counts["PHISHING"] != EXPECTED_TEST_PHISHING:
        raise SystemExit(
            "\nERROR: Fixed-test PHISHING count changed."
        )

    return (
        rows,
        fieldnames,
        path_column,
        label_column,
        paths,
        labels,
        class_counts,
    )


def load_locked_threshold() -> tuple[float, dict]:
    if not SELECTED_THRESHOLD_FILE.is_file():
        raise SystemExit(
            "\nERROR: Selected threshold file was not found:\n"
            f"{SELECTED_THRESHOLD_FILE}\n\n"
            "Run optimize_model2_validation_threshold.py first."
        )

    data = json.loads(
        SELECTED_THRESHOLD_FILE.read_text(
            encoding="utf-8"
        )
    )

    if data.get("fixed_test_set_used") is not False:
        raise SystemExit(
            "\nERROR: Threshold-selection record does not "
            "confirm validation-only selection."
        )

    if data.get("selection_rule") != "maximum_validation_f1_score":
        raise SystemExit(
            "\nERROR: Unexpected threshold selection rule."
        )

    threshold = float(data["selected_threshold"])

    if not 0.0 < threshold < 1.0:
        raise SystemExit(
            "\nERROR: Locked threshold is outside (0, 1)."
        )

    recorded_model = Path(str(data.get("model", "")))

    if os.path.normcase(os.path.normpath(str(recorded_model))) != (
        os.path.normcase(
            os.path.normpath(str(stage2.BEST_MODEL_PATH))
        )
    ):
        raise SystemExit(
            "\nERROR: Locked model does not match the Stage 2 "
            "best-model path."
        )

    return threshold, data


def calculate_probability_auc(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    roc_metric = tf.keras.metrics.AUC(
        curve="ROC"
    )
    pr_metric = tf.keras.metrics.AUC(
        curve="PR"
    )

    roc_metric.update_state(true_labels, probabilities)
    pr_metric.update_state(true_labels, probabilities)

    return (
        float(roc_metric.result().numpy()),
        float(pr_metric.result().numpy()),
    )


def save_confusion_matrix_plot(
    metrics: dict[str, float | int],
    threshold: float,
) -> None:
    matrix = np.array(
        [
            [
                int(metrics["true_negative"]),
                int(metrics["false_positive"]),
            ],
            [
                int(metrics["false_negative"]),
                int(metrics["true_positive"]),
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
        "Model 2 Fixed-Test Confusion Matrix\n"
        f"Locked threshold = {threshold:.3f}"
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
                fontsize=14,
                color=(
                    "white"
                    if value > maximum_value / 2
                    else "black"
                ),
            )

    figure.tight_layout()
    figure.savefig(
        CONFUSION_MATRIX_PLOT,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_roc_pr_plot(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    roc_auc: float,
    pr_auc: float,
) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(
        true_labels,
        probabilities,
    )

    precision, recall, _ = precision_recall_curve(
        true_labels,
        probabilities,
    )

    phishing_prevalence = float(np.mean(true_labels))

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    axes[0].plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {roc_auc:.4f}",
    )
    axes[0].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random baseline",
    )
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].set_title("Fixed-Test ROC Curve")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        recall,
        precision,
        label=f"PR-AUC = {pr_auc:.4f}",
    )
    axes[1].axhline(
        phishing_prevalence,
        linestyle="--",
        color="gray",
        label=(
            "Class baseline = "
            f"{phishing_prevalence:.4f}"
        ),
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Fixed-Test Precision–Recall Curve")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(
        ROC_PR_PLOT,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    print("=" * 80)
    print("MODEL 2 - ONE-TIME FIXED TEST EVALUATION")
    print("=" * 80)
    print("This script does not train or modify the model.")
    print("The locked validation-selected threshold will be used.")
    print()

    if not stage2.BEST_MODEL_PATH.is_file():
        raise SystemExit(
            "\nERROR: Best Stage 2 model was not found:\n"
            f"{stage2.BEST_MODEL_PATH}"
        )

    locked_threshold, threshold_record = (
        load_locked_threshold()
    )

    (
        test_rows,
        original_fieldnames,
        path_column,
        label_column,
        test_paths,
        test_labels,
        class_counts,
    ) = load_fixed_test_manifest()

    print("LOCKED EVALUATION CONFIGURATION")
    print("-" * 80)
    print(f"Model: {stage2.BEST_MODEL_PATH}")
    print(f"Threshold: {locked_threshold:.3f}")
    print("Selection rule: maximum validation F1-score")
    print(f"Fixed test images: {len(test_paths):,}")
    print(
        "Test LEGITIMATE: "
        f"{class_counts['LEGITIMATE']:,}"
    )
    print(
        "Test PHISHING: "
        f"{class_counts['PHISHING']:,}"
    )
    print()

    model_sha256 = sha256_file(stage2.BEST_MODEL_PATH)
    manifest_sha256 = sha256_file(TEST_MANIFEST)
    threshold_file_sha256 = sha256_file(
        SELECTED_THRESHOLD_FILE
    )

    test_dataset = stage1.create_dataset(
        test_paths,
        test_labels,
        training=False,
    )

    model = tf.keras.models.load_model(
        stage2.BEST_MODEL_PATH
    )

    print("Generating fixed-test predictions...")

    probabilities = model.predict(
        test_dataset,
        verbose=1,
    ).reshape(-1)

    true_labels = np.asarray(
        test_labels,
        dtype=np.int32,
    )

    if len(probabilities) != len(true_labels):
        raise SystemExit(
            "\nERROR: Test prediction and label counts differ."
        )

    locked_metrics = threshold_selection.calculate_metrics(
        true_labels,
        probabilities,
        locked_threshold,
    )

    default_metrics = threshold_selection.calculate_metrics(
        true_labels,
        probabilities,
        threshold_selection.DEFAULT_THRESHOLD,
    )

    roc_auc, pr_auc = calculate_probability_auc(
        true_labels,
        probabilities,
    )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    additional_fields = [
        "true_numeric_label",
        "phishing_probability",
        "locked_threshold",
        "predicted_numeric_label",
        "predicted_label",
        "prediction_correct",
        "error_type",
    ]

    output_fieldnames = list(original_fieldnames)

    for field in additional_fields:
        if field not in output_fieldnames:
            output_fieldnames.append(field)

    prediction_rows: list[dict[str, str]] = []
    false_positive_rows: list[dict[str, str]] = []
    false_negative_rows: list[dict[str, str]] = []

    for row, true_label, probability in zip(
        test_rows,
        true_labels,
        probabilities,
    ):
        predicted_label = int(
            probability >= locked_threshold
        )

        if true_label == 0 and predicted_label == 1:
            error_type = "FALSE_POSITIVE"
        elif true_label == 1 and predicted_label == 0:
            error_type = "FALSE_NEGATIVE"
        else:
            error_type = "CORRECT"

        output_row = dict(row)
        output_row["true_numeric_label"] = str(int(true_label))
        output_row["phishing_probability"] = f"{float(probability):.8f}"
        output_row["locked_threshold"] = f"{locked_threshold:.3f}"
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
        output_row["error_type"] = error_type

        prediction_rows.append(output_row)

        if error_type == "FALSE_POSITIVE":
            false_positive_rows.append(output_row)
        elif error_type == "FALSE_NEGATIVE":
            false_negative_rows.append(output_row)

    def write_output_csv(
        output_path: Path,
        rows: list[dict[str, str]],
    ) -> None:
        with output_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=output_fieldnames,
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(rows)

    write_output_csv(TEST_PREDICTIONS_CSV, prediction_rows)
    write_output_csv(FALSE_POSITIVES_CSV, false_positive_rows)
    write_output_csv(FALSE_NEGATIVES_CSV, false_negative_rows)

    save_confusion_matrix_plot(
        locked_metrics,
        locked_threshold,
    )

    save_roc_pr_plot(
        true_labels,
        probabilities,
        roc_auc,
        pr_auc,
    )

    validation_metrics = threshold_record[
        "selected_validation_metrics"
    ]

    comparison_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "balanced_accuracy",
    ]

    generalization_differences = {
        metric_name: (
            float(locked_metrics[metric_name])
            - float(validation_metrics[metric_name])
        )
        for metric_name in comparison_metrics
    }

    result_data = {
        "evaluation_type": "one_time_fixed_test",
        "model": str(stage2.BEST_MODEL_PATH),
        "model_sha256": model_sha256,
        "test_manifest": str(TEST_MANIFEST),
        "test_manifest_sha256": manifest_sha256,
        "selected_threshold_file": str(
            SELECTED_THRESHOLD_FILE
        ),
        "selected_threshold_file_sha256": (
            threshold_file_sha256
        ),
        "locked_threshold": locked_threshold,
        "threshold_selection_rule": (
            "maximum_validation_f1_score"
        ),
        "test_images": len(test_paths),
        "test_legitimate": class_counts["LEGITIMATE"],
        "test_phishing": class_counts["PHISHING"],
        "locked_threshold_metrics": locked_metrics,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "default_0_5_reference_metrics": default_metrics,
        "validation_selected_threshold_metrics": validation_metrics,
        "test_minus_validation": generalization_differences,
        "threshold_changed_after_test": False,
    }

    RESULTS_JSON.write_text(
        json.dumps(result_data, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        "=" * 80,
        "MODEL 2 FINAL FIXED-TEST RESULTS",
        "=" * 80,
        "",
        "LOCKED CONFIGURATION",
        "-" * 80,
        f"Model: {stage2.BEST_MODEL_PATH}",
        f"Threshold: {locked_threshold:.3f}",
        "Threshold selected using: validation only",
        "Threshold changed after test: NO",
        f"Test images: {len(test_paths):,}",
        (
            "Test LEGITIMATE: "
            f"{class_counts['LEGITIMATE']:,}"
        ),
        (
            "Test PHISHING: "
            f"{class_counts['PHISHING']:,}"
        ),
        "",
        "FINAL TEST METRICS",
        "-" * 80,
        f"Accuracy           : {float(locked_metrics['accuracy']):.6f}",
        f"Precision          : {float(locked_metrics['precision']):.6f}",
        f"Recall             : {float(locked_metrics['recall']):.6f}",
        f"Specificity        : {float(locked_metrics['specificity']):.6f}",
        f"F1-score           : {float(locked_metrics['f1_score']):.6f}",
        (
            "Balanced accuracy   : "
            f"{float(locked_metrics['balanced_accuracy']):.6f}"
        ),
        f"ROC-AUC            : {roc_auc:.6f}",
        f"PR-AUC             : {pr_auc:.6f}",
        f"False-positive rate: {float(locked_metrics['false_positive_rate']):.6f}",
        f"False-negative rate: {float(locked_metrics['false_negative_rate']):.6f}",
        (
            "Confusion matrix    : "
            f"TN={int(locked_metrics['true_negative'])}, "
            f"FP={int(locked_metrics['false_positive'])}, "
            f"FN={int(locked_metrics['false_negative'])}, "
            f"TP={int(locked_metrics['true_positive'])}"
        ),
        "",
        "TEST MINUS VALIDATION DIFFERENCE",
        "-" * 80,
    ]

    for metric_name in comparison_metrics:
        summary_lines.append(
            f"{metric_name:18s}: "
            f"{generalization_differences[metric_name] * 100:+.3f} "
            "percentage points"
        )

    summary_lines.extend(
        [
            "",
            "ERROR COUNTS",
            "-" * 80,
            f"False positives: {len(false_positive_rows):,}",
            f"False negatives: {len(false_negative_rows):,}",
            "",
            "REPORT FILES",
            "-" * 80,
            str(TEST_PREDICTIONS_CSV),
            str(FALSE_POSITIVES_CSV),
            str(FALSE_NEGATIVES_CSV),
            str(RESULTS_JSON),
            str(SUMMARY_FILE),
            str(CONFUSION_MATRIX_PLOT),
            str(ROC_PR_PLOT),
            "",
            "IMPORTANT",
            "-" * 80,
            "This was the one-time fixed-test evaluation.",
            "Do not tune the model or threshold using these test results.",
            "Use an independent external dataset for further development.",
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
