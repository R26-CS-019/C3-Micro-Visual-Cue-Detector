"""
09_audit_viewport_action_bias.py

Audits whether viewport_action is associated with class labels.

This script:
- Does not train the model.
- Does not change the threshold.
- Does not modify screenshots.
- Does not modify dataset manifests.
- Only creates audit reports.
"""

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(
    r"C:\Users\oshan\Desktop\model 1 login pp2"
)

SPLIT_REPORT_DIR = (
    PROJECT_DIR
    / "reports"
    / "05_dataset_split"
)

LOCKED_TEST_REPORT_DIR = (
    PROJECT_DIR
    / "reports"
    / "07_locked_test_evaluation"
)

MANIFEST_PATH = (
    SPLIT_REPORT_DIR
    / "complete_split_manifest.csv"
)

TEST_PREDICTIONS_PATH = (
    LOCKED_TEST_REPORT_DIR
    / "test_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "reports"
    / "09_viewport_action_bias_audit"
)

CLASS_NAMES = [
    "NOT_MALICIOUS",
    "MALICIOUS",
]

HIGH_CLASS_CONCENTRATION = 0.90


# =============================================================================
# GENERAL FUNCTIONS
# =============================================================================

def safe_divide(numerator, denominator):
    """Safely divide two values."""

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def calculate_sha256(file_path):
    """Calculate a file SHA256 without modifying it."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


def normalize_text(value):
    """Normalize category text."""

    return (
        str(value)
        .strip()
        .upper()
        .replace("-", "_")
        .replace(" ", "_")
    )


def normalize_split(value):
    """Normalize split names."""

    normalized = normalize_text(value)

    aliases = {
        "TRAINING": "TRAIN",
        "VAL": "VALIDATION",
        "VALID": "VALIDATION",
        "DEV": "VALIDATION",
        "TESTING": "TEST",
    }

    return aliases.get(
        normalized,
        normalized,
    )


def normalize_class(value):
    """Normalize class names."""

    normalized = normalize_text(value)

    legitimate_names = {
        "0",
        "NOT_MALICIOUS",
        "NOT_PHISHING",
        "NON_MALICIOUS",
        "LEGITIMATE",
        "BENIGN",
    }

    malicious_names = {
        "1",
        "MALICIOUS",
        "PHISHING",
        "PHISH",
    }

    if normalized in legitimate_names:
        return "NOT_MALICIOUS"

    if normalized in malicious_names:
        return "MALICIOUS"

    raise RuntimeError(
        f"Unsupported class value: {value!r}"
    )


def normalize_action(value):
    """Normalize viewport-action names."""

    normalized = normalize_text(value)

    if not normalized:
        return "UNKNOWN"

    return normalized


def write_csv(output_path, fieldnames, rows):
    """Write a UTF-8 CSV report."""

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
# MANIFEST READING
# =============================================================================

def read_complete_manifest():
    """Read and validate the complete split manifest."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Complete split manifest was not found:\n"
            f"{MANIFEST_PATH}"
        )

    with open(
        MANIFEST_PATH,
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
        "viewport_action",
    }

    missing_columns = required_columns.difference(
        fieldnames
    )

    if missing_columns:
        raise RuntimeError(
            "The complete manifest is missing columns:\n"
            f"{sorted(missing_columns)}"
        )

    records = []
    seen_paths = set()

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        split_name = normalize_split(
            row["split"]
        )

        class_from_text = normalize_class(
            row["class_label"]
        )

        class_from_number = normalize_class(
            row["numeric_label"]
        )

        if class_from_text != class_from_number:
            raise RuntimeError(
                f"Class mismatch at manifest row "
                f"{row_number}.\n"
                f"class_label: {row['class_label']}\n"
                f"numeric_label: {row['numeric_label']}"
            )

        relative_path = (
            row["prepared_relative_path"]
            .strip()
            .replace("\\", "/")
        )

        if not relative_path:
            raise RuntimeError(
                f"Empty prepared path at manifest "
                f"row {row_number}."
            )

        duplicate_key = (
            split_name,
            relative_path.lower(),
        )

        if duplicate_key in seen_paths:
            raise RuntimeError(
                "Duplicate prepared path found in "
                f"the same split:\n{relative_path}"
            )

        seen_paths.add(duplicate_key)

        records.append(
            {
                "split": split_name,
                "class_name": class_from_text,
                "viewport_action": normalize_action(
                    row["viewport_action"]
                ),
                "prepared_relative_path": relative_path,
            }
        )

    if not records:
        raise RuntimeError(
            "The complete split manifest is empty."
        )

    return records


# =============================================================================
# TEST PREDICTION READING
# =============================================================================

def read_test_predictions():
    """Read locked-test predictions."""

    if not TEST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Test predictions were not found:\n"
            f"{TEST_PREDICTIONS_PATH}"
        )

    with open(
        TEST_PREDICTIONS_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        fieldnames = reader.fieldnames or []
        rows = list(reader)

    required_columns = {
        "actual_class",
        "predicted_class",
        "viewport_action",
        "malicious_probability",
        "error_type",
    }

    missing_columns = required_columns.difference(
        fieldnames
    )

    if missing_columns:
        raise RuntimeError(
            "The test-predictions CSV is missing "
            f"columns:\n{sorted(missing_columns)}"
        )

    predictions = []

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        try:
            probability = float(
                row["malicious_probability"]
            )

        except ValueError as error:
            raise RuntimeError(
                f"Invalid probability at prediction "
                f"row {row_number}."
            ) from error

        predictions.append(
            {
                "actual_class": normalize_class(
                    row["actual_class"]
                ),
                "predicted_class": normalize_class(
                    row["predicted_class"]
                ),
                "viewport_action": normalize_action(
                    row["viewport_action"]
                ),
                "probability": probability,
                "error_type": normalize_text(
                    row["error_type"]
                ),
            }
        )

    if not predictions:
        raise RuntimeError(
            "The test-predictions CSV is empty."
        )

    return predictions


# =============================================================================
# VIEWPORT DISTRIBUTION
# =============================================================================

def build_scopes(records):
    """Create overall and split-specific record groups."""

    scopes = {
        "OVERALL": records,
    }

    split_names = sorted(
        {
            record["split"]
            for record in records
        }
    )

    for split_name in split_names:
        scopes[split_name] = [
            record
            for record in records
            if record["split"] == split_name
        ]

    return scopes


def create_action_count_rows(scopes):
    """Create detailed viewport-action count rows."""

    output_rows = []

    for scope_name, records in scopes.items():
        actions = sorted(
            {
                record["viewport_action"]
                for record in records
            }
        )

        class_totals = Counter(
            record["class_name"]
            for record in records
        )

        action_totals = Counter(
            record["viewport_action"]
            for record in records
        )

        pair_counts = Counter(
            (
                record["viewport_action"],
                record["class_name"],
            )
            for record in records
        )

        for action in actions:
            for class_name in CLASS_NAMES:
                count = pair_counts[
                    (
                        action,
                        class_name,
                    )
                ]

                within_class_percentage = (
                    safe_divide(
                        count,
                        class_totals[class_name],
                    )
                    * 100
                )

                within_action_percentage = (
                    safe_divide(
                        count,
                        action_totals[action],
                    )
                    * 100
                )

                output_rows.append(
                    {
                        "scope": scope_name,
                        "viewport_action": action,
                        "class_name": class_name,
                        "image_count": count,
                        "class_total": (
                            class_totals[class_name]
                        ),
                        "action_total": (
                            action_totals[action]
                        ),
                        "percentage_within_class": (
                            f"{within_class_percentage:.4f}"
                        ),
                        "percentage_within_action": (
                            f"{within_action_percentage:.4f}"
                        ),
                    }
                )

    return output_rows


def create_action_balance_rows(scopes):
    """Measure class concentration for each action."""

    output_rows = []

    for scope_name, records in scopes.items():
        actions = sorted(
            {
                record["viewport_action"]
                for record in records
            }
        )

        pair_counts = Counter(
            (
                record["viewport_action"],
                record["class_name"],
            )
            for record in records
        )

        for action in actions:
            legitimate_count = pair_counts[
                (
                    action,
                    "NOT_MALICIOUS",
                )
            ]

            malicious_count = pair_counts[
                (
                    action,
                    "MALICIOUS",
                )
            ]

            total = (
                legitimate_count
                + malicious_count
            )

            malicious_rate = safe_divide(
                malicious_count,
                total,
            )

            legitimate_rate = safe_divide(
                legitimate_count,
                total,
            )

            if malicious_count >= legitimate_count:
                majority_class = "MALICIOUS"
                majority_rate = malicious_rate
            else:
                majority_class = "NOT_MALICIOUS"
                majority_rate = legitimate_rate

            high_concentration = (
                total >= 10
                and majority_rate
                >= HIGH_CLASS_CONCENTRATION
            )

            output_rows.append(
                {
                    "scope": scope_name,
                    "viewport_action": action,
                    "not_malicious_count": (
                        legitimate_count
                    ),
                    "malicious_count": (
                        malicious_count
                    ),
                    "total_images": total,
                    "malicious_rate": (
                        f"{malicious_rate:.6f}"
                    ),
                    "majority_class": majority_class,
                    "majority_class_rate": (
                        f"{majority_rate:.6f}"
                    ),
                    "high_class_concentration": (
                        "YES"
                        if high_concentration
                        else "NO"
                    ),
                }
            )

    return output_rows


# =============================================================================
# ASSOCIATION MEASUREMENT
# =============================================================================

def calculate_cramers_v(records):
    """
    Calculate Cramer's V between viewport action and class.

    0 means little association.
    Values closer to 1 mean stronger association.
    """

    actions = sorted(
        {
            record["viewport_action"]
            for record in records
        }
    )

    if len(actions) < 2:
        return 0.0

    total = len(records)

    action_totals = Counter(
        record["viewport_action"]
        for record in records
    )

    class_totals = Counter(
        record["class_name"]
        for record in records
    )

    pair_counts = Counter(
        (
            record["viewport_action"],
            record["class_name"],
        )
        for record in records
    )

    chi_square = 0.0

    for action in actions:
        for class_name in CLASS_NAMES:
            observed = pair_counts[
                (
                    action,
                    class_name,
                )
            ]

            expected = (
                action_totals[action]
                * class_totals[class_name]
                / total
            )

            if expected > 0:
                chi_square += (
                    (observed - expected) ** 2
                    / expected
                )

    denominator_dimension = min(
        len(actions) - 1,
        len(CLASS_NAMES) - 1,
    )

    if denominator_dimension <= 0:
        return 0.0

    return math.sqrt(
        chi_square
        / (
            total
            * denominator_dimension
        )
    )


def association_strength(cramers_v):
    """Give a simple description of association strength."""

    if cramers_v >= 0.70:
        return "VERY_STRONG"

    if cramers_v >= 0.50:
        return "STRONG"

    if cramers_v >= 0.30:
        return "MODERATE"

    if cramers_v >= 0.10:
        return "WEAK"

    return "VERY_WEAK"


def create_association_rows(scopes):
    """Create association results for each split."""

    output_rows = []

    for scope_name, records in scopes.items():
        cramers_v = calculate_cramers_v(
            records
        )

        output_rows.append(
            {
                "scope": scope_name,
                "image_count": len(records),
                "cramers_v": (
                    f"{cramers_v:.6f}"
                ),
                "association_strength": (
                    association_strength(
                        cramers_v
                    )
                ),
            }
        )

    return output_rows


# =============================================================================
# TEST ERROR RATES
# =============================================================================

def calculate_prediction_metrics(predictions):
    """Calculate classification metrics for selected rows."""

    tn = sum(
        row["actual_class"] == "NOT_MALICIOUS"
        and row["predicted_class"] == "NOT_MALICIOUS"
        for row in predictions
    )

    fp = sum(
        row["actual_class"] == "NOT_MALICIOUS"
        and row["predicted_class"] == "MALICIOUS"
        for row in predictions
    )

    fn = sum(
        row["actual_class"] == "MALICIOUS"
        and row["predicted_class"] == "NOT_MALICIOUS"
        for row in predictions
    )

    tp = sum(
        row["actual_class"] == "MALICIOUS"
        and row["predicted_class"] == "MALICIOUS"
        for row in predictions
    )

    total = tn + fp + fn + tp

    return {
        "total_images": total,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "accuracy": safe_divide(
            tn + tp,
            total,
        ),
        "specificity": safe_divide(
            tn,
            tn + fp,
        ),
        "recall": safe_divide(
            tp,
            tp + fn,
        ),
        "false_positive_rate": safe_divide(
            fp,
            fp + tn,
        ),
        "false_negative_rate": safe_divide(
            fn,
            fn + tp,
        ),
    }


def create_test_action_metric_rows(predictions):
    """Calculate test errors for every viewport action."""

    output_rows = []

    action_names = sorted(
        {
            row["viewport_action"]
            for row in predictions
        }
    )

    action_names.insert(
        0,
        "OVERALL",
    )

    for action_name in action_names:
        if action_name == "OVERALL":
            selected_rows = predictions
        else:
            selected_rows = [
                row
                for row in predictions
                if row["viewport_action"]
                == action_name
            ]

        metrics = calculate_prediction_metrics(
            selected_rows
        )

        output_rows.append(
            {
                "viewport_action": action_name,
                "total_images": (
                    metrics["total_images"]
                ),
                "true_negatives": (
                    metrics["true_negatives"]
                ),
                "false_positives": (
                    metrics["false_positives"]
                ),
                "false_negatives": (
                    metrics["false_negatives"]
                ),
                "true_positives": (
                    metrics["true_positives"]
                ),
                "accuracy": (
                    f"{metrics['accuracy']:.6f}"
                ),
                "specificity": (
                    f"{metrics['specificity']:.6f}"
                ),
                "recall": (
                    f"{metrics['recall']:.6f}"
                ),
                "false_positive_rate": (
                    f"{metrics['false_positive_rate']:.6f}"
                ),
                "false_negative_rate": (
                    f"{metrics['false_negative_rate']:.6f}"
                ),
            }
        )

    return output_rows


# =============================================================================
# VALIDATION
# =============================================================================

def validate_test_consistency(
    manifest_records,
    predictions,
):
    """Check that manifest and prediction test counts match."""

    test_manifest_records = [
        record
        for record in manifest_records
        if record["split"] == "TEST"
    ]

    if len(test_manifest_records) != len(predictions):
        raise RuntimeError(
            "TEST count mismatch.\n"
            f"Manifest TEST rows: "
            f"{len(test_manifest_records)}\n"
            f"Prediction rows   : "
            f"{len(predictions)}"
        )

    manifest_counts = Counter(
        (
            row["viewport_action"],
            row["class_name"],
        )
        for row in test_manifest_records
    )

    prediction_counts = Counter(
        (
            row["viewport_action"],
            row["actual_class"],
        )
        for row in predictions
    )

    if manifest_counts != prediction_counts:
        raise RuntimeError(
            "Viewport/class counts do not match between "
            "the TEST manifest and test predictions."
        )


# =============================================================================
# SUMMARY
# =============================================================================

def create_summary(
    records,
    predictions,
    association_rows,
    balance_rows,
    test_metric_rows,
):
    """Create the final viewport-action audit summary."""

    split_counts = Counter(
        record["split"]
        for record in records
    )

    high_concentration_rows = [
        row
        for row in balance_rows
        if (
            row["high_class_concentration"] == "YES"
            and row["scope"] != "OVERALL"
        )
    ]

    strong_association_rows = [
        row
        for row in association_rows
        if row["association_strength"]
        in {
            "STRONG",
            "VERY_STRONG",
        }
    ]

    if (
        high_concentration_rows
        or strong_association_rows
    ):
        audit_result = (
            "POTENTIAL_VIEWPORT_ACTION_CLASS_"
            "ASSOCIATION_FOUND"
        )
    else:
        audit_result = (
            "NO_STRONG_VIEWPORT_ACTION_CLASS_"
            "ASSOCIATION_FOUND"
        )

    summary_lines = [
        "=" * 80,
        "MODEL 1 VIEWPORT-ACTION BIAS AUDIT COMPLETE",
        "=" * 80,
        "",
        f"Manifest             : {MANIFEST_PATH}",
        f"Manifest SHA256      : {calculate_sha256(MANIFEST_PATH)}",
        f"Test predictions     : {TEST_PREDICTIONS_PATH}",
        f"Predictions SHA256   : {calculate_sha256(TEST_PREDICTIONS_PATH)}",
        "",
        "DATASET COUNTS",
        "-" * 80,
        f"Total images         : {len(records)}",
    ]

    for split_name in sorted(split_counts):
        summary_lines.append(
            f"{split_name:<20} : "
            f"{split_counts[split_name]}"
        )

    summary_lines.extend(
        [
            "",
            "VIEWPORT/CLASS ASSOCIATION",
            "-" * 80,
        ]
    )

    for row in association_rows:
        summary_lines.append(
            f"{row['scope']:<20} : "
            f"Cramer's V = {row['cramers_v']} "
            f"({row['association_strength']})"
        )

    summary_lines.extend(
        [
            "",
            "LOCKED TEST ERROR DISTRIBUTION",
            "-" * 80,
        ]
    )

    for row in test_metric_rows:
        summary_lines.append(
            f"{row['viewport_action']:<20} : "
            f"Total={row['total_images']}, "
            f"TN={row['true_negatives']}, "
            f"FP={row['false_positives']}, "
            f"FN={row['false_negatives']}, "
            f"TP={row['true_positives']}"
        )

    summary_lines.extend(
        [
            "",
            "AUDIT RESULT",
            "-" * 80,
            audit_result,
            "",
            "INTERPRETATION",
            "-" * 80,
            (
                "A strong association does not prove that "
                "cropping directly caused the model errors."
            ),
            (
                "However, it can indicate that preprocessing "
                "or dataset source is connected to class."
            ),
            (
                "The model may learn visual source or image-"
                "composition shortcuts instead of phishing cues."
            ),
            "",
            "IMPORTANT",
            "-" * 80,
            "No model training was performed.",
            "The threshold was not changed.",
            "No screenshot or manifest was modified.",
            "=" * 80,
        ]
    )

    summary_text = "\n".join(
        summary_lines
    )

    with open(
        OUTPUT_DIR
        / "viewport_action_audit_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(summary_text)

    audit_json = {
        "audit_result": audit_result,
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": calculate_sha256(
            MANIFEST_PATH
        ),
        "test_predictions_path": str(
            TEST_PREDICTIONS_PATH
        ),
        "test_predictions_sha256": calculate_sha256(
            TEST_PREDICTIONS_PATH
        ),
        "total_manifest_images": len(records),
        "total_test_predictions": len(predictions),
        "split_counts": dict(split_counts),
        "association_results": association_rows,
        "high_concentration_actions": (
            high_concentration_rows
        ),
        "test_metrics_by_action": (
            test_metric_rows
        ),
    }

    with open(
        OUTPUT_DIR
        / "viewport_action_audit.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit_json,
            file,
            indent=4,
        )

    return summary_text


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("MODEL 1 VIEWPORT-ACTION BIAS AUDIT")
    print("=" * 80)

    print(
        f"Complete manifest   : "
        f"{MANIFEST_PATH}"
    )

    print(
        f"Test predictions    : "
        f"{TEST_PREDICTIONS_PATH}"
    )

    print(
        f"Output directory    : "
        f"{OUTPUT_DIR}"
    )

    print()
    print("Reading complete split manifest...")

    records = read_complete_manifest()

    print(
        f"Manifest images     : "
        f"{len(records)}"
    )

    print("Reading locked-test predictions...")

    predictions = read_test_predictions()

    print(
        f"Test predictions    : "
        f"{len(predictions)}"
    )

    validate_test_consistency(
        manifest_records=records,
        predictions=predictions,
    )

    print(
        "TEST manifest and prediction counts match."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scopes = build_scopes(
        records
    )

    action_count_rows = (
        create_action_count_rows(
            scopes
        )
    )

    action_balance_rows = (
        create_action_balance_rows(
            scopes
        )
    )

    association_rows = (
        create_association_rows(
            scopes
        )
    )

    test_metric_rows = (
        create_test_action_metric_rows(
            predictions
        )
    )

    write_csv(
        OUTPUT_DIR
        / "viewport_action_counts.csv",
        [
            "scope",
            "viewport_action",
            "class_name",
            "image_count",
            "class_total",
            "action_total",
            "percentage_within_class",
            "percentage_within_action",
        ],
        action_count_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "viewport_action_class_balance.csv",
        [
            "scope",
            "viewport_action",
            "not_malicious_count",
            "malicious_count",
            "total_images",
            "malicious_rate",
            "majority_class",
            "majority_class_rate",
            "high_class_concentration",
        ],
        action_balance_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "viewport_action_association.csv",
        [
            "scope",
            "image_count",
            "cramers_v",
            "association_strength",
        ],
        association_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "test_metrics_by_viewport_action.csv",
        [
            "viewport_action",
            "total_images",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
            "accuracy",
            "specificity",
            "recall",
            "false_positive_rate",
            "false_negative_rate",
        ],
        test_metric_rows,
    )

    summary_text = create_summary(
        records=records,
        predictions=predictions,
        association_rows=association_rows,
        balance_rows=action_balance_rows,
        test_metric_rows=test_metric_rows,
    )

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
        print("VIEWPORT-ACTION BIAS AUDIT FAILED")
        print("=" * 80)
        print(error)
        print()
        print(
            "No model, threshold, screenshot, "
            "or dataset manifest was modified."
        )

        sys.exit(1)