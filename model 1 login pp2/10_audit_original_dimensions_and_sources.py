"""
10_audit_original_dimensions_and_sources.py

Audits the original screenshot dimensions, file formats,
source folders and filename patterns for Model 1.

Original dataset:
C:\\Users\\oshan\\Desktop\\new model 1 pp2\\dataset

Important:
- This script does not train the model.
- It does not change the threshold.
- It does not modify screenshots.
- It does not modify manifests or dataset splits.
"""

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(
    r"C:\Users\oshan\Desktop\model 1 login pp2"
)

ORIGINAL_DATASET_DIR = Path(
    r"C:\Users\oshan\Desktop\new model 1 pp2\dataset"
)

MANIFEST_PATH = (
    PROJECT_DIR
    / "reports"
    / "05_dataset_split"
    / "complete_split_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "reports"
    / "10_original_dimensions_sources_audit"
)

TARGET_WIDTH = 384
TARGET_HEIGHT = 160

CLASS_NAMES = [
    "NOT_MALICIOUS",
    "MALICIOUS",
]

FEATURE_NAMES = [
    "original_dimensions",
    "aspect_ratio_bucket",
    "orientation",
    "image_size_bucket",
    "target_dimension_match",
    "file_extension",
    "detected_image_format",
    "source_folder",
    "filename_pattern",
    "viewport_action",
]


# =============================================================================
# GENERAL FUNCTIONS
# =============================================================================

def safe_divide(numerator, denominator):
    """Safely divide two numbers."""

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def calculate_sha256(file_path):
    """Calculate SHA256 without modifying a file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            block = file.read(
                1024 * 1024
            )

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
    """Normalize dataset split names."""

    normalized = normalize_text(
        value
    )

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

    normalized = normalize_text(
        value
    )

    legitimate_values = {
        "0",
        "NOT_MALICIOUS",
        "NON_MALICIOUS",
        "NOT_PHISHING",
        "LEGITIMATE",
        "BENIGN",
    }

    malicious_values = {
        "1",
        "MALICIOUS",
        "PHISHING",
        "PHISH",
    }

    if normalized in legitimate_values:
        return "NOT_MALICIOUS"

    if normalized in malicious_values:
        return "MALICIOUS"

    raise RuntimeError(
        f"Unsupported class value: "
        f"{value!r}"
    )


def normalize_action(value):
    """Normalize viewport-action names."""

    normalized = normalize_text(
        value
    )

    if not normalized:
        return "UNKNOWN"

    return normalized


def write_csv(
    output_path,
    fieldnames,
    rows,
):
    """Write a CSV report."""

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# IMAGE CATEGORY FUNCTIONS
# =============================================================================

def get_orientation(width, height):
    """Return the image orientation."""

    if width > height:
        return "LANDSCAPE"

    if height > width:
        return "PORTRAIT"

    return "SQUARE"


def get_aspect_ratio_bucket(
    width,
    height,
):
    """Create an aspect-ratio category."""

    ratio = safe_divide(
        width,
        height,
    )

    if ratio < 0.75:
        return "PORTRAIT_NARROW"

    if ratio < 1.25:
        return "NEAR_SQUARE"

    if ratio < 1.75:
        return "LANDSCAPE_STANDARD"

    if ratio < 2.25:
        return "LANDSCAPE_WIDE"

    if ratio < 2.75:
        return "LANDSCAPE_TARGET_LIKE"

    return "LANDSCAPE_ULTRAWIDE"


def get_image_size_bucket(
    width,
    height,
):
    """Create a pixel-size category."""

    megapixels = (
        width
        * height
        / 1_000_000
    )

    if megapixels < 0.10:
        return "VERY_SMALL_UNDER_0_1_MP"

    if megapixels < 0.50:
        return "SMALL_0_1_TO_0_5_MP"

    if megapixels < 2.00:
        return "MEDIUM_0_5_TO_2_MP"

    if megapixels < 5.00:
        return "LARGE_2_TO_5_MP"

    return "VERY_LARGE_OVER_5_MP"


def get_source_folder(
    original_relative_path,
):
    """Extract the first source folder."""

    normalized_path = (
        str(original_relative_path)
        .strip()
        .replace("\\", "/")
    )

    path_parts = [
        part
        for part in normalized_path.split("/")
        if part
    ]

    if len(path_parts) > 1:
        return normalize_text(
            path_parts[0]
        )

    return "NO_SOURCE_FOLDER"


def get_filename_pattern(filename):
    """Create a broad filename pattern."""

    stem = (
        Path(filename)
        .stem
        .lower()
    )

    if "non_malicious" in stem:
        return "CONTAINS_NON_MALICIOUS"

    if "not_malicious" in stem:
        return "CONTAINS_NOT_MALICIOUS"

    if stem.startswith("screenshot"):
        return "SCREENSHOT_NUMBERED"

    if "login" in stem:
        return "CONTAINS_LOGIN"

    if re.fullmatch(
        r"[a-f0-9]{32,64}",
        stem,
    ):
        return "HASH_STYLE_FILENAME"

    if re.fullmatch(
        r"\d+",
        stem,
    ):
        return "NUMERIC_FILENAME"

    return "OTHER_FILENAME_PATTERN"


# =============================================================================
# MANIFEST READING
# =============================================================================

def read_manifest():
    """Read and validate the complete manifest."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Complete manifest was not found:\n"
            f"{MANIFEST_PATH}"
        )

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        fieldnames = (
            reader.fieldnames
            or []
        )

        rows = list(
            reader
        )

    required_columns = {
        "split",
        "class_label",
        "numeric_label",
        "prepared_relative_path",
        "original_relative_path",
        "viewport_action",
    }

    missing_columns = (
        required_columns.difference(
            fieldnames
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Manifest is missing columns:\n"
            f"{sorted(missing_columns)}"
        )

    records = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        class_from_text = (
            normalize_class(
                row["class_label"]
            )
        )

        class_from_number = (
            normalize_class(
                row["numeric_label"]
            )
        )

        if (
            class_from_text
            != class_from_number
        ):
            raise RuntimeError(
                f"Class-label mismatch at "
                f"manifest row {index}."
            )

        original_relative_path = (
            row["original_relative_path"]
            .strip()
            .replace("\\", "/")
        )

        if not original_relative_path:
            raise RuntimeError(
                f"Empty original path at "
                f"manifest row {index}."
            )

        records.append(
            {
                "manifest_index": index,
                "split": normalize_split(
                    row["split"]
                ),
                "class_name": (
                    class_from_text
                ),
                "numeric_label": (
                    1
                    if class_from_text
                    == "MALICIOUS"
                    else 0
                ),
                "viewport_action": (
                    normalize_action(
                        row[
                            "viewport_action"
                        ]
                    )
                ),
                "prepared_relative_path": (
                    row[
                        "prepared_relative_path"
                    ]
                    .strip()
                    .replace("\\", "/")
                ),
                "original_relative_path": (
                    original_relative_path
                ),
                "source_folder": (
                    get_source_folder(
                        original_relative_path
                    )
                ),
                "filename_pattern": (
                    get_filename_pattern(
                        Path(
                            original_relative_path
                        ).name
                    )
                ),
            }
        )

    if not records:
        raise RuntimeError(
            "The complete manifest is empty."
        )

    return records


# =============================================================================
# ORIGINAL IMAGE INSPECTION
# =============================================================================

def resolve_original_path(
    original_relative_path,
):
    """Resolve an original screenshot path."""

    normalized_relative_path = (
        str(original_relative_path)
        .strip()
        .replace("\\", "/")
    )

    return (
        ORIGINAL_DATASET_DIR
        / Path(
            normalized_relative_path
        )
    )


def inspect_original_images(
    manifest_records,
):
    """Read original dimensions and formats."""

    inventory_rows = []

    print(
        "Inspecting original screenshots..."
    )

    for position, record in enumerate(
        manifest_records,
        start=1,
    ):
        original_path = (
            resolve_original_path(
                record[
                    "original_relative_path"
                ]
            )
        )

        inventory_row = dict(
            record
        )

        inventory_row.update(
            {
                "resolved_original_path": (
                    str(original_path)
                ),
                "file_status": "",
                "read_status": "",
                "original_width": "",
                "original_height": "",
                "original_dimensions": "",
                "original_aspect_ratio": "",
                "aspect_ratio_bucket": "",
                "orientation": "",
                "megapixels": "",
                "image_size_bucket": "",
                "target_dimension_match": "",
                "geometric_change_required": "",
                "file_extension": "",
                "detected_image_format": "",
                "frame_count": "",
                "error_message": "",
            }
        )

        if not original_path.exists():
            inventory_row[
                "file_status"
            ] = "NOT_FOUND"

            inventory_row[
                "read_status"
            ] = "NOT_INSPECTED"

            inventory_rows.append(
                inventory_row
            )

        else:
            inventory_row[
                "file_status"
            ] = "FOUND"

            try:
                with Image.open(
                    original_path
                ) as image:
                    width, height = (
                        image.size
                    )

                    image_format = (
                        image.format
                        if image.format
                        else "UNKNOWN"
                    )

                    frame_count = int(
                        getattr(
                            image,
                            "n_frames",
                            1,
                        )
                    )

                if (
                    width <= 0
                    or height <= 0
                ):
                    raise RuntimeError(
                        "Invalid image dimensions."
                    )

                aspect_ratio = safe_divide(
                    width,
                    height,
                )

                megapixels = (
                    width
                    * height
                    / 1_000_000
                )

                target_match = (
                    width == TARGET_WIDTH
                    and height
                    == TARGET_HEIGHT
                )

                inventory_row.update(
                    {
                        "read_status": "OK",
                        "original_width": (
                            width
                        ),
                        "original_height": (
                            height
                        ),
                        "original_dimensions": (
                            f"{width}x{height}"
                        ),
                        "original_aspect_ratio": (
                            f"{aspect_ratio:.6f}"
                        ),
                        "aspect_ratio_bucket": (
                            get_aspect_ratio_bucket(
                                width,
                                height,
                            )
                        ),
                        "orientation": (
                            get_orientation(
                                width,
                                height,
                            )
                        ),
                        "megapixels": (
                            f"{megapixels:.6f}"
                        ),
                        "image_size_bucket": (
                            get_image_size_bucket(
                                width,
                                height,
                            )
                        ),
                        "target_dimension_match": (
                            "YES"
                            if target_match
                            else "NO"
                        ),
                        "geometric_change_required": (
                            "NO"
                            if target_match
                            else "YES"
                        ),
                        "file_extension": (
                            original_path
                            .suffix
                            .lower()
                        ),
                        "detected_image_format": (
                            str(
                                image_format
                            ).upper()
                        ),
                        "frame_count": (
                            frame_count
                        ),
                    }
                )

            except Exception as error:
                inventory_row[
                    "read_status"
                ] = "READ_ERROR"

                inventory_row[
                    "error_message"
                ] = str(error)

            inventory_rows.append(
                inventory_row
            )

        if (
            position % 250 == 0
            or position
            == len(manifest_records)
        ):
            print(
                f"Inspected {position}/"
                f"{len(manifest_records)}"
            )

    return inventory_rows


# =============================================================================
# CRAMER'S V
# =============================================================================

def calculate_cramers_v(
    rows,
    feature_name,
):
    """Calculate association between a feature and class."""

    usable_rows = [
        row
        for row in rows
        if (
            row["read_status"] == "OK"
            and str(
                row.get(
                    feature_name,
                    "",
                )
            ).strip()
        )
    ]

    if not usable_rows:
        return 0.0, 0, 0

    categories = sorted(
        {
            str(
                row[feature_name]
            )
            for row in usable_rows
        }
    )

    present_classes = sorted(
        {
            row["class_name"]
            for row in usable_rows
        }
    )

    if (
        len(categories) < 2
        or len(present_classes) < 2
    ):
        return (
            0.0,
            len(usable_rows),
            len(categories),
        )

    total = len(
        usable_rows
    )

    category_totals = Counter(
        str(row[feature_name])
        for row in usable_rows
    )

    class_totals = Counter(
        row["class_name"]
        for row in usable_rows
    )

    pair_counts = Counter(
        (
            str(row[feature_name]),
            row["class_name"],
        )
        for row in usable_rows
    )

    chi_square = 0.0

    for category in categories:
        for class_name in present_classes:
            observed = pair_counts[
                (
                    category,
                    class_name,
                )
            ]

            expected = (
                category_totals[category]
                * class_totals[class_name]
                / total
            )

            if expected > 0:
                chi_square += (
                    (observed - expected) ** 2
                    / expected
                )

    denominator_dimension = min(
        len(categories) - 1,
        len(present_classes) - 1,
    )

    if denominator_dimension <= 0:
        return (
            0.0,
            total,
            len(categories),
        )

    cramers_v = math.sqrt(
        chi_square
        / (
            total
            * denominator_dimension
        )
    )

    return (
        cramers_v,
        total,
        len(categories),
    )


def association_strength(value):
    """Describe Cramer's V strength."""

    if value >= 0.70:
        return "VERY_STRONG"

    if value >= 0.50:
        return "STRONG"

    if value >= 0.30:
        return "MODERATE"

    if value >= 0.10:
        return "WEAK"

    return "VERY_WEAK"


def feature_visibility_note(
    feature_name,
):
    """Explain whether the model could see the feature."""

    if feature_name in {
        "source_folder",
        "filename_pattern",
    }:
        return (
            "METADATA_ONLY_NOT_DIRECTLY_VISIBLE"
        )

    if feature_name in {
        "file_extension",
        "detected_image_format",
    }:
        return (
            "POSSIBLY_INDIRECT_THROUGH_ENCODING"
        )

    return (
        "POTENTIALLY_VISIBLE_THROUGH_"
        "GEOMETRY_OR_RESAMPLING"
    )


def build_association_rows(
    inventory_rows,
):
    """Create association reports overall and by split."""

    scopes = {
        "OVERALL": inventory_rows,
    }

    split_names = sorted(
        {
            row["split"]
            for row in inventory_rows
        }
    )

    for split_name in split_names:
        scopes[split_name] = [
            row
            for row in inventory_rows
            if row["split"]
            == split_name
        ]

    output_rows = []

    for scope_name, rows in scopes.items():
        for feature_name in FEATURE_NAMES:
            (
                cramers_v,
                usable_count,
                category_count,
            ) = calculate_cramers_v(
                rows,
                feature_name,
            )

            output_rows.append(
                {
                    "scope": scope_name,
                    "feature": (
                        feature_name
                    ),
                    "usable_images": (
                        usable_count
                    ),
                    "category_count": (
                        category_count
                    ),
                    "cramers_v": (
                        cramers_v
                    ),
                    "association_strength": (
                        association_strength(
                            cramers_v
                        )
                    ),
                    "model_visibility_note": (
                        feature_visibility_note(
                            feature_name
                        )
                    ),
                }
            )

    return output_rows


# =============================================================================
# CATEGORY COUNTS
# =============================================================================

def build_category_count_rows(
    inventory_rows,
):
    """Count each feature category by class."""

    scopes = {
        "OVERALL": inventory_rows,
    }

    for split_name in sorted(
        {
            row["split"]
            for row in inventory_rows
        }
    ):
        scopes[split_name] = [
            row
            for row in inventory_rows
            if row["split"]
            == split_name
        ]

    output_rows = []

    for scope_name, rows in scopes.items():
        for feature_name in FEATURE_NAMES:
            usable_rows = [
                row
                for row in rows
                if (
                    row["read_status"]
                    == "OK"
                    and str(
                        row.get(
                            feature_name,
                            "",
                        )
                    ).strip()
                )
            ]

            class_totals = Counter(
                row["class_name"]
                for row in usable_rows
            )

            pair_counts = Counter(
                (
                    str(
                        row[feature_name]
                    ),
                    row["class_name"],
                )
                for row in usable_rows
            )

            categories = sorted(
                {
                    str(
                        row[feature_name]
                    )
                    for row in usable_rows
                }
            )

            for category in categories:
                for class_name in CLASS_NAMES:
                    count = pair_counts[
                        (
                            category,
                            class_name,
                        )
                    ]

                    percentage = (
                        safe_divide(
                            count,
                            class_totals[
                                class_name
                            ],
                        )
                        * 100
                    )

                    output_rows.append(
                        {
                            "scope": (
                                scope_name
                            ),
                            "feature": (
                                feature_name
                            ),
                            "category": (
                                category
                            ),
                            "class_name": (
                                class_name
                            ),
                            "image_count": (
                                count
                            ),
                            "class_total": (
                                class_totals[
                                    class_name
                                ]
                            ),
                            "percentage_within_class": (
                                f"{percentage:.4f}"
                            ),
                        }
                    )

    return output_rows


def build_dimension_action_rows(
    inventory_rows,
):
    """Cross-tab dimensions and viewport action."""

    usable_rows = [
        row
        for row in inventory_rows
        if row["read_status"] == "OK"
    ]

    counts = Counter(
        (
            row["split"],
            row["class_name"],
            row["viewport_action"],
            row["original_dimensions"],
        )
        for row in usable_rows
    )

    output_rows = []

    for key in sorted(
        counts.keys()
    ):
        (
            split_name,
            class_name,
            viewport_action,
            original_dimensions,
        ) = key

        output_rows.append(
            {
                "split": split_name,
                "class_name": class_name,
                "viewport_action": (
                    viewport_action
                ),
                "original_dimensions": (
                    original_dimensions
                ),
                "image_count": counts[key],
            }
        )

    return output_rows


# =============================================================================
# SUMMARY
# =============================================================================

def create_summary(
    inventory_rows,
    association_rows,
):
    """Create final audit summary."""

    file_status_counts = Counter(
        row["file_status"]
        for row in inventory_rows
    )

    read_status_counts = Counter(
        row["read_status"]
        for row in inventory_rows
    )

    usable_rows = [
        row
        for row in inventory_rows
        if row["read_status"] == "OK"
    ]

    resolution_rate = safe_divide(
        len(usable_rows),
        len(inventory_rows),
    )

    class_counts = Counter(
        row["class_name"]
        for row in usable_rows
    )

    target_match_counts = Counter(
        (
            row["class_name"],
            row["target_dimension_match"],
        )
        for row in usable_rows
    )

    top_dimensions = {}

    for class_name in CLASS_NAMES:
        dimension_counts = Counter(
            row["original_dimensions"]
            for row in usable_rows
            if row["class_name"]
            == class_name
        )

        top_dimensions[class_name] = (
            dimension_counts.most_common(
                10
            )
        )

    overall_associations = {
        row["feature"]: row
        for row in association_rows
        if row["scope"] == "OVERALL"
    }

    geometry_features = [
        "original_dimensions",
        "aspect_ratio_bucket",
        "orientation",
        "image_size_bucket",
        "target_dimension_match",
    ]

    strongest_geometry_value = max(
        (
            overall_associations[
                feature_name
            ]["cramers_v"]
            for feature_name
            in geometry_features
            if feature_name
            in overall_associations
        ),
        default=0.0,
    )

    if resolution_rate < 0.90:
        audit_result = (
            "AUDIT_INCOMPLETE_TOO_MANY_"
            "ORIGINAL_FILES_UNRESOLVED"
        )

    elif strongest_geometry_value >= 0.70:
        audit_result = (
            "VERY_STRONG_ORIGINAL_GEOMETRY_"
            "CLASS_ASSOCIATION_FOUND"
        )

    elif strongest_geometry_value >= 0.50:
        audit_result = (
            "STRONG_ORIGINAL_GEOMETRY_"
            "CLASS_ASSOCIATION_FOUND"
        )

    else:
        audit_result = (
            "NO_STRONG_ORIGINAL_GEOMETRY_"
            "CLASS_ASSOCIATION_FOUND"
        )

    summary_lines = [
        "=" * 80,
        (
            "MODEL 1 ORIGINAL DIMENSIONS "
            "AND SOURCES AUDIT COMPLETE"
        ),
        "=" * 80,
        "",
        f"Manifest             : {MANIFEST_PATH}",
        f"Manifest SHA256      : {calculate_sha256(MANIFEST_PATH)}",
        f"Original dataset     : {ORIGINAL_DATASET_DIR}",
        "",
        "FILE RESOLUTION",
        "-" * 80,
        (
            f"Manifest images      : "
            f"{len(inventory_rows)}"
        ),
        (
            f"Successfully inspected: "
            f"{len(usable_rows)}"
        ),
        (
            f"Resolution rate      : "
            f"{resolution_rate:.6f}"
        ),
    ]

    for status_name in sorted(
        file_status_counts
    ):
        summary_lines.append(
            f"File {status_name:<17}: "
            f"{file_status_counts[status_name]}"
        )

    for status_name in sorted(
        read_status_counts
    ):
        summary_lines.append(
            f"Read {status_name:<17}: "
            f"{read_status_counts[status_name]}"
        )

    summary_lines.extend(
        [
            "",
            "USABLE ORIGINAL IMAGES BY CLASS",
            "-" * 80,
        ]
    )

    for class_name in CLASS_NAMES:
        summary_lines.append(
            f"{class_name:<22}: "
            f"{class_counts[class_name]}"
        )

    summary_lines.extend(
        [
            "",
            (
                f"TARGET DIMENSION MATCH "
                f"({TARGET_WIDTH}x{TARGET_HEIGHT})"
            ),
            "-" * 80,
        ]
    )

    for class_name in CLASS_NAMES:
        summary_lines.append(
            f"{class_name} - YES: "
            f"{target_match_counts[(class_name, 'YES')]}"
        )

        summary_lines.append(
            f"{class_name} - NO : "
            f"{target_match_counts[(class_name, 'NO')]}"
        )

    summary_lines.extend(
        [
            "",
            "TOP ORIGINAL DIMENSIONS",
            "-" * 80,
        ]
    )

    for class_name in CLASS_NAMES:
        summary_lines.append(
            f"{class_name}:"
        )

        for dimensions, count in (
            top_dimensions[class_name]
        ):
            summary_lines.append(
                f"  {dimensions:<15}: "
                f"{count}"
            )

    summary_lines.extend(
        [
            "",
            "OVERALL FEATURE/CLASS ASSOCIATION",
            "-" * 80,
        ]
    )

    for feature_name in sorted(
        overall_associations
    ):
        association = (
            overall_associations[
                feature_name
            ]
        )

        summary_lines.append(
            f"{feature_name:<24}: "
            f"Cramer's V = "
            f"{association['cramers_v']:.6f} "
            f"({association['association_strength']})"
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
                "Source folders and filenames are "
                "metadata and are not directly seen "
                "by the model."
            ),
            (
                "However, strong source association "
                "shows that the classes came from "
                "different collection pipelines."
            ),
            (
                "Strong image-geometry association "
                "can affect cropping, scaling, "
                "sharpness and page composition."
            ),
            "",
            "IMPORTANT",
            "-" * 80,
            "No model training was performed.",
            "No screenshot was modified.",
            "No manifest or split was modified.",
            "=" * 80,
        ]
    )

    summary_text = "\n".join(
        summary_lines
    )

    with open(
        OUTPUT_DIR
        / "original_dimensions_sources_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            summary_text
        )

    audit_json = {
        "audit_result": audit_result,
        "manifest_path": str(
            MANIFEST_PATH
        ),
        "manifest_sha256": (
            calculate_sha256(
                MANIFEST_PATH
            )
        ),
        "original_dataset_path": str(
            ORIGINAL_DATASET_DIR
        ),
        "manifest_images": len(
            inventory_rows
        ),
        "successfully_inspected": len(
            usable_rows
        ),
        "resolution_rate": (
            resolution_rate
        ),
        "class_counts": dict(
            class_counts
        ),
        "top_dimensions": {
            class_name: [
                {
                    "dimensions": dimensions,
                    "count": count,
                }
                for dimensions, count in (
                    top_dimensions[
                        class_name
                    ]
                )
            ]
            for class_name in CLASS_NAMES
        },
        "association_results": (
            association_rows
        ),
    }

    with open(
        OUTPUT_DIR
        / "original_dimensions_sources_audit.json",
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
    print(
        "MODEL 1 ORIGINAL DIMENSIONS "
        "AND SOURCES AUDIT"
    )
    print("=" * 80)

    print(
        f"Manifest            : "
        f"{MANIFEST_PATH}"
    )

    print(
        f"Original dataset    : "
        f"{ORIGINAL_DATASET_DIR}"
    )

    print(
        f"Output directory    : "
        f"{OUTPUT_DIR}"
    )

    print()

    if not ORIGINAL_DATASET_DIR.exists():
        raise FileNotFoundError(
            "Original dataset directory "
            f"was not found:\n"
            f"{ORIGINAL_DATASET_DIR}"
        )

    print(
        "Reading complete split manifest..."
    )

    manifest_records = read_manifest()

    print(
        f"Manifest images     : "
        f"{len(manifest_records)}"
    )

    print()

    inventory_rows = (
        inspect_original_images(
            manifest_records
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Creating audit reports..."
    )

    write_csv(
        OUTPUT_DIR
        / "original_image_inventory.csv",
        [
            "manifest_index",
            "split",
            "class_name",
            "numeric_label",
            "viewport_action",
            "prepared_relative_path",
            "original_relative_path",
            "source_folder",
            "filename_pattern",
            "resolved_original_path",
            "file_status",
            "read_status",
            "original_width",
            "original_height",
            "original_dimensions",
            "original_aspect_ratio",
            "aspect_ratio_bucket",
            "orientation",
            "megapixels",
            "image_size_bucket",
            "target_dimension_match",
            "geometric_change_required",
            "file_extension",
            "detected_image_format",
            "frame_count",
            "error_message",
        ],
        inventory_rows,
    )

    category_count_rows = (
        build_category_count_rows(
            inventory_rows
        )
    )

    write_csv(
        OUTPUT_DIR
        / "original_feature_category_counts.csv",
        [
            "scope",
            "feature",
            "category",
            "class_name",
            "image_count",
            "class_total",
            "percentage_within_class",
        ],
        category_count_rows,
    )

    association_rows = (
        build_association_rows(
            inventory_rows
        )
    )

    write_csv(
        OUTPUT_DIR
        / "original_feature_class_association.csv",
        [
            "scope",
            "feature",
            "usable_images",
            "category_count",
            "cramers_v",
            "association_strength",
            "model_visibility_note",
        ],
        association_rows,
    )

    dimension_action_rows = (
        build_dimension_action_rows(
            inventory_rows
        )
    )

    write_csv(
        OUTPUT_DIR
        / "viewport_original_dimension_counts.csv",
        [
            "split",
            "class_name",
            "viewport_action",
            "original_dimensions",
            "image_count",
        ],
        dimension_action_rows,
    )

    summary_text = create_summary(
        inventory_rows=inventory_rows,
        association_rows=association_rows,
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
        print(
            "ORIGINAL DIMENSIONS AND "
            "SOURCES AUDIT FAILED"
        )
        print("=" * 80)
        print(error)
        print()
        print(
            "No model, threshold, screenshot, "
            "manifest or split was modified."
        )

        sys.exit(1)