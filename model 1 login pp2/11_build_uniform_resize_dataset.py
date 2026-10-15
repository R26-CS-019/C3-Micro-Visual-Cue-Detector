"""
11_build_uniform_resize_dataset.py

Creates a new uniformly resized Model 1 dataset.

Every original screenshot, regardless of class, is processed using:

    RGB conversion
    Direct resize to 384x160
    LANCZOS interpolation
    PNG output

Important:
- Existing prepared images are not modified.
- Original images are not modified.
- Existing manifests and reports are not modified.
- Existing TRAIN/VALIDATION/TEST assignments are preserved.
- The new dataset is created in a separate folder.
"""

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(
    r"C:\Users\oshan\Desktop\model 1 login pp2"
)

ORIGINAL_DATASET_DIR = Path(
    r"C:\Users\oshan\Desktop\new model 1 pp2\dataset"
)

SOURCE_MANIFEST_PATH = (
    PROJECT_DIR
    / "reports"
    / "05_dataset_split"
    / "complete_split_manifest.csv"
)

OUTPUT_DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "prepared_login_uniform_384x160"
)

OUTPUT_REPORT_DIR = (
    PROJECT_DIR
    / "reports"
    / "11_uniform_resize_dataset"
)

TARGET_WIDTH = 384
TARGET_HEIGHT = 160

PROCESSING_ACTION = "UNIFORM_DIRECT_RESIZE"

EXPECTED_TOTAL_IMAGES = 2846

EXPECTED_CLASS_COUNTS = {
    "NOT_MALICIOUS": 1423,
    "MALICIOUS": 1423,
}

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


# =============================================================================
# GENERAL FUNCTIONS
# =============================================================================

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
        "NON_MALICIOUS",
        "NOT_PHISHING",
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


def output_directory_contains_files():
    """Check whether the new output folder already has files."""

    if not OUTPUT_DATASET_DIR.exists():
        return False

    return any(
        path.is_file()
        for path in OUTPUT_DATASET_DIR.rglob("*")
    )


# =============================================================================
# READ AND VALIDATE SOURCE MANIFEST
# =============================================================================

def read_source_manifest():
    """Read the existing duplicate-safe split manifest."""

    if not SOURCE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Source manifest was not found:\n"
            f"{SOURCE_MANIFEST_PATH}"
        )

    with open(
        SOURCE_MANIFEST_PATH,
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
            "Source manifest is missing columns:\n"
            f"{sorted(missing_columns)}"
        )

    records = []

    for manifest_index, row in enumerate(
        rows,
        start=1,
    ):
        split_name = normalize_split(
            row["split"]
        )

        if split_name not in {
            "TRAIN",
            "VALIDATION",
            "TEST",
        }:
            raise RuntimeError(
                f"Unsupported split at row "
                f"{manifest_index}: {split_name}"
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
                f"{manifest_index}."
            )

        numeric_label = (
            1
            if class_from_text == "MALICIOUS"
            else 0
        )

        original_relative_path = (
            row["original_relative_path"]
            .strip()
            .replace("\\", "/")
        )

        if not original_relative_path:
            raise RuntimeError(
                f"Missing original path at row "
                f"{manifest_index}."
            )

        original_path = (
            ORIGINAL_DATASET_DIR
            / Path(original_relative_path)
        )

        original_filename = Path(
            original_relative_path
        ).name

        output_filename = (
            Path(original_filename).stem
            + ".png"
        )

        output_relative_path = (
            Path(split_name)
            / class_from_text
            / output_filename
        )

        output_path = (
            OUTPUT_DATASET_DIR
            / output_relative_path
        )

        records.append(
            {
                "manifest_index": manifest_index,
                "split": split_name,
                "class_label": class_from_text,
                "numeric_label": numeric_label,
                "original_relative_path": (
                    original_relative_path
                ),
                "original_path": original_path,
                "old_prepared_relative_path": (
                    row["prepared_relative_path"]
                    .strip()
                    .replace("\\", "/")
                ),
                "old_viewport_action": normalize_text(
                    row["viewport_action"]
                ),
                "old_output_sha256": (
                    row["output_sha256"]
                    .strip()
                    .lower()
                ),
                "uniform_relative_path": (
                    output_relative_path
                    .as_posix()
                ),
                "uniform_output_path": output_path,
            }
        )

    return records


def validate_manifest_counts(records):
    """Validate expected dataset and split counts."""

    if len(records) != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_IMAGES} "
            f"manifest images, but found "
            f"{len(records)}."
        )

    class_counts = Counter(
        record["class_label"]
        for record in records
    )

    split_counts = Counter(
        record["split"]
        for record in records
    )

    split_class_counts = Counter(
        (
            record["split"],
            record["class_label"],
        )
        for record in records
    )

    for class_name, expected_count in (
        EXPECTED_CLASS_COUNTS.items()
    ):
        actual_count = class_counts[
            class_name
        ]

        if actual_count != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} "
                f"{class_name} images, but found "
                f"{actual_count}."
            )

    for split_name, expected_count in (
        EXPECTED_SPLIT_COUNTS.items()
    ):
        actual_count = split_counts[
            split_name
        ]

        if actual_count != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} "
                f"{split_name} images, but found "
                f"{actual_count}."
            )

    for key, expected_count in (
        EXPECTED_SPLIT_CLASS_COUNTS.items()
    ):
        actual_count = split_class_counts[
            key
        ]

        if actual_count != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} images "
                f"for {key}, but found "
                f"{actual_count}."
            )


# =============================================================================
# PREFLIGHT VALIDATION
# =============================================================================

def preflight_original_images(records):
    """
    Check every original image before creating output files.

    This prevents processing from starting when files are missing
    or output filenames collide.
    """

    print(
        "Running original-image preflight checks..."
    )

    missing_files = []
    unreadable_files = []
    seen_output_paths = set()

    for position, record in enumerate(
        records,
        start=1,
    ):
        original_path = record[
            "original_path"
        ]

        output_path_key = str(
            record["uniform_output_path"]
        ).lower()

        if output_path_key in seen_output_paths:
            raise RuntimeError(
                "Duplicate uniform output path:\n"
                f"{record['uniform_output_path']}"
            )

        seen_output_paths.add(
            output_path_key
        )

        if not original_path.exists():
            missing_files.append(
                str(original_path)
            )

        else:
            try:
                with Image.open(
                    original_path
                ) as image:
                    width, height = image.size

                    if width <= 0 or height <= 0:
                        raise RuntimeError(
                            "Invalid dimensions."
                        )

                    image.verify()

                record["original_width"] = width
                record["original_height"] = height

            except Exception as error:
                unreadable_files.append(
                    (
                        str(original_path),
                        str(error),
                    )
                )

        if (
            position % 250 == 0
            or position == len(records)
        ):
            print(
                f"Preflight checked "
                f"{position}/{len(records)}"
            )

    if missing_files:
        examples = "\n".join(
            missing_files[:10]
        )

        raise RuntimeError(
            f"{len(missing_files)} original "
            f"images were not found.\n\n"
            f"Examples:\n{examples}"
        )

    if unreadable_files:
        examples = "\n".join(
            f"{path} | {error}"
            for path, error
            in unreadable_files[:10]
        )

        raise RuntimeError(
            f"{len(unreadable_files)} original "
            f"images could not be read.\n\n"
            f"Examples:\n{examples}"
        )


# =============================================================================
# UNIFORM RESIZING
# =============================================================================

def resize_all_images(records):
    """Directly resize every class using exactly the same method."""

    output_manifest_rows = []

    print()
    print("=" * 80)
    print("CREATING UNIFORMLY RESIZED DATASET")
    print("=" * 80)
    print()

    for position, record in enumerate(
        records,
        start=1,
    ):
        original_path = record[
            "original_path"
        ]

        output_path = record[
            "uniform_output_path"
        ]

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with Image.open(
            original_path
        ) as original_image:
            original_image = (
                ImageOps.exif_transpose(
                    original_image
                )
            )

            original_image = (
                original_image.convert(
                    "RGB"
                )
            )

            resized_image = (
                original_image.resize(
                    (
                        TARGET_WIDTH,
                        TARGET_HEIGHT,
                    ),
                    resample=(
                        Image.Resampling.LANCZOS
                    ),
                )
            )

            resized_image.save(
                output_path,
                format="PNG",
                optimize=False,
                compress_level=6,
            )

        with Image.open(
            output_path
        ) as verification_image:
            output_width, output_height = (
                verification_image.size
            )

            output_mode = (
                verification_image.mode
            )

            output_format = (
                verification_image.format
            )

        if (
            output_width != TARGET_WIDTH
            or output_height != TARGET_HEIGHT
        ):
            raise RuntimeError(
                "Incorrect output dimensions:\n"
                f"{output_path}\n"
                f"Found: {output_width}x"
                f"{output_height}"
            )

        if output_mode != "RGB":
            raise RuntimeError(
                f"Incorrect output mode for:\n"
                f"{output_path}\n"
                f"Found: {output_mode}"
            )

        output_sha256 = calculate_sha256(
            output_path
        )

        output_manifest_rows.append(
            {
                "split": record["split"],
                "class_label": (
                    record["class_label"]
                ),
                "numeric_label": (
                    record["numeric_label"]
                ),
                "original_relative_path": (
                    record[
                        "original_relative_path"
                    ]
                ),
                "original_width": (
                    record["original_width"]
                ),
                "original_height": (
                    record["original_height"]
                ),
                "old_prepared_relative_path": (
                    record[
                        "old_prepared_relative_path"
                    ]
                ),
                "old_viewport_action": (
                    record[
                        "old_viewport_action"
                    ]
                ),
                "uniform_relative_path": (
                    record[
                        "uniform_relative_path"
                    ]
                ),
                "uniform_output_path": str(
                    output_path
                ),
                "processing_action": (
                    PROCESSING_ACTION
                ),
                "interpolation": "LANCZOS",
                "output_width": (
                    output_width
                ),
                "output_height": (
                    output_height
                ),
                "output_mode": (
                    output_mode
                ),
                "output_format": (
                    output_format
                ),
                "output_sha256": (
                    output_sha256
                ),
            }
        )

        if (
            position % 250 == 0
            or position == len(records)
        ):
            print(
                f"Created {position}/"
                f"{len(records)} images"
            )

    return output_manifest_rows


# =============================================================================
# REPORTS
# =============================================================================

def save_manifests(rows):
    """Save complete and split-specific uniform manifests."""

    fieldnames = [
        "split",
        "class_label",
        "numeric_label",
        "original_relative_path",
        "original_width",
        "original_height",
        "old_prepared_relative_path",
        "old_viewport_action",
        "uniform_relative_path",
        "uniform_output_path",
        "processing_action",
        "interpolation",
        "output_width",
        "output_height",
        "output_mode",
        "output_format",
        "output_sha256",
    ]

    write_csv(
        OUTPUT_REPORT_DIR
        / "complete_uniform_manifest.csv",
        fieldnames,
        rows,
    )

    for split_name, filename in [
        (
            "TRAIN",
            "train_uniform_manifest.csv",
        ),
        (
            "VALIDATION",
            "validation_uniform_manifest.csv",
        ),
        (
            "TEST",
            "test_uniform_manifest.csv",
        ),
    ]:
        split_rows = [
            row
            for row in rows
            if row["split"] == split_name
        ]

        write_csv(
            OUTPUT_REPORT_DIR
            / filename,
            fieldnames,
            split_rows,
        )


def create_summary(rows):
    """Create the uniform dataset summary."""

    class_counts = Counter(
        row["class_label"]
        for row in rows
    )

    split_counts = Counter(
        row["split"]
        for row in rows
    )

    split_class_counts = Counter(
        (
            row["split"],
            row["class_label"],
        )
        for row in rows
    )

    old_action_counts = Counter(
        (
            row["class_label"],
            row["old_viewport_action"],
        )
        for row in rows
    )

    output_dimensions = Counter(
        (
            row["output_width"],
            row["output_height"],
        )
        for row in rows
    )

    summary_lines = [
        "=" * 80,
        "UNIFORM RESIZE DATASET CREATION COMPLETE",
        "=" * 80,
        "",
        f"Original dataset      : {ORIGINAL_DATASET_DIR}",
        f"Source manifest       : {SOURCE_MANIFEST_PATH}",
        (
            f"Source manifest SHA256: "
            f"{calculate_sha256(SOURCE_MANIFEST_PATH)}"
        ),
        f"New dataset           : {OUTPUT_DATASET_DIR}",
        "",
        "PROCESSING",
        "-" * 80,
        (
            f"Action                : "
            f"{PROCESSING_ACTION}"
        ),
        (
            f"Target dimensions     : "
            f"{TARGET_WIDTH}x{TARGET_HEIGHT}"
        ),
        "Colour mode           : RGB",
        "Output format         : PNG",
        "Interpolation         : LANCZOS",
        "Cropping              : NONE FOR BOTH CLASSES",
        "",
        "DATASET COUNTS",
        "-" * 80,
        f"Total images          : {len(rows)}",
        (
            f"NOT_MALICIOUS         : "
            f"{class_counts['NOT_MALICIOUS']}"
        ),
        (
            f"MALICIOUS             : "
            f"{class_counts['MALICIOUS']}"
        ),
        "",
        (
            f"TRAIN                 : "
            f"{split_counts['TRAIN']}"
        ),
        (
            f"  NOT_MALICIOUS       : "
            f"{split_class_counts[('TRAIN', 'NOT_MALICIOUS')]}"
        ),
        (
            f"  MALICIOUS           : "
            f"{split_class_counts[('TRAIN', 'MALICIOUS')]}"
        ),
        (
            f"VALIDATION            : "
            f"{split_counts['VALIDATION']}"
        ),
        (
            f"  NOT_MALICIOUS       : "
            f"{split_class_counts[('VALIDATION', 'NOT_MALICIOUS')]}"
        ),
        (
            f"  MALICIOUS           : "
            f"{split_class_counts[('VALIDATION', 'MALICIOUS')]}"
        ),
        (
            f"TEST                  : "
            f"{split_counts['TEST']}"
        ),
        (
            f"  NOT_MALICIOUS       : "
            f"{split_class_counts[('TEST', 'NOT_MALICIOUS')]}"
        ),
        (
            f"  MALICIOUS           : "
            f"{split_class_counts[('TEST', 'MALICIOUS')]}"
        ),
        "",
        "OLD PREPROCESSING DISTRIBUTION",
        "-" * 80,
    ]

    for key in sorted(
        old_action_counts
    ):
        class_name, action = key

        summary_lines.append(
            f"{class_name:<15} | "
            f"{action:<15} : "
            f"{old_action_counts[key]}"
        )

    summary_lines.extend(
        [
            "",
            "NEW OUTPUT DIMENSIONS",
            "-" * 80,
        ]
    )

    for dimensions, count in sorted(
        output_dimensions.items()
    ):
        width, height = dimensions

        summary_lines.append(
            f"{width}x{height}: {count}"
        )

    summary_lines.extend(
        [
            "",
            "IMPORTANT",
            "-" * 80,
            (
                "All classes were processed using "
                "the same direct-resize operation."
            ),
            (
                "No TOP_CROP operation was used "
                "for either class."
            ),
            (
                "The existing prepared dataset was "
                "not modified."
            ),
            (
                "The original screenshots were not "
                "modified."
            ),
            (
                "Existing split assignments were "
                "preserved."
            ),
            (
                "The existing TEST split is now "
                "diagnostic only because its results "
                "were already inspected."
            ),
            (
                "A separate external dataset is still "
                "required for final evaluation."
            ),
            "=" * 80,
        ]
    )

    summary_text = "\n".join(
        summary_lines
    )

    with open(
        OUTPUT_REPORT_DIR
        / "uniform_resize_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(summary_text)

    run_information = {
        "processing_action": (
            PROCESSING_ACTION
        ),
        "target_width": TARGET_WIDTH,
        "target_height": TARGET_HEIGHT,
        "colour_mode": "RGB",
        "output_format": "PNG",
        "interpolation": "LANCZOS",
        "cropping": "NONE",
        "original_dataset": str(
            ORIGINAL_DATASET_DIR
        ),
        "source_manifest": str(
            SOURCE_MANIFEST_PATH
        ),
        "source_manifest_sha256": (
            calculate_sha256(
                SOURCE_MANIFEST_PATH
            )
        ),
        "output_dataset": str(
            OUTPUT_DATASET_DIR
        ),
        "total_images": len(rows),
        "class_counts": dict(
            class_counts
        ),
        "split_counts": dict(
            split_counts
        ),
    }

    with open(
        OUTPUT_REPORT_DIR
        / "uniform_resize_run.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_information,
            file,
            indent=4,
        )

    return summary_text


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("BUILD UNIFORM RESIZE DATASET")
    print("=" * 80)

    print(
        f"Original dataset    : "
        f"{ORIGINAL_DATASET_DIR}"
    )

    print(
        f"Source manifest     : "
        f"{SOURCE_MANIFEST_PATH}"
    )

    print(
        f"New dataset         : "
        f"{OUTPUT_DATASET_DIR}"
    )

    print(
        f"Target dimensions   : "
        f"{TARGET_WIDTH}x{TARGET_HEIGHT}"
    )

    print(
        f"Processing action   : "
        f"{PROCESSING_ACTION}"
    )

    print()

    if not ORIGINAL_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Original dataset was not found:\n"
            f"{ORIGINAL_DATASET_DIR}"
        )

    if output_directory_contains_files():
        raise RuntimeError(
            "The new output dataset folder already "
            "contains files:\n"
            f"{OUTPUT_DATASET_DIR}\n\n"
            "No files were overwritten."
        )

    print("Reading source manifest...")

    records = read_source_manifest()

    print(
        f"Manifest records    : "
        f"{len(records)}"
    )

    validate_manifest_counts(
        records
    )

    print(
        "Manifest counts and split assignments "
        "are valid."
    )

    print()

    preflight_original_images(
        records
    )

    print()
    print(
        "All original screenshots passed "
        "the preflight checks."
    )

    OUTPUT_DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_manifest_rows = resize_all_images(
        records
    )

    save_manifests(
        output_manifest_rows
    )

    summary_text = create_summary(
        output_manifest_rows
    )

    print()
    print(summary_text)

    print()
    print(
        f"New dataset saved to: "
        f"{OUTPUT_DATASET_DIR}"
    )

    print(
        f"Reports saved to    : "
        f"{OUTPUT_REPORT_DIR}"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print()
        print("=" * 80)
        print(
            "UNIFORM RESIZE DATASET "
            "CREATION FAILED"
        )
        print("=" * 80)
        print(error)
        print()
        print(
            "No original image, existing prepared "
            "image, model, split or threshold "
            "was modified."
        )

        sys.exit(1)