"""
13_build_uniform_top_viewport_dataset.py

Create a new aspect-preserving screenshot dataset for Model 1.

Every image in both classes receives exactly the same operation:

1. Apply EXIF orientation.
2. Convert to RGB.
3. Resize to width 384 while preserving the original aspect ratio.
4. Keep the top 160 pixels.
5. Save as a 384x160 PNG.

Important:
- Original screenshots are never modified.
- Existing prepared datasets are never modified.
- Existing TRAIN, VALIDATION and TEST assignments are preserved.
- The previously inspected TEST split is diagnostic only, not a new final test.
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
    / "prepared_login_uniform_top_viewport_384x160"
)

OUTPUT_REPORT_DIR = (
    PROJECT_DIR
    / "reports"
    / "13_uniform_top_viewport_dataset"
)

COMPLETE_OUTPUT_MANIFEST_PATH = (
    OUTPUT_REPORT_DIR
    / "complete_uniform_top_viewport_manifest.csv"
)

TARGET_WIDTH = 384
TARGET_HEIGHT = 160

PROCESSING_ACTION = (
    "UNIFORM_WIDTH_SCALE_TOP_VIEWPORT"
)

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

CLASS_TO_NUMERIC_LABEL = {
    "NOT_MALICIOUS": 0,
    "MALICIOUS": 1,
}

REQUIRED_SOURCE_COLUMNS = {
    "split",
    "class_label",
    "numeric_label",
    "prepared_relative_path",
    "original_relative_path",
    "viewport_action",
    "output_sha256",
}

OUTPUT_FIELDNAMES = [
    "split",
    "class_label",
    "numeric_label",
    "prepared_relative_path",
    "original_relative_path",
    "processing_action",
    "viewport_action",
    "source_viewport_action",
    "original_width",
    "original_height",
    "original_aspect_ratio",
    "resized_width_before_crop",
    "resized_height_before_crop",
    "pixels_removed_below_viewport",
    "output_width",
    "output_height",
    "output_colour_mode",
    "output_format",
    "interpolation",
    "original_sha256",
    "output_sha256",
]


# Pillow renamed its resampling constants in newer versions.
try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def calculate_sha256(file_path):
    """Calculate a file SHA256 without modifying the file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


def normalise_relative_path(value, field_name, row_number):
    """Convert a manifest path into a safe relative Path."""

    text = str(value).strip().replace("\\", "/")

    if not text:
        raise RuntimeError(
            f"Empty {field_name} at manifest row "
            f"{row_number}."
        )

    relative_path = Path(text)

    if relative_path.is_absolute():
        raise RuntimeError(
            f"{field_name} must be relative at manifest "
            f"row {row_number}: {text}"
        )

    if any(part in {"", ".", ".."} for part in relative_path.parts):
        raise RuntimeError(
            f"Unsafe {field_name} at manifest row "
            f"{row_number}: {text}"
        )

    return relative_path


def make_output_relative_path(row, row_number):
    """Create a class-based PNG path for a prepared image."""

    old_prepared_path = normalise_relative_path(
        row["prepared_relative_path"],
        "prepared_relative_path",
        row_number,
    )

    class_label = row["class_label"].strip().upper()

    return (
        Path(class_label)
        / old_prepared_path.with_suffix(".png").name
    )


def ensure_new_output_locations():
    """Refuse to overwrite a previous or partial build."""

    existing_paths = [
        path
        for path in [
            OUTPUT_DATASET_DIR,
            OUTPUT_REPORT_DIR,
        ]
        if path.exists()
    ]

    if existing_paths:
        formatted_paths = "\n".join(
            f"  - {path}"
            for path in existing_paths
        )

        raise RuntimeError(
            "The new output location already exists. "
            "Nothing was overwritten.\n\n"
            f"{formatted_paths}\n\n"
            "If this is an incomplete earlier attempt, "
            "inspect it first and then move or remove only "
            "that specific output folder before rerunning."
        )


def write_csv(output_path, rows):
    """Write a manifest CSV."""

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# MANIFEST VALIDATION
# =============================================================================

def read_and_validate_source_manifest():
    """Read the source manifest and validate labels and split counts."""

    if not SOURCE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Source manifest was not found:\n"
            f"{SOURCE_MANIFEST_PATH}"
        )

    with open(
        SOURCE_MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)

    missing_columns = (
        REQUIRED_SOURCE_COLUMNS
        .difference(fieldnames)
    )

    if missing_columns:
        raise RuntimeError(
            "Source manifest is missing columns:\n"
            f"{sorted(missing_columns)}"
        )

    if len(rows) != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_IMAGES} manifest "
            f"rows, but found {len(rows)}."
        )

    class_counts = Counter()
    split_counts = Counter()
    split_class_counts = Counter()

    seen_original_paths = set()
    seen_output_paths = set()

    validated_rows = []

    for row_number, row in enumerate(rows, start=2):
        split = row["split"].strip().upper()
        class_label = row["class_label"].strip().upper()

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

        try:
            numeric_label = int(
                row["numeric_label"].strip()
            )
        except ValueError as error:
            raise RuntimeError(
                f"Invalid numeric label at manifest "
                f"row {row_number}: "
                f"{row['numeric_label']!r}"
            ) from error

        expected_numeric_label = (
            CLASS_TO_NUMERIC_LABEL[class_label]
        )

        if numeric_label != expected_numeric_label:
            raise RuntimeError(
                f"Class/numeric-label mismatch at "
                f"manifest row {row_number}."
            )

        original_relative_path = normalise_relative_path(
            row["original_relative_path"],
            "original_relative_path",
            row_number,
        )

        output_relative_path = make_output_relative_path(
            row,
            row_number,
        )

        original_key = str(
            original_relative_path
        ).casefold()

        output_key = str(
            output_relative_path
        ).casefold()

        if original_key in seen_original_paths:
            raise RuntimeError(
                "Duplicate original path in manifest:\n"
                f"{original_relative_path}"
            )

        if output_key in seen_output_paths:
            raise RuntimeError(
                "Duplicate output path would be created:\n"
                f"{output_relative_path}"
            )

        seen_original_paths.add(original_key)
        seen_output_paths.add(output_key)

        validated_rows.append(
            {
                "source_row": row,
                "split": split,
                "class_label": class_label,
                "numeric_label": numeric_label,
                "original_relative_path": (
                    original_relative_path
                ),
                "output_relative_path": (
                    output_relative_path
                ),
            }
        )

        class_counts[class_label] += 1
        split_counts[split] += 1
        split_class_counts[(split, class_label)] += 1

    if dict(class_counts) != EXPECTED_CLASS_COUNTS:
        raise RuntimeError(
            "Unexpected class counts.\n"
            f"Expected: {EXPECTED_CLASS_COUNTS}\n"
            f"Actual  : {dict(class_counts)}"
        )

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

    return validated_rows


# =============================================================================
# PREFLIGHT
# =============================================================================

def preflight_original_images(records):
    """Validate every original before creating the output dataset."""

    preflight_records = []

    for index, record in enumerate(records, start=1):
        original_path = (
            ORIGINAL_DATASET_DIR
            / record["original_relative_path"]
        )

        if not original_path.is_file():
            raise FileNotFoundError(
                "Original screenshot was not found:\n"
                f"{original_path}"
            )

        try:
            with Image.open(original_path) as image:
                detected_format = image.format or "UNKNOWN"
                oriented_image = ImageOps.exif_transpose(image)
                original_width, original_height = (
                    oriented_image.size
                )
        except Exception as error:
            raise RuntimeError(
                "Could not read original screenshot:\n"
                f"{original_path}\n{error}"
            ) from error

        if original_width <= 0 or original_height <= 0:
            raise RuntimeError(
                "Invalid original dimensions:\n"
                f"{original_path}\n"
                f"{original_width}x{original_height}"
            )

        resized_height = int(round(
            original_height
            * TARGET_WIDTH
            / original_width
        ))

        if resized_height < TARGET_HEIGHT:
            raise RuntimeError(
                "An image is too wide to produce the requested "
                "top viewport without padding:\n"
                f"{original_path}\n"
                f"Original: {original_width}x{original_height}\n"
                f"Width-scaled result: "
                f"{TARGET_WIDTH}x{resized_height}"
            )

        output_path = (
            OUTPUT_DATASET_DIR
            / record["output_relative_path"]
        )

        enriched_record = dict(record)
        enriched_record.update(
            {
                "original_path": original_path,
                "output_path": output_path,
                "original_width": original_width,
                "original_height": original_height,
                "detected_original_format": (
                    detected_format
                ),
                "resized_height": resized_height,
            }
        )

        preflight_records.append(enriched_record)

        if index % 250 == 0 or index == len(records):
            print(
                f"Preflight checked {index}/{len(records)}"
            )

    return preflight_records


# =============================================================================
# DATASET CREATION
# =============================================================================

def create_one_image(record):
    """Create one aspect-preserved top-viewport image."""

    original_path = record["original_path"]
    output_path = record["output_path"]

    with Image.open(original_path) as source_image:
        image = ImageOps.exif_transpose(
            source_image
        ).convert("RGB")

        resized_image = image.resize(
            (
                TARGET_WIDTH,
                record["resized_height"],
            ),
            resample=LANCZOS,
        )

        top_viewport = resized_image.crop(
            (
                0,
                0,
                TARGET_WIDTH,
                TARGET_HEIGHT,
            )
        )

        if top_viewport.size != (
            TARGET_WIDTH,
            TARGET_HEIGHT,
        ):
            raise RuntimeError(
                "Unexpected output dimensions for:\n"
                f"{original_path}"
            )

        if top_viewport.mode != "RGB":
            raise RuntimeError(
                "Unexpected output colour mode for:\n"
                f"{original_path}"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        top_viewport.save(
            output_path,
            format="PNG",
            optimize=False,
            compress_level=6,
        )

    with Image.open(output_path) as saved_image:
        if saved_image.size != (
            TARGET_WIDTH,
            TARGET_HEIGHT,
        ):
            raise RuntimeError(
                "Saved file has incorrect dimensions:\n"
                f"{output_path}"
            )

        if saved_image.mode != "RGB":
            raise RuntimeError(
                "Saved file has incorrect colour mode:\n"
                f"{output_path}"
            )

        if saved_image.format != "PNG":
            raise RuntimeError(
                "Saved file has incorrect format:\n"
                f"{output_path}"
            )

    original_sha256 = calculate_sha256(
        original_path
    )

    output_sha256 = calculate_sha256(
        output_path
    )

    source_row = record["source_row"]

    return {
        "split": record["split"],
        "class_label": record["class_label"],
        "numeric_label": record["numeric_label"],
        "prepared_relative_path": str(
            record["output_relative_path"]
        ),
        "original_relative_path": str(
            record["original_relative_path"]
        ),
        "processing_action": PROCESSING_ACTION,
        "viewport_action": PROCESSING_ACTION,
        "source_viewport_action": (
            source_row["viewport_action"].strip()
        ),
        "original_width": record["original_width"],
        "original_height": record["original_height"],
        "original_aspect_ratio": (
            f"{record['original_width'] / record['original_height']:.10f}"
        ),
        "resized_width_before_crop": TARGET_WIDTH,
        "resized_height_before_crop": (
            record["resized_height"]
        ),
        "pixels_removed_below_viewport": (
            record["resized_height"]
            - TARGET_HEIGHT
        ),
        "output_width": TARGET_WIDTH,
        "output_height": TARGET_HEIGHT,
        "output_colour_mode": "RGB",
        "output_format": "PNG",
        "interpolation": "LANCZOS",
        "original_sha256": original_sha256,
        "output_sha256": output_sha256,
    }


def create_dataset(records):
    """Create the complete new prepared dataset."""

    output_rows = []

    for index, record in enumerate(records, start=1):
        output_rows.append(
            create_one_image(record)
        )

        if index % 250 == 0 or index == len(records):
            print(
                f"Created {index}/{len(records)} images"
            )

    return output_rows


# =============================================================================
# REPORTS
# =============================================================================

def validate_created_dataset(output_rows):
    """Validate final counts and uniform processing metadata."""

    if len(output_rows) != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_IMAGES} output rows, "
            f"but created {len(output_rows)}."
        )

    missing_files = [
        row["prepared_relative_path"]
        for row in output_rows
        if not (
            OUTPUT_DATASET_DIR
            / normalise_relative_path(
                row["prepared_relative_path"],
                "prepared_relative_path",
                0,
            )
        ).is_file()
    ]

    if missing_files:
        raise RuntimeError(
            f"Created dataset is missing "
            f"{len(missing_files)} files."
        )

    actions = {
        row["processing_action"]
        for row in output_rows
    }

    if actions != {PROCESSING_ACTION}:
        raise RuntimeError(
            "More than one processing action was found "
            "in the created dataset."
        )

    output_hashes = [
        row["output_sha256"]
        for row in output_rows
    ]

    if len(set(output_hashes)) != len(output_hashes):
        duplicate_count = (
            len(output_hashes)
            - len(set(output_hashes))
        )

        raise RuntimeError(
            f"The new dataset contains {duplicate_count} "
            "repeated output hashes. Review the source "
            "dataset before training."
        )


def save_reports(output_rows, source_manifest_sha256):
    """Save complete and split-specific manifests and summaries."""

    write_csv(
        COMPLETE_OUTPUT_MANIFEST_PATH,
        output_rows,
    )

    split_file_names = {
        "TRAIN": "train_manifest.csv",
        "VALIDATION": "validation_manifest.csv",
        "TEST": "test_manifest_diagnostic_only.csv",
    }

    for split, file_name in split_file_names.items():
        split_rows = [
            row
            for row in output_rows
            if row["split"] == split
        ]

        write_csv(
            OUTPUT_REPORT_DIR / file_name,
            split_rows,
        )

    class_counts = Counter(
        row["class_label"]
        for row in output_rows
    )

    split_counts = Counter(
        row["split"]
        for row in output_rows
    )

    crop_distribution = Counter(
        (
            row["class_label"],
            int(row["pixels_removed_below_viewport"]),
        )
        for row in output_rows
    )

    manifest_sha256 = calculate_sha256(
        COMPLETE_OUTPUT_MANIFEST_PATH
    )

    metrics = {
        "processing_action": PROCESSING_ACTION,
        "target_width": TARGET_WIDTH,
        "target_height": TARGET_HEIGHT,
        "total_images": len(output_rows),
        "class_counts": dict(class_counts),
        "split_counts": dict(split_counts),
        "source_manifest_path": str(
            SOURCE_MANIFEST_PATH
        ),
        "source_manifest_sha256": (
            source_manifest_sha256
        ),
        "output_dataset_path": str(
            OUTPUT_DATASET_DIR
        ),
        "output_manifest_path": str(
            COMPLETE_OUTPUT_MANIFEST_PATH
        ),
        "output_manifest_sha256": manifest_sha256,
        "test_split_status": "DIAGNOSTIC_ONLY",
    }

    with open(
        OUTPUT_REPORT_DIR / "build_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    summary_lines = [
        "=" * 80,
        "UNIFORM TOP-VIEWPORT DATASET CREATION COMPLETE",
        "=" * 80,
        "",
        f"Original dataset      : {ORIGINAL_DATASET_DIR}",
        f"Source manifest       : {SOURCE_MANIFEST_PATH}",
        f"Source manifest SHA256: {source_manifest_sha256}",
        f"New dataset           : {OUTPUT_DATASET_DIR}",
        f"New manifest          : {COMPLETE_OUTPUT_MANIFEST_PATH}",
        f"New manifest SHA256   : {manifest_sha256}",
        "",
        "PROCESSING",
        "-" * 80,
        f"Action                : {PROCESSING_ACTION}",
        f"Target dimensions     : {TARGET_WIDTH}x{TARGET_HEIGHT}",
        "Aspect ratio          : PRESERVED DURING RESIZE",
        f"Resize rule           : WIDTH -> {TARGET_WIDTH}",
        f"Viewport rule         : TOP {TARGET_HEIGHT} PIXELS",
        "Colour mode           : RGB",
        "Output format         : PNG",
        "Interpolation         : LANCZOS",
        "Same algorithm/classes: YES",
        "",
        "DATASET COUNTS",
        "-" * 80,
        f"Total images          : {len(output_rows)}",
        f"NOT_MALICIOUS         : {class_counts['NOT_MALICIOUS']}",
        f"MALICIOUS             : {class_counts['MALICIOUS']}",
        "",
        f"TRAIN                 : {split_counts['TRAIN']}",
        f"VALIDATION            : {split_counts['VALIDATION']}",
        f"TEST diagnostic only  : {split_counts['TEST']}",
        "",
        "PIXELS REMOVED BELOW TOP VIEWPORT",
        "-" * 80,
    ]

    for (class_label, removed_pixels), count in sorted(
        crop_distribution.items()
    ):
        summary_lines.append(
            f"{class_label:15s} | "
            f"{removed_pixels:3d} pixels : {count}"
        )

    summary_lines.extend(
        [
            "",
            "IMPORTANT",
            "-" * 80,
            "Both classes used exactly the same algorithm.",
            "No image was directly stretched to 384x160.",
            "Original aspect ratios were preserved during resizing.",
            "Original screenshots were not modified.",
            "Existing prepared datasets were not modified.",
            "Existing split assignments were preserved.",
            "The existing TEST split is diagnostic only because it was inspected.",
            "A separate external dataset is required for final evaluation.",
            "Audit and visually review this dataset before training.",
            "=" * 80,
        ]
    )

    summary_text = "\n".join(summary_lines)

    with open(
        OUTPUT_REPORT_DIR / "build_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(summary_text)

    return summary_text


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("BUILD UNIFORM TOP-VIEWPORT DATASET")
    print("=" * 80)
    print(f"Original dataset    : {ORIGINAL_DATASET_DIR}")
    print(f"Source manifest     : {SOURCE_MANIFEST_PATH}")
    print(f"New dataset         : {OUTPUT_DATASET_DIR}")
    print(
        f"Target dimensions   : "
        f"{TARGET_WIDTH}x{TARGET_HEIGHT}"
    )
    print(f"Processing action   : {PROCESSING_ACTION}")
    print()

    if not ORIGINAL_DATASET_DIR.is_dir():
        raise FileNotFoundError(
            "Original dataset directory was not found:\n"
            f"{ORIGINAL_DATASET_DIR}"
        )

    ensure_new_output_locations()

    print("Reading source manifest...")
    records = read_and_validate_source_manifest()
    print(f"Manifest records    : {len(records)}")
    print("Manifest counts and split assignments are valid.")
    print()

    print("Running original-image preflight checks...")
    records = preflight_original_images(records)
    print()
    print("All original screenshots passed the preflight checks.")
    print()

    source_manifest_sha256 = calculate_sha256(
        SOURCE_MANIFEST_PATH
    )

    OUTPUT_DATASET_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    OUTPUT_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    print("=" * 80)
    print("CREATING UNIFORM TOP-VIEWPORT DATASET")
    print("=" * 80)
    print()

    output_rows = create_dataset(records)

    print()
    print("Validating created dataset...")
    validate_created_dataset(output_rows)
    print("Created files and hashes are valid.")

    print()
    print("Creating manifests and reports...")

    summary_text = save_reports(
        output_rows,
        source_manifest_sha256,
    )

    print()
    print(summary_text)
    print()
    print(f"New dataset saved to: {OUTPUT_DATASET_DIR}")
    print(f"Reports saved to    : {OUTPUT_REPORT_DIR}")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print()
        print("=" * 80)
        print("UNIFORM TOP-VIEWPORT DATASET CREATION FAILED")
        print("=" * 80)
        print(error)
        print()
        print("No original screenshot was modified.")
        print("No existing prepared dataset was modified.")

        sys.exit(1)
