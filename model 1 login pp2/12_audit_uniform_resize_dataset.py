"""
12_audit_uniform_resize_dataset.py

Audits the uniformly resized Model 1 dataset.

Checks:
- All expected files exist.
- Every output is readable PNG.
- Every output is RGB.
- Every output is exactly 384x160.
- Every output SHA256 matches the manifest.
- Detects exact duplicates after uniform resizing.
- Detects cross-split and cross-class exact duplicates.
- Creates visual sheets comparing originals with resized images.

This script does not modify any dataset image.
"""

import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(
    r"C:\Users\oshan\Desktop\model 1 login pp2"
)

ORIGINAL_DATASET_DIR = Path(
    r"C:\Users\oshan\Desktop\new model 1 pp2\dataset"
)

UNIFORM_DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "prepared_login_uniform_384x160"
)

UNIFORM_MANIFEST_PATH = (
    PROJECT_DIR
    / "reports"
    / "11_uniform_resize_dataset"
    / "complete_uniform_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "reports"
    / "12_uniform_resize_audit"
)

REVIEW_SHEET_DIR = (
    OUTPUT_DIR
    / "review_sheets"
)

EXPECTED_TOTAL = 2846

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

TARGET_WIDTH = 384
TARGET_HEIGHT = 160
EXPECTED_MODE = "RGB"
EXPECTED_FORMAT = "PNG"
EXPECTED_ACTION = "UNIFORM_DIRECT_RESIZE"

RANDOM_SEED = 20260826
SAMPLES_PER_SPLIT_CLASS = 8

GRID_COLUMNS = 2
GRID_ROWS = 4

IMAGE_BOX_WIDTH = 370
IMAGE_BOX_HEIGHT = 155

TILE_WIDTH = 800
TILE_HEIGHT = 260
HEADER_HEIGHT = 75

SHEET_WIDTH = (
    GRID_COLUMNS
    * TILE_WIDTH
)

SHEET_HEIGHT = (
    HEADER_HEIGHT
    + GRID_ROWS
    * TILE_HEIGHT
)


# =============================================================================
# GENERAL FUNCTIONS
# =============================================================================

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
    """Normalize split names."""

    value = normalize_text(value)

    aliases = {
        "TRAINING": "TRAIN",
        "VAL": "VALIDATION",
        "VALID": "VALIDATION",
        "DEV": "VALIDATION",
        "TESTING": "TEST",
    }

    return aliases.get(
        value,
        value,
    )


def normalize_class(value):
    """Normalize class names."""

    value = normalize_text(value)

    if value in {
        "0",
        "NOT_MALICIOUS",
        "NON_MALICIOUS",
        "NOT_PHISHING",
        "LEGITIMATE",
        "BENIGN",
    }:
        return "NOT_MALICIOUS"

    if value in {
        "1",
        "MALICIOUS",
        "PHISHING",
        "PHISH",
    }:
        return "MALICIOUS"

    raise RuntimeError(
        f"Unsupported class value: "
        f"{value!r}"
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


def shorten_text(text, maximum_length):
    """Shorten long text for review sheets."""

    text = str(text)

    if len(text) <= maximum_length:
        return text

    return (
        text[: maximum_length - 3]
        + "..."
    )


# =============================================================================
# FONTS
# =============================================================================

def load_font(size, bold=False):
    """Load a Windows font with fallback."""

    if bold:
        font_paths = [
            Path(
                r"C:\Windows\Fonts\arialbd.ttf"
            ),
            Path(
                r"C:\Windows\Fonts\calibrib.ttf"
            ),
            Path(
                r"C:\Windows\Fonts\segoeuib.ttf"
            ),
        ]

    else:
        font_paths = [
            Path(
                r"C:\Windows\Fonts\arial.ttf"
            ),
            Path(
                r"C:\Windows\Fonts\calibri.ttf"
            ),
            Path(
                r"C:\Windows\Fonts\segoeui.ttf"
            ),
        ]

    for font_path in font_paths:
        if font_path.exists():
            return ImageFont.truetype(
                str(font_path),
                size=size,
            )

    return ImageFont.load_default()


TITLE_FONT = load_font(
    24,
    bold=True,
)

SUBTITLE_FONT = load_font(
    15,
    bold=False,
)

TILE_TITLE_FONT = load_font(
    16,
    bold=True,
)

TILE_TEXT_FONT = load_font(
    12,
    bold=False,
)

PANEL_FONT = load_font(
    12,
    bold=True,
)


# =============================================================================
# READ MANIFEST
# =============================================================================

def read_uniform_manifest():
    """Read and validate the uniform manifest."""

    if not UNIFORM_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Uniform manifest not found:\n"
            f"{UNIFORM_MANIFEST_PATH}"
        )

    with open(
        UNIFORM_MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        fieldnames = (
            reader.fieldnames
            or []
        )

        rows = list(reader)

    required_columns = {
        "split",
        "class_label",
        "numeric_label",
        "original_relative_path",
        "original_width",
        "original_height",
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
    }

    missing_columns = (
        required_columns.difference(
            fieldnames
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Uniform manifest is missing "
            f"columns:\n"
            f"{sorted(missing_columns)}"
        )

    records = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        split_name = normalize_split(
            row["split"]
        )

        class_name = normalize_class(
            row["class_label"]
        )

        numeric_class = normalize_class(
            row["numeric_label"]
        )

        if class_name != numeric_class:
            raise RuntimeError(
                f"Class mismatch at uniform "
                f"manifest row {index}."
            )

        uniform_relative_path = (
            row["uniform_relative_path"]
            .strip()
            .replace("\\", "/")
        )

        manifest_absolute_path = (
            row["uniform_output_path"]
            .strip()
        )

        if manifest_absolute_path:
            uniform_path = Path(
                manifest_absolute_path
            )
        else:
            uniform_path = (
                UNIFORM_DATASET_DIR
                / Path(
                    uniform_relative_path
                )
            )

        original_relative_path = (
            row["original_relative_path"]
            .strip()
            .replace("\\", "/")
        )

        original_path = (
            ORIGINAL_DATASET_DIR
            / Path(
                original_relative_path
            )
        )

        records.append(
            {
                "manifest_index": index,
                "split": split_name,
                "class_label": class_name,
                "numeric_label": (
                    1
                    if class_name
                    == "MALICIOUS"
                    else 0
                ),
                "original_relative_path": (
                    original_relative_path
                ),
                "original_path": (
                    original_path
                ),
                "manifest_original_width": (
                    row["original_width"]
                ),
                "manifest_original_height": (
                    row["original_height"]
                ),
                "old_viewport_action": (
                    normalize_text(
                        row[
                            "old_viewport_action"
                        ]
                    )
                ),
                "uniform_relative_path": (
                    uniform_relative_path
                ),
                "uniform_path": (
                    uniform_path
                ),
                "processing_action": (
                    normalize_text(
                        row[
                            "processing_action"
                        ]
                    )
                ),
                "interpolation": (
                    normalize_text(
                        row["interpolation"]
                    )
                ),
                "manifest_output_width": (
                    row["output_width"]
                ),
                "manifest_output_height": (
                    row["output_height"]
                ),
                "manifest_output_mode": (
                    normalize_text(
                        row["output_mode"]
                    )
                ),
                "manifest_output_format": (
                    normalize_text(
                        row["output_format"]
                    )
                ),
                "manifest_output_sha256": (
                    row["output_sha256"]
                    .strip()
                    .lower()
                ),
            }
        )

    return records


def validate_manifest_counts(records):
    """Validate total, split and class counts."""

    if len(records) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} rows, "
            f"but found {len(records)}."
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

    for class_name, expected in (
        EXPECTED_CLASS_COUNTS.items()
    ):
        actual = class_counts[
            class_name
        ]

        if actual != expected:
            raise RuntimeError(
                f"Expected {expected} "
                f"{class_name} images, "
                f"but found {actual}."
            )

    for split_name, expected in (
        EXPECTED_SPLIT_COUNTS.items()
    ):
        actual = split_counts[
            split_name
        ]

        if actual != expected:
            raise RuntimeError(
                f"Expected {expected} "
                f"{split_name} images, "
                f"but found {actual}."
            )

    for key, expected in (
        EXPECTED_SPLIT_CLASS_COUNTS.items()
    ):
        actual = split_class_counts[
            key
        ]

        if actual != expected:
            raise RuntimeError(
                f"Expected {expected} images "
                f"for {key}, but found "
                f"{actual}."
            )


# =============================================================================
# FILE AUDIT
# =============================================================================

def audit_uniform_files(records):
    """Audit every uniform output file."""

    audit_rows = []
    issue_rows = []

    print(
        "Auditing uniform files and hashes..."
    )

    for position, record in enumerate(
        records,
        start=1,
    ):
        uniform_path = record[
            "uniform_path"
        ]

        issues = []

        actual_width = ""
        actual_height = ""
        actual_mode = ""
        actual_format = ""
        actual_sha256 = ""

        if (
            record["processing_action"]
            != EXPECTED_ACTION
        ):
            issues.append(
                "MANIFEST_ACTION_MISMATCH"
            )

        if not uniform_path.exists():
            issues.append(
                "FILE_NOT_FOUND"
            )

        else:
            try:
                with Image.open(
                    uniform_path
                ) as image:
                    image.load()

                    (
                        actual_width,
                        actual_height,
                    ) = image.size

                    actual_mode = image.mode

                    actual_format = (
                        image.format
                        if image.format
                        else "UNKNOWN"
                    )

                if (
                    actual_width
                    != TARGET_WIDTH
                    or actual_height
                    != TARGET_HEIGHT
                ):
                    issues.append(
                        "DIMENSION_MISMATCH"
                    )

                if (
                    actual_mode
                    != EXPECTED_MODE
                ):
                    issues.append(
                        "MODE_MISMATCH"
                    )

                if (
                    str(actual_format).upper()
                    != EXPECTED_FORMAT
                ):
                    issues.append(
                        "FORMAT_MISMATCH"
                    )

                actual_sha256 = (
                    calculate_sha256(
                        uniform_path
                    )
                )

                if (
                    actual_sha256
                    != record[
                        "manifest_output_sha256"
                    ]
                ):
                    issues.append(
                        "SHA256_MISMATCH"
                    )

            except Exception as error:
                issues.append(
                    "IMAGE_READ_ERROR"
                )

                record[
                    "_read_error_message"
                ] = str(error)

        audit_status = (
            "PASS"
            if not issues
            else "FAIL"
        )

        audit_row = {
            "manifest_index": (
                record["manifest_index"]
            ),
            "split": record["split"],
            "class_label": (
                record["class_label"]
            ),
            "numeric_label": (
                record["numeric_label"]
            ),
            "uniform_relative_path": (
                record[
                    "uniform_relative_path"
                ]
            ),
            "uniform_output_path": str(
                uniform_path
            ),
            "processing_action": (
                record["processing_action"]
            ),
            "actual_width": actual_width,
            "actual_height": actual_height,
            "actual_mode": actual_mode,
            "actual_format": (
                str(actual_format).upper()
            ),
            "manifest_sha256": (
                record[
                    "manifest_output_sha256"
                ]
            ),
            "actual_sha256": actual_sha256,
            "audit_status": audit_status,
            "issue_codes": (
                ";".join(issues)
            ),
            "error_message": (
                record.get(
                    "_read_error_message",
                    "",
                )
            ),
        }

        audit_rows.append(
            audit_row
        )

        record[
            "_audit_status"
        ] = audit_status

        record[
            "_actual_sha256"
        ] = actual_sha256

        if issues:
            issue_rows.append(
                audit_row
            )

        if (
            position % 250 == 0
            or position == len(records)
        ):
            print(
                f"Audited {position}/"
                f"{len(records)}"
            )

    return (
        audit_rows,
        issue_rows,
    )


# =============================================================================
# EXACT DUPLICATE AUDIT
# =============================================================================

def audit_exact_duplicates(records):
    """Find exact hashes created by uniform resizing."""

    hash_groups = defaultdict(
        list
    )

    for record in records:
        if (
            record.get(
                "_audit_status"
            )
            != "PASS"
        ):
            continue

        output_hash = record.get(
            "_actual_sha256",
            "",
        )

        if output_hash:
            hash_groups[
                output_hash
            ].append(record)

    duplicate_rows = []

    duplicate_group_count = 0
    cross_split_group_count = 0
    cross_class_group_count = 0

    for output_hash, members in sorted(
        hash_groups.items()
    ):
        if len(members) < 2:
            continue

        duplicate_group_count += 1

        split_names = sorted(
            {
                member["split"]
                for member in members
            }
        )

        class_names = sorted(
            {
                member["class_label"]
                for member in members
            }
        )

        cross_split = (
            len(split_names) > 1
        )

        cross_class = (
            len(class_names) > 1
        )

        if cross_split:
            cross_split_group_count += 1

        if cross_class:
            cross_class_group_count += 1

        group_id = (
            f"DUP-{duplicate_group_count:04d}"
        )

        for member in members:
            duplicate_rows.append(
                {
                    "duplicate_group_id": (
                        group_id
                    ),
                    "output_sha256": (
                        output_hash
                    ),
                    "group_size": (
                        len(members)
                    ),
                    "cross_split": (
                        "YES"
                        if cross_split
                        else "NO"
                    ),
                    "cross_class": (
                        "YES"
                        if cross_class
                        else "NO"
                    ),
                    "group_splits": (
                        "|".join(
                            split_names
                        )
                    ),
                    "group_classes": (
                        "|".join(
                            class_names
                        )
                    ),
                    "member_split": (
                        member["split"]
                    ),
                    "member_class": (
                        member[
                            "class_label"
                        ]
                    ),
                    "member_path": str(
                        member[
                            "uniform_path"
                        ]
                    ),
                }
            )

    duplicate_summary = {
        "duplicate_groups": (
            duplicate_group_count
        ),
        "cross_split_groups": (
            cross_split_group_count
        ),
        "cross_class_groups": (
            cross_class_group_count
        ),
    }

    return (
        duplicate_rows,
        duplicate_summary,
    )


# =============================================================================
# REVIEW SHEETS
# =============================================================================

def fit_image_to_canvas(
    image_path,
    width,
    height,
):
    """Display an image without changing its aspect ratio."""

    with Image.open(
        image_path
    ) as image:
        image = ImageOps.exif_transpose(
            image
        )

        image = image.convert(
            "RGB"
        )

        image.thumbnail(
            (
                width,
                height,
            ),
            Image.Resampling.LANCZOS,
        )

        canvas = Image.new(
            "RGB",
            (
                width,
                height,
            ),
            color=(
                225,
                225,
                225,
            ),
        )

        paste_x = (
            width
            - image.width
        ) // 2

        paste_y = (
            height
            - image.height
        ) // 2

        canvas.paste(
            image,
            (
                paste_x,
                paste_y,
            ),
        )

    return canvas


def create_review_sheet(
    selected_records,
    split_name,
    class_name,
    output_path,
):
    """Create one original-versus-uniform review sheet."""

    sheet = Image.new(
        "RGB",
        (
            SHEET_WIDTH,
            SHEET_HEIGHT,
        ),
        color=(
            240,
            242,
            245,
        ),
    )

    draw = ImageDraw.Draw(
        sheet
    )

    draw.rectangle(
        [
            0,
            0,
            SHEET_WIDTH,
            HEADER_HEIGHT,
        ],
        fill=(
            31,
            42,
            56,
        ),
    )

    title = (
        "Uniform Resize Quality Review"
    )

    subtitle = (
        f"Split: {split_name} | "
        f"Class: {class_name} | "
        f"Original vs direct resize"
    )

    draw.text(
        (
            16,
            10,
        ),
        title,
        font=TITLE_FONT,
        fill=(
            255,
            255,
            255,
        ),
    )

    draw.text(
        (
            16,
            43,
        ),
        subtitle,
        font=SUBTITLE_FONT,
        fill=(
            255,
            255,
            255,
        ),
    )

    for local_index, record in enumerate(
        selected_records
    ):
        grid_row = (
            local_index
            // GRID_COLUMNS
        )

        grid_column = (
            local_index
            % GRID_COLUMNS
        )

        tile_x = (
            grid_column
            * TILE_WIDTH
        )

        tile_y = (
            HEADER_HEIGHT
            + grid_row
            * TILE_HEIGHT
        )

        draw.rounded_rectangle(
            [
                tile_x + 5,
                tile_y + 5,
                tile_x
                + TILE_WIDTH
                - 5,
                tile_y
                + TILE_HEIGHT
                - 5,
            ],
            radius=8,
            fill=(
                255,
                255,
                255,
            ),
            outline=(
                145,
                145,
                145,
            ),
            width=2,
        )

        original_x = (
            tile_x + 15
        )

        uniform_x = (
            tile_x
            + TILE_WIDTH
            - IMAGE_BOX_WIDTH
            - 15
        )

        label_y = (
            tile_y + 10
        )

        image_y = (
            tile_y + 30
        )

        draw.text(
            (
                original_x,
                label_y,
            ),
            "Original (aspect preserved)",
            font=PANEL_FONT,
            fill=(
                25,
                80,
                145,
            ),
        )

        draw.text(
            (
                uniform_x,
                label_y,
            ),
            "Uniform direct resize 384x160",
            font=PANEL_FONT,
            fill=(
                25,
                130,
                65,
            ),
        )

        try:
            original_panel = (
                fit_image_to_canvas(
                    record[
                        "original_path"
                    ],
                    IMAGE_BOX_WIDTH,
                    IMAGE_BOX_HEIGHT,
                )
            )

            sheet.paste(
                original_panel,
                (
                    original_x,
                    image_y,
                ),
            )

        except Exception:
            draw.rectangle(
                [
                    original_x,
                    image_y,
                    original_x
                    + IMAGE_BOX_WIDTH,
                    image_y
                    + IMAGE_BOX_HEIGHT,
                ],
                fill=(
                    230,
                    210,
                    210,
                ),
            )

            draw.text(
                (
                    original_x + 10,
                    image_y + 65,
                ),
                "ORIGINAL READ ERROR",
                font=TILE_TEXT_FONT,
                fill=(
                    170,
                    20,
                    20,
                ),
            )

        try:
            uniform_panel = (
                fit_image_to_canvas(
                    record[
                        "uniform_path"
                    ],
                    IMAGE_BOX_WIDTH,
                    IMAGE_BOX_HEIGHT,
                )
            )

            sheet.paste(
                uniform_panel,
                (
                    uniform_x,
                    image_y,
                ),
            )

        except Exception:
            draw.rectangle(
                [
                    uniform_x,
                    image_y,
                    uniform_x
                    + IMAGE_BOX_WIDTH,
                    image_y
                    + IMAGE_BOX_HEIGHT,
                ],
                fill=(
                    230,
                    210,
                    210,
                ),
            )

            draw.text(
                (
                    uniform_x + 10,
                    image_y + 65,
                ),
                "UNIFORM READ ERROR",
                font=TILE_TEXT_FONT,
                fill=(
                    170,
                    20,
                    20,
                ),
            )

        metadata_y = (
            image_y
            + IMAGE_BOX_HEIGHT
            + 7
        )

        filename = Path(
            record[
                "original_relative_path"
            ]
        ).name

        metadata_lines = [
            (
                f"Sample {local_index + 1:02d} | "
                f"Original: "
                f"{record['manifest_original_width']}x"
                f"{record['manifest_original_height']} | "
                f"Old action: "
                f"{record['old_viewport_action']}"
            ),
            (
                f"File: "
                f"{shorten_text(filename, 85)}"
            ),
            (
                f"Uniform SHA256: "
                f"{record['_actual_sha256'][:20]}..."
            ),
        ]

        for line_index, line in enumerate(
            metadata_lines
        ):
            draw.text(
                (
                    original_x,
                    metadata_y
                    + line_index * 16,
                ),
                line,
                font=(
                    TILE_TITLE_FONT
                    if line_index == 0
                    else TILE_TEXT_FONT
                ),
                fill=(
                    35,
                    35,
                    35,
                ),
            )

    sheet.save(
        output_path,
        format="PNG",
        optimize=True,
    )


def create_all_review_sheets(records):
    """Create deterministic samples for every split and class."""

    REVIEW_SHEET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    random_generator = random.Random(
        RANDOM_SEED
    )

    review_manifest_rows = []
    review_sheet_count = 0

    for split_name in [
        "TRAIN",
        "VALIDATION",
        "TEST",
    ]:
        for class_name in [
            "NOT_MALICIOUS",
            "MALICIOUS",
        ]:
            eligible_records = [
                record
                for record in records
                if (
                    record["split"]
                    == split_name
                    and record[
                        "class_label"
                    ]
                    == class_name
                    and record.get(
                        "_audit_status"
                    )
                    == "PASS"
                )
            ]

            eligible_records.sort(
                key=lambda record: (
                    record[
                        "uniform_relative_path"
                    ]
                )
            )

            if (
                len(eligible_records)
                < SAMPLES_PER_SPLIT_CLASS
            ):
                raise RuntimeError(
                    f"Not enough valid records "
                    f"for {split_name}/"
                    f"{class_name}."
                )

            selected_records = (
                random_generator.sample(
                    eligible_records,
                    SAMPLES_PER_SPLIT_CLASS,
                )
            )

            selected_records.sort(
                key=lambda record: (
                    record[
                        "uniform_relative_path"
                    ]
                )
            )

            output_filename = (
                f"{split_name}_"
                f"{class_name}_"
                f"review.png"
            )

            output_path = (
                REVIEW_SHEET_DIR
                / output_filename
            )

            create_review_sheet(
                selected_records=(
                    selected_records
                ),
                split_name=split_name,
                class_name=class_name,
                output_path=output_path,
            )

            review_sheet_count += 1

            for sample_number, record in enumerate(
                selected_records,
                start=1,
            ):
                review_manifest_rows.append(
                    {
                        "review_sheet": (
                            output_filename
                        ),
                        "sample_number": (
                            sample_number
                        ),
                        "split": split_name,
                        "class_label": (
                            class_name
                        ),
                        "original_relative_path": (
                            record[
                                "original_relative_path"
                            ]
                        ),
                        "uniform_relative_path": (
                            record[
                                "uniform_relative_path"
                            ]
                        ),
                        "old_viewport_action": (
                            record[
                                "old_viewport_action"
                            ]
                        ),
                        "uniform_sha256": (
                            record[
                                "_actual_sha256"
                            ]
                        ),
                    }
                )

            print(
                f"Created review sheet: "
                f"{output_filename}"
            )

    write_csv(
        OUTPUT_DIR
        / "review_sample_manifest.csv",
        [
            "review_sheet",
            "sample_number",
            "split",
            "class_label",
            "original_relative_path",
            "uniform_relative_path",
            "old_viewport_action",
            "uniform_sha256",
        ],
        review_manifest_rows,
    )

    return review_sheet_count


# =============================================================================
# SUMMARY
# =============================================================================

def create_summary(
    records,
    audit_rows,
    issue_rows,
    duplicate_summary,
    review_sheet_count,
):
    """Create the final audit summary."""

    pass_count = sum(
        row["audit_status"] == "PASS"
        for row in audit_rows
    )

    fail_count = sum(
        row["audit_status"] == "FAIL"
        for row in audit_rows
    )

    issue_counts = Counter()

    for row in issue_rows:
        issue_codes = (
            row["issue_codes"]
            .split(";")
        )

        for issue_code in issue_codes:
            if issue_code:
                issue_counts[
                    issue_code
                ] += 1

    if fail_count > 0:
        audit_result = (
            "FAILED_FILE_OR_HASH_VALIDATION"
        )

    elif (
        duplicate_summary[
            "cross_split_groups"
        ] > 0
        or duplicate_summary[
            "cross_class_groups"
        ] > 0
    ):
        audit_result = (
            "PASS_WITH_DUPLICATE_REVIEW_REQUIRED"
        )

    else:
        audit_result = (
            "PASS_READY_FOR_VISUAL_REVIEW"
        )

    summary_lines = [
        "=" * 80,
        "UNIFORM RESIZE DATASET AUDIT COMPLETE",
        "=" * 80,
        "",
        f"Uniform dataset      : {UNIFORM_DATASET_DIR}",
        f"Uniform manifest     : {UNIFORM_MANIFEST_PATH}",
        (
            f"Manifest SHA256      : "
            f"{calculate_sha256(UNIFORM_MANIFEST_PATH)}"
        ),
        "",
        "FILE VALIDATION",
        "-" * 80,
        f"Manifest images      : {len(records)}",
        f"Passed               : {pass_count}",
        f"Failed               : {fail_count}",
        f"Expected dimensions  : {TARGET_WIDTH}x{TARGET_HEIGHT}",
        f"Expected mode        : {EXPECTED_MODE}",
        f"Expected format      : {EXPECTED_FORMAT}",
        f"Expected action      : {EXPECTED_ACTION}",
        "",
        "ISSUES",
        "-" * 80,
    ]

    if issue_counts:
        for issue_code, count in sorted(
            issue_counts.items()
        ):
            summary_lines.append(
                f"{issue_code:<30}: "
                f"{count}"
            )
    else:
        summary_lines.append(
            "No file, format, dimension or "
            "SHA256 issues were found."
        )

    summary_lines.extend(
        [
            "",
            "EXACT DUPLICATE AUDIT",
            "-" * 80,
            (
                f"Duplicate hash groups: "
                f"{duplicate_summary['duplicate_groups']}"
            ),
            (
                f"Cross-split groups   : "
                f"{duplicate_summary['cross_split_groups']}"
            ),
            (
                f"Cross-class groups   : "
                f"{duplicate_summary['cross_class_groups']}"
            ),
            "",
            "VISUAL REVIEW",
            "-" * 80,
            (
                f"Review sheets created: "
                f"{review_sheet_count}"
            ),
            (
                f"Samples per split/class: "
                f"{SAMPLES_PER_SPLIT_CLASS}"
            ),
            (
                f"Random seed          : "
                f"{RANDOM_SEED}"
            ),
            "",
            "AUDIT RESULT",
            "-" * 80,
            audit_result,
            "",
            "IMPORTANT",
            "-" * 80,
            (
                "No dataset image was modified "
                "by this audit."
            ),
            (
                "The visual sheets compare each "
                "original with its uniform resize."
            ),
            (
                "Check that text, forms and logos "
                "remain understandable."
            ),
            (
                "Do not train until the visual "
                "review sheets are inspected."
            ),
            "=" * 80,
        ]
    )

    summary_text = "\n".join(
        summary_lines
    )

    with open(
        OUTPUT_DIR
        / "uniform_resize_audit_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            summary_text
        )

    with open(
        OUTPUT_DIR
        / "uniform_resize_audit.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "audit_result": audit_result,
                "manifest_images": (
                    len(records)
                ),
                "passed": pass_count,
                "failed": fail_count,
                "issue_counts": dict(
                    issue_counts
                ),
                "duplicate_summary": (
                    duplicate_summary
                ),
                "review_sheet_count": (
                    review_sheet_count
                ),
                "random_seed": (
                    RANDOM_SEED
                ),
            },
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
        "AUDIT UNIFORM RESIZE DATASET"
    )
    print("=" * 80)

    print(
        f"Dataset             : "
        f"{UNIFORM_DATASET_DIR}"
    )

    print(
        f"Manifest            : "
        f"{UNIFORM_MANIFEST_PATH}"
    )

    print(
        f"Output directory    : "
        f"{OUTPUT_DIR}"
    )

    print()

    if not UNIFORM_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Uniform dataset not found:\n"
            f"{UNIFORM_DATASET_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Reading uniform manifest..."
    )

    records = read_uniform_manifest()

    print(
        f"Manifest records    : "
        f"{len(records)}"
    )

    validate_manifest_counts(
        records
    )

    print(
        "Manifest counts are valid."
    )

    print()

    (
        audit_rows,
        issue_rows,
    ) = audit_uniform_files(
        records
    )

    write_csv(
        OUTPUT_DIR
        / "uniform_file_audit.csv",
        [
            "manifest_index",
            "split",
            "class_label",
            "numeric_label",
            "uniform_relative_path",
            "uniform_output_path",
            "processing_action",
            "actual_width",
            "actual_height",
            "actual_mode",
            "actual_format",
            "manifest_sha256",
            "actual_sha256",
            "audit_status",
            "issue_codes",
            "error_message",
        ],
        audit_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "uniform_file_issues.csv",
        [
            "manifest_index",
            "split",
            "class_label",
            "numeric_label",
            "uniform_relative_path",
            "uniform_output_path",
            "processing_action",
            "actual_width",
            "actual_height",
            "actual_mode",
            "actual_format",
            "manifest_sha256",
            "actual_sha256",
            "audit_status",
            "issue_codes",
            "error_message",
        ],
        issue_rows,
    )

    print()
    print(
        "Checking exact duplicate hashes..."
    )

    (
        duplicate_rows,
        duplicate_summary,
    ) = audit_exact_duplicates(
        records
    )

    write_csv(
        OUTPUT_DIR
        / "exact_duplicate_hash_groups.csv",
        [
            "duplicate_group_id",
            "output_sha256",
            "group_size",
            "cross_split",
            "cross_class",
            "group_splits",
            "group_classes",
            "member_split",
            "member_class",
            "member_path",
        ],
        duplicate_rows,
    )

    print(
        f"Duplicate groups     : "
        f"{duplicate_summary['duplicate_groups']}"
    )

    print(
        f"Cross-split groups   : "
        f"{duplicate_summary['cross_split_groups']}"
    )

    print(
        f"Cross-class groups   : "
        f"{duplicate_summary['cross_class_groups']}"
    )

    if issue_rows:
        print()
        print(
            "File issues were found. "
            "Visual sheets will not be created."
        )

        review_sheet_count = 0

    else:
        print()
        print(
            "Creating visual review sheets..."
        )

        review_sheet_count = (
            create_all_review_sheets(
                records
            )
        )

    summary_text = create_summary(
        records=records,
        audit_rows=audit_rows,
        issue_rows=issue_rows,
        duplicate_summary=(
            duplicate_summary
        ),
        review_sheet_count=(
            review_sheet_count
        ),
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
            "UNIFORM RESIZE DATASET "
            "AUDIT FAILED"
        )
        print("=" * 80)
        print(error)
        print()
        print(
            "No original image, uniform image, "
            "model, split or manifest "
            "was modified."
        )

        sys.exit(1)