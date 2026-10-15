"""
14_audit_and_compare_top_viewport_dataset.py

Audit the new uniform top-viewport dataset and compare every image with the
corresponding image in the older prepared dataset.

This script performs read-only checks on dataset images and manifests. It:

- verifies all new files, dimensions, modes, formats and SHA256 hashes;
- checks exact duplicate hashes, including cross-split and cross-class groups;
- compares old and new files using pixel equality, MAE, RMSE, pHash and dHash;
- creates comparison CSV/JSON/text reports;
- creates visual review sheets for the most changed images in each class.

No screenshot, manifest, split or trained model is modified.
"""

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(
    r"C:\Users\oshan\Desktop\model 1 login pp2"
)

OLD_DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "prepared_login_384x160"
)

NEW_DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "prepared_login_uniform_top_viewport_384x160"
)

NEW_MANIFEST_PATH = (
    PROJECT_DIR
    / "reports"
    / "13_uniform_top_viewport_dataset"
    / "complete_uniform_top_viewport_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "reports"
    / "14_top_viewport_comparison_audit"
)

TARGET_WIDTH = 384
TARGET_HEIGHT = 160
EXPECTED_IMAGES = 2846
EXPECTED_ACTION = "UNIFORM_WIDTH_SCALE_TOP_VIEWPORT"

EXPECTED_CLASS_COUNTS = {
    "NOT_MALICIOUS": 1423,
    "MALICIOUS": 1423,
}

EXPECTED_SPLIT_COUNTS = {
    "TRAIN": 1992,
    "VALIDATION": 426,
    "TEST": 428,
}

REVIEW_SAMPLES_PER_CLASS = 8

REQUIRED_MANIFEST_COLUMNS = {
    "split",
    "class_label",
    "numeric_label",
    "prepared_relative_path",
    "original_relative_path",
    "processing_action",
    "output_width",
    "output_height",
    "output_colour_mode",
    "output_format",
    "output_sha256",
}

COMPARISON_FIELDNAMES = [
    "manifest_index",
    "split",
    "class_label",
    "numeric_label",
    "prepared_relative_path",
    "original_relative_path",
    "processing_action",
    "old_file_found",
    "new_file_found",
    "new_manifest_hash_match",
    "old_width",
    "old_height",
    "old_mode",
    "old_format",
    "new_width",
    "new_height",
    "new_mode",
    "new_format",
    "old_file_sha256",
    "new_file_sha256",
    "file_sha256_identical",
    "pixels_identical",
    "mean_absolute_error_0_255",
    "normalized_mean_absolute_error",
    "root_mean_squared_error_0_255",
    "changed_pixel_fraction",
    "phash_hamming_distance",
    "dhash_hamming_distance",
    "similarity_category",
    "validation_status",
    "validation_issue",
]

DUPLICATE_FIELDNAMES = [
    "duplicate_group",
    "output_sha256",
    "group_size",
    "cross_split",
    "cross_class",
    "split",
    "class_label",
    "numeric_label",
    "prepared_relative_path",
]


try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS

try:
    BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    BILINEAR = Image.BILINEAR


# =============================================================================
# HASH AND PATH HELPERS
# =============================================================================

def calculate_sha256(file_path):
    """Calculate a SHA256 hash without changing the file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


def normalise_relative_path(value, row_number):
    """Convert a manifest value into a safe relative path."""

    text = str(value).strip().replace("\\", "/")

    if not text:
        raise RuntimeError(
            f"Empty prepared_relative_path at row {row_number}."
        )

    relative_path = Path(text)

    if relative_path.is_absolute():
        raise RuntimeError(
            "Prepared path must be relative at row "
            f"{row_number}: {text}"
        )

    if any(part in {"", ".", ".."} for part in relative_path.parts):
        raise RuntimeError(
            f"Unsafe prepared path at row {row_number}: {text}"
        )

    return relative_path


def bool_text(value):
    """Return a consistent CSV boolean string."""

    return "YES" if value else "NO"


def optional_float(value, digits=10):
    """Format a float for a CSV, preserving blank values."""

    if value is None:
        return ""

    return f"{float(value):.{digits}f}"


# =============================================================================
# PERCEPTUAL HASH FUNCTIONS
# =============================================================================

def create_dct_matrix(size):
    """Create an orthonormal DCT-II transform matrix."""

    matrix = np.zeros((size, size), dtype=np.float64)

    for frequency in range(size):
        scale = math.sqrt(1.0 / size)

        if frequency > 0:
            scale = math.sqrt(2.0 / size)

        for position in range(size):
            matrix[frequency, position] = (
                scale
                * math.cos(
                    math.pi
                    * (2 * position + 1)
                    * frequency
                    / (2 * size)
                )
            )

    return matrix


DCT_MATRIX_32 = create_dct_matrix(32)


def bits_to_integer(bits):
    """Convert a Boolean array to one Python integer."""

    value = 0

    for bit in np.asarray(bits).reshape(-1):
        value = (value << 1) | int(bool(bit))

    return value


def calculate_phash(image):
    """Calculate a 64-bit perceptual DCT hash."""

    grayscale = image.convert("L").resize(
        (32, 32),
        resample=LANCZOS,
    )

    pixels = np.asarray(
        grayscale,
        dtype=np.float64,
    )

    dct = (
        DCT_MATRIX_32
        @ pixels
        @ DCT_MATRIX_32.T
    )

    low_frequencies = dct[:8, :8]
    values_without_dc = low_frequencies.reshape(-1)[1:]
    median = float(np.median(values_without_dc))
    bits = low_frequencies > median

    return bits_to_integer(bits)


def calculate_dhash(image):
    """Calculate a 64-bit difference hash."""

    grayscale = image.convert("L").resize(
        (9, 8),
        resample=BILINEAR,
    )

    pixels = np.asarray(
        grayscale,
        dtype=np.int16,
    )

    bits = pixels[:, 1:] > pixels[:, :-1]

    return bits_to_integer(bits)


def hamming_distance(first_hash, second_hash):
    """Calculate bitwise Hamming distance."""

    return int((first_hash ^ second_hash).bit_count())


# =============================================================================
# MANIFEST
# =============================================================================

def read_and_validate_manifest():
    """Read and validate the new dataset manifest."""

    if not NEW_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "New dataset manifest was not found:\n"
            f"{NEW_MANIFEST_PATH}"
        )

    with open(
        NEW_MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)

    missing_columns = (
        REQUIRED_MANIFEST_COLUMNS.difference(fieldnames)
    )

    if missing_columns:
        raise RuntimeError(
            "New manifest is missing columns:\n"
            f"{sorted(missing_columns)}"
        )

    if len(rows) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} manifest rows, "
            f"but found {len(rows)}."
        )

    class_counts = Counter()
    split_counts = Counter()
    seen_paths = set()

    for row_number, row in enumerate(rows, start=2):
        split = row["split"].strip().upper()
        class_label = row["class_label"].strip().upper()
        action = row["processing_action"].strip().upper()

        if split not in EXPECTED_SPLIT_COUNTS:
            raise RuntimeError(
                f"Invalid split at row {row_number}: {split!r}"
            )

        if class_label not in EXPECTED_CLASS_COUNTS:
            raise RuntimeError(
                f"Invalid class at row {row_number}: {class_label!r}"
            )

        if action != EXPECTED_ACTION:
            raise RuntimeError(
                f"Unexpected processing action at row {row_number}: "
                f"{action!r}"
            )

        relative_path = normalise_relative_path(
            row["prepared_relative_path"],
            row_number,
        )

        path_key = str(relative_path).casefold()

        if path_key in seen_paths:
            raise RuntimeError(
                "Duplicate prepared path in manifest:\n"
                f"{relative_path}"
            )

        seen_paths.add(path_key)

        row["split"] = split
        row["class_label"] = class_label
        row["processing_action"] = action
        row["_relative_path"] = relative_path

        class_counts[class_label] += 1
        split_counts[split] += 1

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

    return rows


# =============================================================================
# IMAGE COMPARISON
# =============================================================================

def classify_similarity(
    pixels_identical,
    normalized_mae,
    phash_distance,
    dhash_distance,
):
    """Assign a conservative visual-similarity category."""

    if pixels_identical:
        return "PIXEL_IDENTICAL"

    if (
        normalized_mae <= 0.0025
        and phash_distance <= 1
        and dhash_distance <= 1
    ):
        return "NEAR_IDENTICAL"

    if (
        normalized_mae <= 0.015
        and phash_distance <= 4
        and dhash_distance <= 4
    ):
        return "HIGHLY_SIMILAR"

    if (
        normalized_mae <= 0.05
        and phash_distance <= 8
        and dhash_distance <= 8
    ):
        return "MODERATELY_SIMILAR"

    return "VISUALLY_DIFFERENT"


def compare_image_pair(old_path, new_path):
    """Read and compare one old/new image pair."""

    with Image.open(old_path) as old_source:
        old_format = old_source.format or "UNKNOWN"
        old_mode = old_source.mode
        old_size = old_source.size
        old_image = old_source.convert("RGB")

    with Image.open(new_path) as new_source:
        new_format = new_source.format or "UNKNOWN"
        new_mode = new_source.mode
        new_size = new_source.size
        new_image = new_source.convert("RGB")

    if old_image.size != new_image.size:
        raise RuntimeError(
            "Old and new image dimensions do not match."
        )

    old_array = np.asarray(old_image, dtype=np.int16)
    new_array = np.asarray(new_image, dtype=np.int16)

    difference = np.abs(old_array - new_array)

    pixels_identical = bool(
        np.array_equal(old_array, new_array)
    )

    mean_absolute_error = float(
        np.mean(difference)
    )

    normalized_mae = (
        mean_absolute_error / 255.0
    )

    rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    old_array.astype(np.float64)
                    - new_array.astype(np.float64)
                )
            )
        )
    )

    changed_pixel_fraction = float(
        np.mean(
            np.any(difference != 0, axis=2)
        )
    )

    old_phash = calculate_phash(old_image)
    new_phash = calculate_phash(new_image)
    old_dhash = calculate_dhash(old_image)
    new_dhash = calculate_dhash(new_image)

    phash_distance = hamming_distance(
        old_phash,
        new_phash,
    )

    dhash_distance = hamming_distance(
        old_dhash,
        new_dhash,
    )

    similarity_category = classify_similarity(
        pixels_identical=pixels_identical,
        normalized_mae=normalized_mae,
        phash_distance=phash_distance,
        dhash_distance=dhash_distance,
    )

    return {
        "old_width": old_size[0],
        "old_height": old_size[1],
        "old_mode": old_mode,
        "old_format": old_format,
        "new_width": new_size[0],
        "new_height": new_size[1],
        "new_mode": new_mode,
        "new_format": new_format,
        "pixels_identical": pixels_identical,
        "mean_absolute_error": mean_absolute_error,
        "normalized_mae": normalized_mae,
        "rmse": rmse,
        "changed_pixel_fraction": changed_pixel_fraction,
        "phash_distance": phash_distance,
        "dhash_distance": dhash_distance,
        "similarity_category": similarity_category,
    }


def audit_and_compare(rows):
    """Audit every new file and compare it with the old file."""

    comparison_rows = []

    for index, row in enumerate(rows, start=1):
        relative_path = row["_relative_path"]
        old_path = OLD_DATASET_DIR / relative_path
        new_path = NEW_DATASET_DIR / relative_path

        old_found = old_path.is_file()
        new_found = new_path.is_file()

        validation_issues = []
        comparison = None
        old_sha256 = ""
        new_sha256 = ""
        manifest_hash_match = False

        if not old_found:
            validation_issues.append("OLD_FILE_NOT_FOUND")

        if not new_found:
            validation_issues.append("NEW_FILE_NOT_FOUND")

        if new_found:
            new_sha256 = calculate_sha256(new_path)
            expected_hash = row["output_sha256"].strip().lower()
            manifest_hash_match = (
                bool(expected_hash)
                and new_sha256 == expected_hash
            )

            if not manifest_hash_match:
                validation_issues.append(
                    "NEW_SHA256_MISMATCH"
                )

        if old_found:
            old_sha256 = calculate_sha256(old_path)

        if old_found and new_found:
            try:
                comparison = compare_image_pair(
                    old_path,
                    new_path,
                )
            except Exception as error:
                validation_issues.append(
                    "IMAGE_COMPARISON_FAILED: "
                    f"{error}"
                )

        if comparison is not None:
            if (
                comparison["new_width"] != TARGET_WIDTH
                or comparison["new_height"] != TARGET_HEIGHT
            ):
                validation_issues.append(
                    "NEW_DIMENSION_MISMATCH"
                )

            if comparison["new_mode"] != "RGB":
                validation_issues.append(
                    "NEW_MODE_NOT_RGB"
                )

            if comparison["new_format"] != "PNG":
                validation_issues.append(
                    "NEW_FORMAT_NOT_PNG"
                )

        output_row = {
            "manifest_index": index,
            "split": row["split"],
            "class_label": row["class_label"],
            "numeric_label": row["numeric_label"],
            "prepared_relative_path": str(relative_path),
            "original_relative_path": row[
                "original_relative_path"
            ],
            "processing_action": row[
                "processing_action"
            ],
            "old_file_found": bool_text(old_found),
            "new_file_found": bool_text(new_found),
            "new_manifest_hash_match": bool_text(
                manifest_hash_match
            ),
            "old_width": "",
            "old_height": "",
            "old_mode": "",
            "old_format": "",
            "new_width": "",
            "new_height": "",
            "new_mode": "",
            "new_format": "",
            "old_file_sha256": old_sha256,
            "new_file_sha256": new_sha256,
            "file_sha256_identical": bool_text(
                bool(old_sha256)
                and old_sha256 == new_sha256
            ),
            "pixels_identical": "",
            "mean_absolute_error_0_255": "",
            "normalized_mean_absolute_error": "",
            "root_mean_squared_error_0_255": "",
            "changed_pixel_fraction": "",
            "phash_hamming_distance": "",
            "dhash_hamming_distance": "",
            "similarity_category": "NOT_COMPARED",
            "validation_status": (
                "PASS"
                if not validation_issues
                else "FAIL"
            ),
            "validation_issue": " | ".join(
                validation_issues
            ),
            "_old_path": old_path,
            "_new_path": new_path,
        }

        if comparison is not None:
            output_row.update(
                {
                    "old_width": comparison["old_width"],
                    "old_height": comparison["old_height"],
                    "old_mode": comparison["old_mode"],
                    "old_format": comparison["old_format"],
                    "new_width": comparison["new_width"],
                    "new_height": comparison["new_height"],
                    "new_mode": comparison["new_mode"],
                    "new_format": comparison["new_format"],
                    "pixels_identical": bool_text(
                        comparison["pixels_identical"]
                    ),
                    "mean_absolute_error_0_255": (
                        optional_float(
                            comparison["mean_absolute_error"],
                            8,
                        )
                    ),
                    "normalized_mean_absolute_error": (
                        optional_float(
                            comparison["normalized_mae"],
                            10,
                        )
                    ),
                    "root_mean_squared_error_0_255": (
                        optional_float(
                            comparison["rmse"],
                            8,
                        )
                    ),
                    "changed_pixel_fraction": (
                        optional_float(
                            comparison[
                                "changed_pixel_fraction"
                            ],
                            10,
                        )
                    ),
                    "phash_hamming_distance": comparison[
                        "phash_distance"
                    ],
                    "dhash_hamming_distance": comparison[
                        "dhash_distance"
                    ],
                    "similarity_category": comparison[
                        "similarity_category"
                    ],
                }
            )

        comparison_rows.append(output_row)

        if index % 250 == 0 or index == len(rows):
            print(
                f"Audited and compared {index}/{len(rows)}"
            )

    return comparison_rows


# =============================================================================
# DUPLICATE AUDIT
# =============================================================================

def audit_exact_duplicates(comparison_rows):
    """Find identical new PNG hashes and identify leakage."""

    hash_groups = defaultdict(list)

    for row in comparison_rows:
        if row["new_file_sha256"]:
            hash_groups[row["new_file_sha256"]].append(row)

    duplicate_groups = [
        group
        for group in hash_groups.values()
        if len(group) > 1
    ]

    duplicate_groups.sort(
        key=lambda group: (
            -len(group),
            group[0]["new_file_sha256"],
        )
    )

    duplicate_rows = []
    cross_split_groups = 0
    cross_class_groups = 0

    for group_number, group in enumerate(
        duplicate_groups,
        start=1,
    ):
        splits = {row["split"] for row in group}
        classes = {row["class_label"] for row in group}
        cross_split = len(splits) > 1
        cross_class = len(classes) > 1

        if cross_split:
            cross_split_groups += 1

        if cross_class:
            cross_class_groups += 1

        for row in group:
            duplicate_rows.append(
                {
                    "duplicate_group": (
                        f"DUP-{group_number:04d}"
                    ),
                    "output_sha256": row[
                        "new_file_sha256"
                    ],
                    "group_size": len(group),
                    "cross_split": bool_text(cross_split),
                    "cross_class": bool_text(cross_class),
                    "split": row["split"],
                    "class_label": row["class_label"],
                    "numeric_label": row["numeric_label"],
                    "prepared_relative_path": row[
                        "prepared_relative_path"
                    ],
                }
            )

    return {
        "group_count": len(duplicate_groups),
        "cross_split_group_count": cross_split_groups,
        "cross_class_group_count": cross_class_groups,
        "rows": duplicate_rows,
    }


# =============================================================================
# VISUAL REVIEW SHEETS
# =============================================================================

def load_font(size, bold=False):
    """Load a readable font on Windows or Linux."""

    candidates = []

    if bold:
        candidates.extend(
            [
                Path(r"C:\Windows\Fonts\arialbd.ttf"),
                Path(
                    "/usr/share/fonts/truetype/dejavu/"
                    "DejaVuSans-Bold.ttf"
                ),
            ]
        )
    else:
        candidates.extend(
            [
                Path(r"C:\Windows\Fonts\arial.ttf"),
                Path(
                    "/usr/share/fonts/truetype/dejavu/"
                    "DejaVuSans.ttf"
                ),
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(
                str(candidate),
                size=size,
            )

    return ImageFont.load_default()


def truncate_text(text, maximum_length):
    """Shorten text for a review card."""

    text = str(text)

    if len(text) <= maximum_length:
        return text

    return text[: maximum_length - 3] + "..."


def create_review_sheet(class_label, comparison_rows):
    """Create an old-vs-new sheet for the most changed images."""

    candidates = [
        row
        for row in comparison_rows
        if (
            row["class_label"] == class_label
            and row["similarity_category"] != "NOT_COMPARED"
        )
    ]

    candidates.sort(
        key=lambda row: (
            float(
                row["normalized_mean_absolute_error"]
            ),
            int(row["phash_hamming_distance"]),
            int(row["dhash_hamming_distance"]),
        ),
        reverse=True,
    )

    selected_rows = candidates[
        :REVIEW_SAMPLES_PER_CLASS
    ]

    if not selected_rows:
        return None

    canvas_width = 1600
    header_height = 80
    card_width = 790
    card_height = 290
    card_gap = 10
    columns = 2
    rows_needed = math.ceil(
        len(selected_rows) / columns
    )
    canvas_height = (
        header_height
        + rows_needed * (card_height + card_gap)
        + 10
    )

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        (239, 242, 246),
    )

    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24, bold=True)
    subtitle_font = load_font(15)
    label_font = load_font(13, bold=True)
    text_font = load_font(13)

    draw.rectangle(
        (0, 0, canvas_width, header_height),
        fill=(31, 42, 57),
    )

    draw.text(
        (16, 12),
        "Old vs Uniform Top-Viewport Comparison",
        font=title_font,
        fill=(250, 250, 250),
    )

    draw.text(
        (16, 48),
        (
            f"Class: {class_label} | "
            "Most changed samples | Read-only audit"
        ),
        font=subtitle_font,
        fill=(230, 235, 242),
    )

    for sample_index, row in enumerate(
        selected_rows,
        start=1,
    ):
        zero_based_index = sample_index - 1
        column = zero_based_index % columns
        grid_row = zero_based_index // columns

        left = 5 + column * (card_width + card_gap)
        top = (
            header_height
            + 5
            + grid_row * (card_height + card_gap)
        )
        right = left + card_width
        bottom = top + card_height

        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=8,
            fill=(255, 255, 255),
            outline=(135, 135, 135),
            width=2,
        )

        old_image = Image.open(
            row["_old_path"]
        ).convert("RGB")

        new_image = Image.open(
            row["_new_path"]
        ).convert("RGB")

        old_image.thumbnail(
            (380, 160),
            resample=LANCZOS,
        )

        new_image.thumbnail(
            (380, 160),
            resample=LANCZOS,
        )

        old_x = left + 8
        new_x = left + 402
        image_y = top + 27

        draw.text(
            (old_x, top + 7),
            "OLD PREPARED",
            font=label_font,
            fill=(31, 78, 121),
        )

        draw.text(
            (new_x, top + 7),
            "NEW TOP VIEWPORT",
            font=label_font,
            fill=(20, 125, 65),
        )

        draw.rectangle(
            (
                old_x,
                image_y,
                old_x + 380,
                image_y + 160,
            ),
            outline=(185, 185, 185),
            width=1,
        )

        draw.rectangle(
            (
                new_x,
                image_y,
                new_x + 380,
                image_y + 160,
            ),
            outline=(185, 185, 185),
            width=1,
        )

        old_paste_x = old_x + (380 - old_image.width) // 2
        old_paste_y = image_y + (160 - old_image.height) // 2
        new_paste_x = new_x + (380 - new_image.width) // 2
        new_paste_y = image_y + (160 - new_image.height) // 2

        canvas.paste(
            old_image,
            (old_paste_x, old_paste_y),
        )

        canvas.paste(
            new_image,
            (new_paste_x, new_paste_y),
        )

        metadata_y = image_y + 168

        draw.text(
            (left + 8, metadata_y),
            (
                f"Sample {sample_index:02d} | "
                f"{row['split']} | "
                f"{row['similarity_category']}"
            ),
            font=label_font,
            fill=(30, 30, 30),
        )

        draw.text(
            (left + 8, metadata_y + 21),
            (
                f"Normalized MAE: "
                f"{float(row['normalized_mean_absolute_error']):.6f} | "
                f"pHash: {row['phash_hamming_distance']} | "
                f"dHash: {row['dhash_hamming_distance']}"
            ),
            font=text_font,
            fill=(45, 45, 45),
        )

        draw.text(
            (left + 8, metadata_y + 42),
            (
                "File: "
                + truncate_text(
                    row["prepared_relative_path"],
                    94,
                )
            ),
            font=text_font,
            fill=(45, 45, 45),
        )

    output_path = (
        OUTPUT_DIR
        / f"{class_label}_old_vs_new_most_changed.png"
    )

    canvas.save(
        output_path,
        format="PNG",
        optimize=False,
        compress_level=6,
    )

    return output_path


# =============================================================================
# OUTPUT REPORTS
# =============================================================================

def write_csv(output_path, fieldnames, rows):
    """Write a CSV report without private working fields."""

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


def mean_or_zero(values):
    """Return a safe arithmetic mean."""

    if not values:
        return 0.0

    return float(np.mean(values))


def summarise_class(class_label, comparison_rows):
    """Calculate comparison metrics for one class."""

    rows = [
        row
        for row in comparison_rows
        if (
            row["class_label"] == class_label
            and row["similarity_category"] != "NOT_COMPARED"
        )
    ]

    categories = Counter(
        row["similarity_category"]
        for row in rows
    )

    normalized_mae_values = [
        float(row["normalized_mean_absolute_error"])
        for row in rows
    ]

    phash_values = [
        int(row["phash_hamming_distance"])
        for row in rows
    ]

    dhash_values = [
        int(row["dhash_hamming_distance"])
        for row in rows
    ]

    return {
        "compared_images": len(rows),
        "similarity_categories": dict(categories),
        "mean_normalized_mae": mean_or_zero(
            normalized_mae_values
        ),
        "maximum_normalized_mae": (
            max(normalized_mae_values)
            if normalized_mae_values
            else 0.0
        ),
        "mean_phash_distance": mean_or_zero(
            phash_values
        ),
        "maximum_phash_distance": (
            max(phash_values)
            if phash_values
            else 0
        ),
        "mean_dhash_distance": mean_or_zero(
            dhash_values
        ),
        "maximum_dhash_distance": (
            max(dhash_values)
            if dhash_values
            else 0
        ),
    }


def determine_audit_result(
    failed_validation_count,
    duplicate_audit,
    category_counts,
    compared_count,
):
    """Return a conservative automated audit conclusion."""

    if failed_validation_count > 0:
        return "AUDIT_FAILED_FILE_FORMAT_OR_HASH_ISSUES"

    if (
        duplicate_audit["cross_split_group_count"] > 0
        or duplicate_audit["cross_class_group_count"] > 0
    ):
        return "AUDIT_FAILED_DUPLICATE_LEAKAGE_FOUND"

    if compared_count == 0:
        return "AUDIT_INCOMPLETE_NO_IMAGE_PAIRS_COMPARED"

    pixel_or_near_count = (
        category_counts["PIXEL_IDENTICAL"]
        + category_counts["NEAR_IDENTICAL"]
    )

    similar_count = (
        pixel_or_near_count
        + category_counts["HIGHLY_SIMILAR"]
    )

    pixel_or_near_fraction = (
        pixel_or_near_count / compared_count
    )

    similar_fraction = similar_count / compared_count

    if pixel_or_near_fraction >= 0.95:
        return (
            "NEW_DATASET_EFFECTIVELY_EQUIVALENT_TO_OLD_"
            "DO_NOT_CLAIM_AS_INDEPENDENT_CORRECTION"
        )

    if similar_fraction >= 0.95:
        return (
            "NEW_DATASET_VISUALLY_VERY_SIMILAR_TO_OLD_"
            "REVIEW_BEFORE_RETRAINING"
        )

    return "NEW_DATASET_MEANINGFULLY_DIFFERS_VISUAL_REVIEW_REQUIRED"


def save_reports(comparison_rows, duplicate_audit, review_paths):
    """Save CSV, JSON and summary reports."""

    comparison_csv_path = (
        OUTPUT_DIR
        / "old_vs_top_viewport_comparison.csv"
    )

    duplicate_csv_path = (
        OUTPUT_DIR
        / "new_dataset_exact_duplicate_groups.csv"
    )

    issue_csv_path = (
        OUTPUT_DIR
        / "validation_issues.csv"
    )

    write_csv(
        comparison_csv_path,
        COMPARISON_FIELDNAMES,
        comparison_rows,
    )

    write_csv(
        duplicate_csv_path,
        DUPLICATE_FIELDNAMES,
        duplicate_audit["rows"],
    )

    issue_rows = [
        row
        for row in comparison_rows
        if row["validation_status"] == "FAIL"
    ]

    write_csv(
        issue_csv_path,
        COMPARISON_FIELDNAMES,
        issue_rows,
    )

    category_counts = Counter(
        row["similarity_category"]
        for row in comparison_rows
        if row["similarity_category"] != "NOT_COMPARED"
    )

    compared_rows = [
        row
        for row in comparison_rows
        if row["similarity_category"] != "NOT_COMPARED"
    ]

    file_sha_identical_count = sum(
        row["file_sha256_identical"] == "YES"
        for row in comparison_rows
    )

    pixel_identical_count = sum(
        row["pixels_identical"] == "YES"
        for row in comparison_rows
    )

    manifest_hash_match_count = sum(
        row["new_manifest_hash_match"] == "YES"
        for row in comparison_rows
    )

    audit_result = determine_audit_result(
        failed_validation_count=len(issue_rows),
        duplicate_audit=duplicate_audit,
        category_counts=category_counts,
        compared_count=len(compared_rows),
    )

    class_metrics = {
        class_label: summarise_class(
            class_label,
            comparison_rows,
        )
        for class_label in EXPECTED_CLASS_COUNTS
    }

    metrics = {
        "audit_result": audit_result,
        "old_dataset_path": str(OLD_DATASET_DIR),
        "new_dataset_path": str(NEW_DATASET_DIR),
        "new_manifest_path": str(NEW_MANIFEST_PATH),
        "new_manifest_sha256": calculate_sha256(
            NEW_MANIFEST_PATH
        ),
        "manifest_images": len(comparison_rows),
        "successfully_compared": len(compared_rows),
        "validation_passed": (
            len(comparison_rows) - len(issue_rows)
        ),
        "validation_failed": len(issue_rows),
        "new_manifest_hash_matches": (
            manifest_hash_match_count
        ),
        "file_sha256_identical": (
            file_sha_identical_count
        ),
        "pixel_identical": pixel_identical_count,
        "similarity_categories": dict(category_counts),
        "class_metrics": class_metrics,
        "duplicate_hash_groups": duplicate_audit[
            "group_count"
        ],
        "cross_split_duplicate_groups": duplicate_audit[
            "cross_split_group_count"
        ],
        "cross_class_duplicate_groups": duplicate_audit[
            "cross_class_group_count"
        ],
        "review_sheets": [
            str(path)
            for path in review_paths
            if path is not None
        ],
        "test_split_status": "DIAGNOSTIC_ONLY",
    }

    with open(
        OUTPUT_DIR / "comparison_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    compared_count = len(compared_rows)

    def fraction(count):
        if compared_count == 0:
            return 0.0

        return count / compared_count

    similar_count = (
        category_counts["PIXEL_IDENTICAL"]
        + category_counts["NEAR_IDENTICAL"]
        + category_counts["HIGHLY_SIMILAR"]
    )

    summary_lines = [
        "=" * 80,
        "TOP-VIEWPORT DATASET AUDIT AND COMPARISON COMPLETE",
        "=" * 80,
        "",
        f"Old dataset          : {OLD_DATASET_DIR}",
        f"New dataset          : {NEW_DATASET_DIR}",
        f"New manifest         : {NEW_MANIFEST_PATH}",
        f"Manifest SHA256      : {metrics['new_manifest_sha256']}",
        "",
        "FILE AND HASH VALIDATION",
        "-" * 80,
        f"Manifest images      : {len(comparison_rows)}",
        f"Successfully compared: {compared_count}",
        f"Validation passed    : {len(comparison_rows) - len(issue_rows)}",
        f"Validation failed    : {len(issue_rows)}",
        f"Manifest hash matches: {manifest_hash_match_count}",
        "",
        "OLD VS NEW COMPARISON",
        "-" * 80,
        f"File SHA identical   : {file_sha_identical_count}",
        f"Pixel identical      : {pixel_identical_count}",
        (
            "PIXEL_IDENTICAL       : "
            f"{category_counts['PIXEL_IDENTICAL']} "
            f"({fraction(category_counts['PIXEL_IDENTICAL']):.6f})"
        ),
        (
            "NEAR_IDENTICAL        : "
            f"{category_counts['NEAR_IDENTICAL']} "
            f"({fraction(category_counts['NEAR_IDENTICAL']):.6f})"
        ),
        (
            "HIGHLY_SIMILAR        : "
            f"{category_counts['HIGHLY_SIMILAR']} "
            f"({fraction(category_counts['HIGHLY_SIMILAR']):.6f})"
        ),
        (
            "MODERATELY_SIMILAR    : "
            f"{category_counts['MODERATELY_SIMILAR']} "
            f"({fraction(category_counts['MODERATELY_SIMILAR']):.6f})"
        ),
        (
            "VISUALLY_DIFFERENT    : "
            f"{category_counts['VISUALLY_DIFFERENT']} "
            f"({fraction(category_counts['VISUALLY_DIFFERENT']):.6f})"
        ),
        (
            "Pixel/near/high total : "
            f"{similar_count} "
            f"({fraction(similar_count):.6f})"
        ),
        "",
        "EXACT DUPLICATE AUDIT - NEW DATASET",
        "-" * 80,
        f"Duplicate groups     : {duplicate_audit['group_count']}",
        f"Cross-split groups   : {duplicate_audit['cross_split_group_count']}",
        f"Cross-class groups   : {duplicate_audit['cross_class_group_count']}",
        "",
        "CLASS COMPARISON",
        "-" * 80,
    ]

    for class_label, class_result in class_metrics.items():
        summary_lines.extend(
            [
                f"{class_label}:",
                (
                    "  Compared           : "
                    f"{class_result['compared_images']}"
                ),
                (
                    "  Mean normalized MAE: "
                    f"{class_result['mean_normalized_mae']:.8f}"
                ),
                (
                    "  Maximum norm. MAE  : "
                    f"{class_result['maximum_normalized_mae']:.8f}"
                ),
                (
                    "  Mean pHash distance: "
                    f"{class_result['mean_phash_distance']:.6f}"
                ),
                (
                    "  Mean dHash distance: "
                    f"{class_result['mean_dhash_distance']:.6f}"
                ),
                "",
            ]
        )

    summary_lines.extend(
        [
            "AUDIT RESULT",
            "-" * 80,
            audit_result,
            "",
            "INTERPRETATION",
            "-" * 80,
            "Pixel equality tests whether displayed RGB values are identical.",
            "File SHA equality is stricter and also depends on PNG encoding.",
            "pHash and dHash estimate perceptual visual similarity.",
            "A highly similar result means changing the model architecture alone",
            "does not prove that the preprocessing/source confound was corrected.",
            "Inspect both review sheets before deciding whether to retrain.",
            "",
            "IMPORTANT",
            "-" * 80,
            "No screenshot or manifest was modified.",
            "No model training was performed.",
            "The previous TEST split remains diagnostic only.",
            "A separate external dataset is required for final evaluation.",
            "=" * 80,
        ]
    )

    summary_text = "\n".join(summary_lines)

    with open(
        OUTPUT_DIR / "comparison_summary.txt",
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
    print("AUDIT AND COMPARE UNIFORM TOP-VIEWPORT DATASET")
    print("=" * 80)
    print(f"Old dataset         : {OLD_DATASET_DIR}")
    print(f"New dataset         : {NEW_DATASET_DIR}")
    print(f"New manifest        : {NEW_MANIFEST_PATH}")
    print(f"Output directory    : {OUTPUT_DIR}")
    print()

    if not OLD_DATASET_DIR.is_dir():
        raise FileNotFoundError(
            "Old prepared dataset was not found:\n"
            f"{OLD_DATASET_DIR}"
        )

    if not NEW_DATASET_DIR.is_dir():
        raise FileNotFoundError(
            "New top-viewport dataset was not found:\n"
            f"{NEW_DATASET_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading and validating new manifest...")
    rows = read_and_validate_manifest()
    print(f"Manifest records    : {len(rows)}")
    print("Manifest labels, counts and actions are valid.")
    print()

    print("Auditing new files and comparing old vs new...")
    comparison_rows = audit_and_compare(rows)
    print()

    print("Checking exact duplicates in the new dataset...")
    duplicate_audit = audit_exact_duplicates(
        comparison_rows
    )
    print(
        f"Duplicate groups     : "
        f"{duplicate_audit['group_count']}"
    )
    print(
        f"Cross-split groups   : "
        f"{duplicate_audit['cross_split_group_count']}"
    )
    print(
        f"Cross-class groups   : "
        f"{duplicate_audit['cross_class_group_count']}"
    )
    print()

    print("Creating old-vs-new visual review sheets...")
    review_paths = []

    for class_label in [
        "NOT_MALICIOUS",
        "MALICIOUS",
    ]:
        review_path = create_review_sheet(
            class_label,
            comparison_rows,
        )

        review_paths.append(review_path)

        if review_path is not None:
            print(f"Created: {review_path.name}")

    print()
    print("Saving audit and comparison reports...")

    summary_text = save_reports(
        comparison_rows,
        duplicate_audit,
        review_paths,
    )

    print()
    print(summary_text)
    print()
    print(f"Reports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print()
        print("=" * 80)
        print("TOP-VIEWPORT COMPARISON AUDIT FAILED")
        print("=" * 80)
        print(error)
        print()
        print("No screenshot or manifest was modified.")
        print("No model training was performed.")

        sys.exit(1)
