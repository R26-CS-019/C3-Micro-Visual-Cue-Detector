"""
16_create_efficientnet_validation_error_review.py

Create read-only visual review sheets and manual-review CSV files for the
validation errors produced by Step 15.

Scientific rules:
- Uses only the saved VALIDATION predictions from Step 15.
- Does not load or evaluate the diagnostic TEST split.
- Does not train or modify the model.
- Keeps the selected validation threshold fixed at 0.4660.
- Does not modify original or prepared screenshots.
"""

import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(
    r"C:\Users\oshan\Desktop\model 1 login pp2"
)

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "prepared_login_uniform_top_viewport_384x160"
)

MANIFEST_PATH = (
    PROJECT_DIR
    / "reports"
    / "13_uniform_top_viewport_dataset"
    / "complete_uniform_top_viewport_manifest.csv"
)

TRAINING_REPORT_DIR = (
    PROJECT_DIR
    / "reports"
    / "15_efficientnetv2b0_top_viewport_training"
)

PREDICTIONS_PATH = (
    TRAINING_REPORT_DIR
    / "validation_predictions.csv"
)

METRICS_PATH = (
    TRAINING_REPORT_DIR
    / "validation_metrics.json"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "model1_efficientnetv2b0_top_viewport"
    / "model1_efficientnetv2b0_selected_best.keras"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "reports"
    / "16_efficientnet_validation_error_review"
)

SELECTED_THRESHOLD = 0.4660

EXPECTED_VALIDATION_IMAGES = 426
EXPECTED_NOT_MALICIOUS_IMAGES = 213
EXPECTED_MALICIOUS_IMAGES = 213
EXPECTED_FALSE_POSITIVES = 21
EXPECTED_FALSE_NEGATIVES = 49

EXPECTED_WIDTH = 384
EXPECTED_HEIGHT = 160
EXPECTED_PROCESSING_ACTION = (
    "UNIFORM_WIDTH_SCALE_TOP_VIEWPORT"
)

IMAGES_PER_SHEET = 16
GRID_COLUMNS = 4

CLASS_TO_NUMERIC = {
    "NOT_MALICIOUS": 0,
    "MALICIOUS": 1,
}

NUMERIC_TO_CLASS = {
    0: "NOT_MALICIOUS",
    1: "MALICIOUS",
}

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

REQUIRED_PREDICTION_COLUMNS = {
    "validation_index",
    "prepared_relative_path",
    "original_relative_path",
    "actual_numeric_label",
    "actual_class",
    "malicious_probability",
    "selected_validation_threshold",
    "predicted_numeric_label",
    "predicted_class",
    "correct_prediction",
    "result_type",
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def calculate_sha256(file_path):
    """Calculate the SHA256 digest of a file without modifying it."""

    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def normalise_relative_path(value, context):
    """Return a safe platform-specific relative path."""

    cleaned = str(value).strip().replace("\\", "/")

    if not cleaned:
        raise RuntimeError(
            f"Empty prepared path in {context}."
        )

    parts = [part for part in cleaned.split("/") if part]

    if (
        not parts
        or any(part in {".", ".."} for part in parts)
        or ":" in parts[0]
    ):
        raise RuntimeError(
            f"Unsafe prepared path in {context}: {value!r}"
        )

    relative_path = Path(*parts)

    if relative_path.is_absolute():
        raise RuntimeError(
            f"Expected a relative path in {context}: {value!r}"
        )

    return relative_path


def load_json(file_path):
    """Load one JSON object."""

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Expected a JSON object in:\n{file_path}"
        )

    return data


def write_json(file_path, data):
    """Write formatted JSON."""

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def ensure_required_files():
    """Check every required input before creating output reports."""

    required_files = [
        MANIFEST_PATH,
        PREDICTIONS_PATH,
        METRICS_PATH,
        MODEL_PATH,
    ]

    missing_files = [
        path for path in required_files if not path.is_file()
    ]

    if missing_files:
        formatted = "\n".join(
            f"  - {path}" for path in missing_files
        )

        raise FileNotFoundError(
            "Required files were not found:\n"
            f"{formatted}"
        )

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            "Prepared dataset directory was not found:\n"
            f"{DATASET_DIR}"
        )


def expected_result_type(actual, predicted):
    """Return the binary confusion-matrix category."""

    if actual == 0 and predicted == 0:
        return "TRUE_NEGATIVE"

    if actual == 0 and predicted == 1:
        return "FALSE_POSITIVE"

    if actual == 1 and predicted == 0:
        return "FALSE_NEGATIVE"

    return "TRUE_POSITIVE"


# =============================================================================
# MANIFEST AND METRIC VALIDATION
# =============================================================================

def read_validation_manifest():
    """Read and validate only the VALIDATION rows in the manifest."""

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])

        missing_columns = (
            REQUIRED_MANIFEST_COLUMNS - fieldnames
        )

        if missing_columns:
            raise RuntimeError(
                "Manifest is missing required columns:\n"
                f"{sorted(missing_columns)}"
            )

        manifest_rows = list(reader)

    validation_records = {}
    class_counts = {
        "NOT_MALICIOUS": 0,
        "MALICIOUS": 0,
    }

    for row_number, row in enumerate(
        manifest_rows,
        start=2,
    ):
        split = row["split"].strip().upper()

        if split != "VALIDATION":
            continue

        class_label = row["class_label"].strip().upper()

        if class_label not in CLASS_TO_NUMERIC:
            raise RuntimeError(
                f"Invalid class at manifest row {row_number}: "
                f"{class_label!r}"
            )

        try:
            numeric_label = int(row["numeric_label"])
            output_width = int(row["output_width"])
            output_height = int(row["output_height"])
        except ValueError as error:
            raise RuntimeError(
                f"Invalid numeric field at manifest row "
                f"{row_number}."
            ) from error

        if numeric_label != CLASS_TO_NUMERIC[class_label]:
            raise RuntimeError(
                "Manifest class/numeric-label mismatch at row "
                f"{row_number}."
            )

        action = row["processing_action"].strip().upper()

        if action != EXPECTED_PROCESSING_ACTION:
            raise RuntimeError(
                "Unexpected processing action at manifest row "
                f"{row_number}: {action!r}"
            )

        if (
            output_width != EXPECTED_WIDTH
            or output_height != EXPECTED_HEIGHT
        ):
            raise RuntimeError(
                "Unexpected output dimensions at manifest row "
                f"{row_number}."
            )

        if row["output_colour_mode"].strip().upper() != "RGB":
            raise RuntimeError(
                "Unexpected colour mode at manifest row "
                f"{row_number}."
            )

        if row["output_format"].strip().upper() != "PNG":
            raise RuntimeError(
                "Unexpected image format at manifest row "
                f"{row_number}."
            )

        relative_path = normalise_relative_path(
            row["prepared_relative_path"],
            f"manifest row {row_number}",
        )

        path_key = str(relative_path).casefold()

        if path_key in validation_records:
            raise RuntimeError(
                "Duplicate VALIDATION prepared path in manifest:\n"
                f"{relative_path}"
            )

        output_sha256 = row["output_sha256"].strip().lower()

        if (
            len(output_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in output_sha256
            )
        ):
            raise RuntimeError(
                "Invalid output SHA256 at manifest row "
                f"{row_number}."
            )

        validation_records[path_key] = {
            "prepared_relative_path": relative_path,
            "original_relative_path": row[
                "original_relative_path"
            ].strip(),
            "numeric_label": numeric_label,
            "class_label": class_label,
            "output_sha256": output_sha256,
            "image_path": DATASET_DIR / relative_path,
        }

        class_counts[class_label] += 1

    if len(validation_records) != EXPECTED_VALIDATION_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_VALIDATION_IMAGES} VALIDATION "
            f"manifest rows, but found "
            f"{len(validation_records)}."
        )

    expected_class_counts = {
        "NOT_MALICIOUS": EXPECTED_NOT_MALICIOUS_IMAGES,
        "MALICIOUS": EXPECTED_MALICIOUS_IMAGES,
    }

    if class_counts != expected_class_counts:
        raise RuntimeError(
            "Unexpected VALIDATION class counts.\n"
            f"Expected: {expected_class_counts}\n"
            f"Actual  : {class_counts}"
        )

    return validation_records


def validate_saved_metrics():
    """Validate the selected threshold and expected error counts."""

    metrics = load_json(METRICS_PATH)

    try:
        threshold = float(metrics["threshold"])
        validation_images = int(metrics["validation_images"])
        false_positives = int(metrics["false_positives"])
        false_negatives = int(metrics["false_negatives"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "validation_metrics.json does not contain the "
            "expected Step 15 metric fields."
        ) from error

    if not math.isclose(
        threshold,
        SELECTED_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "The saved validation threshold does not match the "
            "locked review threshold.\n"
            f"Saved   : {threshold:.10f}\n"
            f"Expected: {SELECTED_THRESHOLD:.10f}"
        )

    if validation_images != EXPECTED_VALIDATION_IMAGES:
        raise RuntimeError(
            "Unexpected validation image count in saved metrics: "
            f"{validation_images}."
        )

    if false_positives != EXPECTED_FALSE_POSITIVES:
        raise RuntimeError(
            "Unexpected saved false-positive count: "
            f"{false_positives}."
        )

    if false_negatives != EXPECTED_FALSE_NEGATIVES:
        raise RuntimeError(
            "Unexpected saved false-negative count: "
            f"{false_negatives}."
        )

    return metrics


# =============================================================================
# PREDICTION AND IMAGE VALIDATION
# =============================================================================

def read_and_validate_predictions(validation_manifest):
    """Cross-check saved predictions against the VALIDATION manifest."""

    with open(
        PREDICTIONS_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])

        missing_columns = (
            REQUIRED_PREDICTION_COLUMNS - fieldnames
        )

        if missing_columns:
            raise RuntimeError(
                "Prediction CSV is missing required columns:\n"
                f"{sorted(missing_columns)}"
            )

        prediction_rows = list(reader)

    if len(prediction_rows) != EXPECTED_VALIDATION_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_VALIDATION_IMAGES} saved "
            f"validation predictions, but found "
            f"{len(prediction_rows)}."
        )

    validated_rows = []
    seen_indices = set()
    seen_paths = set()

    for csv_row_number, row in enumerate(
        prediction_rows,
        start=2,
    ):
        try:
            validation_index = int(row["validation_index"])
            actual_label = int(row["actual_numeric_label"])
            predicted_label = int(
                row["predicted_numeric_label"]
            )
            probability = float(
                row["malicious_probability"]
            )
            row_threshold = float(
                row["selected_validation_threshold"]
            )
        except ValueError as error:
            raise RuntimeError(
                "Invalid numeric value in prediction CSV row "
                f"{csv_row_number}."
            ) from error

        if validation_index in seen_indices:
            raise RuntimeError(
                "Duplicate validation index in prediction CSV: "
                f"{validation_index}"
            )

        seen_indices.add(validation_index)

        relative_path = normalise_relative_path(
            row["prepared_relative_path"],
            f"prediction CSV row {csv_row_number}",
        )

        path_key = str(relative_path).casefold()

        if path_key in seen_paths:
            raise RuntimeError(
                "Duplicate prepared path in prediction CSV:\n"
                f"{relative_path}"
            )

        seen_paths.add(path_key)

        if path_key not in validation_manifest:
            raise RuntimeError(
                "Prediction path was not found in the "
                "VALIDATION manifest:\n"
                f"{relative_path}"
            )

        manifest_record = validation_manifest[path_key]

        if actual_label not in {0, 1}:
            raise RuntimeError(
                "Invalid actual label in prediction CSV row "
                f"{csv_row_number}."
            )

        if predicted_label not in {0, 1}:
            raise RuntimeError(
                "Invalid predicted label in prediction CSV row "
                f"{csv_row_number}."
            )

        if actual_label != manifest_record["numeric_label"]:
            raise RuntimeError(
                "Prediction/manifest label mismatch for:\n"
                f"{relative_path}"
            )

        actual_class = row["actual_class"].strip().upper()
        predicted_class = row["predicted_class"].strip().upper()

        if actual_class != NUMERIC_TO_CLASS[actual_label]:
            raise RuntimeError(
                "Actual class text does not match its numeric "
                f"label at prediction CSV row {csv_row_number}."
            )

        if predicted_class != NUMERIC_TO_CLASS[predicted_label]:
            raise RuntimeError(
                "Predicted class text does not match its numeric "
                f"label at prediction CSV row {csv_row_number}."
            )

        if not 0.0 <= probability <= 1.0:
            raise RuntimeError(
                "Probability outside [0, 1] at prediction CSV "
                f"row {csv_row_number}."
            )

        if not math.isclose(
            row_threshold,
            SELECTED_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "Prediction CSV threshold mismatch at row "
                f"{csv_row_number}."
            )

        expected_prediction = int(
            probability >= SELECTED_THRESHOLD
        )

        if predicted_label != expected_prediction:
            raise RuntimeError(
                "Saved prediction does not match probability and "
                f"threshold at CSV row {csv_row_number}."
            )

        result_type = expected_result_type(
            actual_label,
            predicted_label,
        )

        if row["result_type"].strip().upper() != result_type:
            raise RuntimeError(
                "Saved result type is inconsistent at prediction "
                f"CSV row {csv_row_number}."
            )

        expected_correct = (
            "YES" if actual_label == predicted_label else "NO"
        )

        if row["correct_prediction"].strip().upper() != expected_correct:
            raise RuntimeError(
                "Saved correctness flag is inconsistent at "
                f"prediction CSV row {csv_row_number}."
            )

        validated_rows.append(
            {
                "validation_index": validation_index,
                "prepared_relative_path": str(relative_path),
                "original_relative_path": manifest_record[
                    "original_relative_path"
                ],
                "output_sha256": manifest_record[
                    "output_sha256"
                ],
                "image_path": manifest_record["image_path"],
                "actual_numeric_label": actual_label,
                "actual_class": actual_class,
                "malicious_probability": probability,
                "decision_threshold": SELECTED_THRESHOLD,
                "predicted_numeric_label": predicted_label,
                "predicted_class": predicted_class,
                "result_type": result_type,
            }
        )

    if seen_paths != set(validation_manifest.keys()):
        raise RuntimeError(
            "Prediction paths do not exactly match the VALIDATION "
            "manifest paths."
        )

    return validated_rows


def validate_validation_images(prediction_rows):
    """Validate each prepared VALIDATION image and its manifest hash."""

    print("Validating VALIDATION image files and hashes...")

    for index, row in enumerate(prediction_rows, start=1):
        image_path = row["image_path"]

        if not image_path.is_file():
            raise FileNotFoundError(
                "Prepared VALIDATION image was not found:\n"
                f"{image_path}"
            )

        actual_sha256 = calculate_sha256(image_path)

        if actual_sha256 != row["output_sha256"]:
            raise RuntimeError(
                "Prepared VALIDATION image SHA256 mismatch:\n"
                f"{image_path}\n"
                f"Expected: {row['output_sha256']}\n"
                f"Actual  : {actual_sha256}"
            )

        try:
            with Image.open(image_path) as image:
                image.load()

                if image.size != (
                    EXPECTED_WIDTH,
                    EXPECTED_HEIGHT,
                ):
                    raise RuntimeError(
                        "Unexpected image dimensions for:\n"
                        f"{image_path}\n"
                        f"Found: {image.size}"
                    )

                if image.mode != "RGB":
                    raise RuntimeError(
                        "Unexpected image colour mode for:\n"
                        f"{image_path}\n"
                        f"Found: {image.mode}"
                    )

        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError(
                "Could not read prepared VALIDATION image:\n"
                f"{image_path}\n"
                f"{error}"
            ) from error

        if index % 100 == 0 or index == len(prediction_rows):
            print(
                f"Validated {index}/{len(prediction_rows)} images"
            )


# =============================================================================
# REVIEW SHEETS
# =============================================================================

def load_font(size, bold=False):
    """Load a readable Windows or cross-platform TrueType font."""

    if bold:
        candidates = [
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
            Path(r"C:\Windows\Fonts\segoeuib.ttf"),
            Path("DejaVuSans-Bold.ttf"),
        ]
    else:
        candidates = [
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path(r"C:\Windows\Fonts\segoeui.ttf"),
            Path("DejaVuSans.ttf"),
        ]

    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except (OSError, ValueError):
            continue

    return ImageFont.load_default()


def shorten_text(text, maximum_characters):
    """Shorten long card text while keeping the beginning and ending."""

    text = str(text)

    if len(text) <= maximum_characters:
        return text

    keep = maximum_characters - 3
    left = keep // 2
    right = keep - left

    return f"{text[:left]}...{text[-right:]}"


def fit_image_for_card(image_path, size):
    """Fit one screenshot inside a fixed RGB review area."""

    with Image.open(image_path) as image:
        screenshot = image.convert("RGB")

        resampling = getattr(
            Image,
            "Resampling",
            Image,
        ).LANCZOS

        return ImageOps.contain(
            screenshot,
            size,
            method=resampling,
        )


def create_review_sheets(error_rows, error_code):
    """Create severity-sorted 4x4 visual review sheets."""

    if error_code == "FP":
        heading = (
            "EfficientNetV2B0 VALIDATION FALSE POSITIVE Review"
        )
        subtitle_class = (
            "Actual NOT_MALICIOUS, predicted MALICIOUS"
        )
        accent_colour = (205, 39, 39)
        output_prefix = "FP_sheet"
    elif error_code == "FN":
        heading = (
            "EfficientNetV2B0 VALIDATION FALSE NEGATIVE Review"
        )
        subtitle_class = (
            "Actual MALICIOUS, predicted NOT_MALICIOUS"
        )
        accent_colour = (210, 120, 0)
        output_prefix = "FN_sheet"
    else:
        raise ValueError(f"Unknown error code: {error_code}")

    sheet_count = math.ceil(
        len(error_rows) / IMAGES_PER_SHEET
    )

    sheet_width = 1640
    outer_margin = 6
    horizontal_gap = 8
    vertical_gap = 10
    header_height = 82
    card_width = 400
    card_height = 276
    image_width = 384
    image_height = 160

    title_font = load_font(25, bold=True)
    subtitle_font = load_font(16)
    card_title_font = load_font(17, bold=True)
    card_text_font = load_font(13)

    output_paths = []

    for sheet_number in range(1, sheet_count + 1):
        start = (sheet_number - 1) * IMAGES_PER_SHEET
        end = start + IMAGES_PER_SHEET
        sheet_rows = error_rows[start:end]

        used_grid_rows = math.ceil(
            len(sheet_rows) / GRID_COLUMNS
        )

        sheet_height = (
            header_height
            + outer_margin
            + used_grid_rows * card_height
            + max(0, used_grid_rows - 1) * vertical_gap
            + outer_margin
        )

        canvas = Image.new(
            "RGB",
            (sheet_width, sheet_height),
            color=(238, 241, 245),
        )

        draw = ImageDraw.Draw(canvas)

        draw.rectangle(
            [0, 0, sheet_width, header_height],
            fill=(31, 42, 57),
        )

        draw.text(
            (18, 12),
            heading,
            fill=(248, 250, 252),
            font=title_font,
        )

        draw.text(
            (18, 47),
            (
                f"Sheet {sheet_number}/{sheet_count} | "
                f"{subtitle_class} | Threshold = "
                f"{SELECTED_THRESHOLD:.4f} | "
                "Sorted by error severity"
            ),
            fill=(235, 238, 242),
            font=subtitle_font,
        )

        for local_index, row in enumerate(sheet_rows):
            review_number = start + local_index + 1
            grid_row = local_index // GRID_COLUMNS
            grid_column = local_index % GRID_COLUMNS

            left = (
                outer_margin
                + grid_column * (card_width + horizontal_gap)
            )

            top = (
                header_height
                + outer_margin
                + grid_row * (card_height + vertical_gap)
            )

            right = left + card_width - 1
            bottom = top + card_height - 1

            draw.rounded_rectangle(
                [left, top, right, bottom],
                radius=8,
                fill=(255, 255, 255),
                outline=(130, 130, 130),
                width=2,
            )

            screenshot = fit_image_for_card(
                row["image_path"],
                (image_width, image_height),
            )

            image_left = left + 8 + (
                image_width - screenshot.width
            ) // 2
            image_top = top + 7 + (
                image_height - screenshot.height
            ) // 2

            canvas.paste(
                screenshot,
                (image_left, image_top),
            )

            text_top = top + 172

            draw.text(
                (left + 8, text_top),
                (
                    f"{error_code}-{review_number:03d} | "
                    f"Validation #{row['validation_index']}"
                ),
                fill=accent_colour,
                font=card_title_font,
            )

            probability = row["malicious_probability"]

            if error_code == "FP":
                error_margin = probability - SELECTED_THRESHOLD
            else:
                error_margin = SELECTED_THRESHOLD - probability

            detail_lines = [
                (
                    f"Malicious probability: {probability:.4f} | "
                    f"Error margin: {error_margin:.4f}"
                ),
                (
                    f"Actual: {row['actual_class']} | "
                    f"Predicted: {row['predicted_class']}"
                ),
                (
                    "File: "
                    + shorten_text(
                        Path(
                            row["prepared_relative_path"]
                        ).name,
                        52,
                    )
                ),
                (
                    "Original: "
                    + shorten_text(
                        Path(
                            row["original_relative_path"]
                        ).name,
                        48,
                    )
                ),
            ]

            for line_index, line in enumerate(detail_lines):
                draw.text(
                    (
                        left + 8,
                        text_top + 25 + line_index * 18,
                    ),
                    line,
                    fill=(25, 25, 25),
                    font=card_text_font,
                )

        output_path = (
            OUTPUT_DIR
            / f"{output_prefix}_{sheet_number:02d}.png"
        )

        canvas.save(
            output_path,
            format="PNG",
            optimize=True,
        )

        output_paths.append(output_path)

        print(
            f"Created: {output_path.name} "
            f"({len(sheet_rows)} images)"
        )

    return output_paths


# =============================================================================
# CSV AND TEXT REPORTS
# =============================================================================

def review_csv_row(row, review_id):
    """Create one manual-review row with blank reviewer fields."""

    probability = row["malicious_probability"]

    if row["result_type"] == "FALSE_POSITIVE":
        error_margin = probability - SELECTED_THRESHOLD
    else:
        error_margin = SELECTED_THRESHOLD - probability

    return {
        "review_id": review_id,
        "validation_index": row["validation_index"],
        "result_type": row["result_type"],
        "prepared_relative_path": row[
            "prepared_relative_path"
        ],
        "prepared_image_path": str(row["image_path"]),
        "original_relative_path": row[
            "original_relative_path"
        ],
        "output_sha256": row["output_sha256"],
        "actual_numeric_label": row[
            "actual_numeric_label"
        ],
        "actual_class": row["actual_class"],
        "malicious_probability": f"{probability:.10f}",
        "decision_threshold": f"{SELECTED_THRESHOLD:.4f}",
        "error_margin": f"{error_margin:.10f}",
        "predicted_numeric_label": row[
            "predicted_numeric_label"
        ],
        "predicted_class": row["predicted_class"],
        "manual_label_correct": "",
        "login_form_visibility": "",
        "login_form_size": "",
        "background_style": "",
        "brand_or_service": "",
        "text_readability": "",
        "possible_source_shortcut": "",
        "primary_visual_pattern": "",
        "possible_error_cause": "",
        "recommended_action": "",
        "reviewer_notes": "",
        "review_status": "NOT_REVIEWED",
    }


def save_manual_review_csv(output_path, rows, prefix):
    """Save a spreadsheet-ready manual-review template."""

    output_rows = [
        review_csv_row(
            row,
            f"{prefix}-{index:03d}",
        )
        for index, row in enumerate(rows, start=1)
    ]

    fieldnames = list(output_rows[0].keys())

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
        writer.writerows(output_rows)

    return output_rows


def save_all_errors_csv(false_positive_rows, false_negative_rows):
    """Save a combined machine-readable error list."""

    fieldnames = [
        "review_id",
        "validation_index",
        "result_type",
        "prepared_relative_path",
        "original_relative_path",
        "output_sha256",
        "actual_numeric_label",
        "actual_class",
        "malicious_probability",
        "decision_threshold",
        "error_margin",
        "predicted_numeric_label",
        "predicted_class",
    ]

    output_rows = []

    for prefix, rows in [
        ("FP", false_positive_rows),
        ("FN", false_negative_rows),
    ]:
        for index, row in enumerate(rows, start=1):
            manual_row = review_csv_row(
                row,
                f"{prefix}-{index:03d}",
            )

            output_rows.append(
                {
                    field: manual_row[field]
                    for field in fieldnames
                }
            )

    with open(
        OUTPUT_DIR / "all_validation_errors.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(output_rows)


def probability_statistics(rows):
    """Return simple probability statistics for one error group."""

    probabilities = [
        row["malicious_probability"] for row in rows
    ]

    return {
        "count": len(probabilities),
        "minimum": float(min(probabilities)),
        "maximum": float(max(probabilities)),
        "mean": float(statistics.fmean(probabilities)),
        "median": float(statistics.median(probabilities)),
    }


def save_review_instructions():
    """Save concise instructions for completing the manual CSVs."""

    instructions = f"""EFFICIENTNETV2B0 VALIDATION ERROR REVIEW INSTRUCTIONS
================================================================================

PURPOSE
--------------------------------------------------------------------------------
Review the {EXPECTED_FALSE_POSITIVES} false positives and
{EXPECTED_FALSE_NEGATIVES} false negatives produced on VALIDATION at the fixed
threshold {SELECTED_THRESHOLD:.4f}.

WORKFLOW
--------------------------------------------------------------------------------
1. Open the FP and FN review-sheet PNG files.
2. Open the matching manual-review CSV.
3. Complete the blank review columns for every row.
4. Do not change labels automatically. Record suspected label issues first.
5. Keep VALIDATION separate from TRAIN during future experiments.
6. Do not use the diagnostic TEST split for model or threshold improvement.

SUGGESTED VALUES
--------------------------------------------------------------------------------
manual_label_correct:
  YES, NO, or UNCERTAIN

login_form_visibility:
  CLEAR, PARTIAL, TINY, or NOT_VISIBLE

login_form_size:
  SMALL, MEDIUM, or LARGE

background_style:
  WHITE_MINIMAL, DARK, PHOTO, ILLUSTRATION, BUSY, or OTHER

text_readability:
  CLEAR, PARTIAL, or UNREADABLE

possible_source_shortcut:
  YES, NO, or UNCERTAIN

primary_visual_pattern examples:
  SMALL_CENTERED_FORM
  LARGE_LOGIN_PANEL
  POPUP_OR_MODAL
  BRANDED_BACKGROUND
  WHITE_MINIMAL_PAGE
  BUSY_PAGE
  UNUSUAL_VIEWPORT_COMPOSITION
  FORM_NOT_CLEARLY_VISIBLE
  OTHER

recommended_action examples:
  KEEP_LABEL
  REVIEW_LABEL
  ADD_SIMILAR_TRAIN_EXAMPLES
  IMPROVE_AUGMENTATION
  EXCLUDE_INVALID_IMAGE
  INVESTIGATE_SOURCE_BIAS

IMPORTANT
--------------------------------------------------------------------------------
This step performs error analysis only. It does not train the model, change the
threshold, load TEST data, or modify screenshots.
"""

    with open(
        OUTPUT_DIR / "manual_review_instructions.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(instructions)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("EFFICIENTNETV2B0 VALIDATION ERROR REVIEW")
    print("=" * 80)
    print(f"Dataset             : {DATASET_DIR}")
    print(f"Manifest            : {MANIFEST_PATH}")
    print(f"Predictions         : {PREDICTIONS_PATH}")
    print(f"Metrics             : {METRICS_PATH}")
    print(f"Selected model      : {MODEL_PATH}")
    print(f"Output directory    : {OUTPUT_DIR}")
    print(
        f"Decision threshold  : {SELECTED_THRESHOLD:.4f}"
    )
    print(f"Images per sheet    : {IMAGES_PER_SHEET}")
    print("TEST usage          : NONE")
    print()

    ensure_required_files()

    print("Reading VALIDATION manifest...")
    validation_manifest = read_validation_manifest()
    print(
        f"Validation records  : {len(validation_manifest)}"
    )

    print("Validating saved metrics...")
    validation_metrics = validate_saved_metrics()

    print("Reading and validating saved predictions...")
    prediction_rows = read_and_validate_predictions(
        validation_manifest
    )
    print(f"Prediction records  : {len(prediction_rows)}")
    print()

    validate_validation_images(prediction_rows)

    false_positive_rows = [
        row
        for row in prediction_rows
        if row["result_type"] == "FALSE_POSITIVE"
    ]

    false_negative_rows = [
        row
        for row in prediction_rows
        if row["result_type"] == "FALSE_NEGATIVE"
    ]

    false_positive_rows.sort(
        key=lambda row: (
            -row["malicious_probability"],
            row["validation_index"],
        )
    )

    false_negative_rows.sort(
        key=lambda row: (
            row["malicious_probability"],
            row["validation_index"],
        )
    )

    if len(false_positive_rows) != EXPECTED_FALSE_POSITIVES:
        raise RuntimeError(
            f"Expected {EXPECTED_FALSE_POSITIVES} false "
            f"positives, but found {len(false_positive_rows)}."
        )

    if len(false_negative_rows) != EXPECTED_FALSE_NEGATIVES:
        raise RuntimeError(
            f"Expected {EXPECTED_FALSE_NEGATIVES} false "
            f"negatives, but found {len(false_negative_rows)}."
        )

    print()
    print("VALIDATION ERROR COUNTS")
    print("-" * 80)
    print(
        f"False positives     : {len(false_positive_rows)}"
    )
    print(
        f"False negatives     : {len(false_negative_rows)}"
    )
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Creating false-positive review sheets...")
    false_positive_sheets = create_review_sheets(
        false_positive_rows,
        "FP",
    )

    print()
    print("Creating false-negative review sheets...")
    false_negative_sheets = create_review_sheets(
        false_negative_rows,
        "FN",
    )

    print()
    print("Creating manual-review CSV files...")

    save_manual_review_csv(
        OUTPUT_DIR / "false_positive_manual_review.csv",
        false_positive_rows,
        "FP",
    )

    save_manual_review_csv(
        OUTPUT_DIR / "false_negative_manual_review.csv",
        false_negative_rows,
        "FN",
    )

    save_all_errors_csv(
        false_positive_rows,
        false_negative_rows,
    )

    save_review_instructions()

    print("Calculating reproducibility hashes...")

    input_hashes = {
        "manifest_sha256": calculate_sha256(MANIFEST_PATH),
        "validation_predictions_sha256": calculate_sha256(
            PREDICTIONS_PATH
        ),
        "validation_metrics_sha256": calculate_sha256(
            METRICS_PATH
        ),
        "selected_model_sha256": calculate_sha256(MODEL_PATH),
    }

    fp_statistics = probability_statistics(
        false_positive_rows
    )
    fn_statistics = probability_statistics(
        false_negative_rows
    )

    metadata = {
        "evaluation_split": "VALIDATION",
        "architecture": "EfficientNetV2B0",
        "selected_model_path": str(MODEL_PATH),
        "dataset_path": str(DATASET_DIR),
        "manifest_path": str(MANIFEST_PATH),
        "validation_predictions_path": str(
            PREDICTIONS_PATH
        ),
        "validation_metrics_path": str(METRICS_PATH),
        "decision_threshold": SELECTED_THRESHOLD,
        "threshold_source": "STEP_15_VALIDATION_ONLY",
        "validation_images": len(prediction_rows),
        "false_positives": len(false_positive_rows),
        "false_negatives": len(false_negative_rows),
        "false_positive_review_sheets": len(
            false_positive_sheets
        ),
        "false_negative_review_sheets": len(
            false_negative_sheets
        ),
        "images_per_sheet": IMAGES_PER_SHEET,
        "false_positive_probability_statistics": (
            fp_statistics
        ),
        "false_negative_probability_statistics": (
            fn_statistics
        ),
        "saved_validation_metrics": validation_metrics,
        "input_hashes": input_hashes,
        "test_split_loaded": False,
        "model_training_performed": False,
        "threshold_changed": False,
        "screenshots_modified": False,
    }

    write_json(
        OUTPUT_DIR / "validation_error_review_metadata.json",
        metadata,
    )

    summary_lines = [
        "=" * 80,
        "EFFICIENTNETV2B0 VALIDATION ERROR REVIEW COMPLETE",
        "=" * 80,
        "",
        f"Selected model        : {MODEL_PATH}",
        (
            "Model SHA256         : "
            f"{input_hashes['selected_model_sha256']}"
        ),
        f"Validation predictions: {PREDICTIONS_PATH}",
        (
            "Predictions SHA256   : "
            f"{input_hashes['validation_predictions_sha256']}"
        ),
        f"Decision threshold    : {SELECTED_THRESHOLD:.4f}",
        "",
        "VALIDATION ERRORS",
        "-" * 80,
        f"Validation images     : {len(prediction_rows)}",
        f"False positives       : {len(false_positive_rows)}",
        f"False negatives       : {len(false_negative_rows)}",
        "",
        "REVIEW SHEETS",
        "-" * 80,
        (
            "False-positive sheets: "
            f"{len(false_positive_sheets)}"
        ),
        (
            "False-negative sheets: "
            f"{len(false_negative_sheets)}"
        ),
        f"Images per sheet      : {IMAGES_PER_SHEET}",
        "Review order          : Most severe errors first",
        "",
        "FALSE-POSITIVE PROBABILITIES",
        "-" * 80,
        f"Minimum               : {fp_statistics['minimum']:.6f}",
        f"Maximum               : {fp_statistics['maximum']:.6f}",
        f"Mean                  : {fp_statistics['mean']:.6f}",
        f"Median                : {fp_statistics['median']:.6f}",
        "",
        "FALSE-NEGATIVE PROBABILITIES",
        "-" * 80,
        f"Minimum               : {fn_statistics['minimum']:.6f}",
        f"Maximum               : {fn_statistics['maximum']:.6f}",
        f"Mean                  : {fn_statistics['mean']:.6f}",
        f"Median                : {fn_statistics['median']:.6f}",
        "",
        "MANUAL REVIEW FILES",
        "-" * 80,
        str(
            OUTPUT_DIR / "false_positive_manual_review.csv"
        ),
        str(
            OUTPUT_DIR / "false_negative_manual_review.csv"
        ),
        str(OUTPUT_DIR / "manual_review_instructions.txt"),
        "",
        "IMPORTANT",
        "-" * 80,
        "This review uses VALIDATION predictions only.",
        "The diagnostic TEST split was not loaded.",
        "No model training was performed.",
        "The selected threshold was not changed.",
        "No original or prepared screenshot was modified.",
        "Do not move validation errors into TRAIN automatically.",
        "=" * 80,
    ]

    summary_text = "\n".join(summary_lines)

    with open(
        OUTPUT_DIR / "validation_error_review_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(summary_text)

    print()
    print(summary_text)
    print()
    print(f"All reports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print()
        print("=" * 80)
        print("VALIDATION ERROR REVIEW FAILED")
        print("=" * 80)
        print(error)
        print()
        print("No model training was performed.")
        print("The diagnostic TEST split was not loaded.")
        print("No screenshot was modified.")

        sys.exit(1)
