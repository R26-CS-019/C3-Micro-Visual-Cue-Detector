"""
08_create_test_error_review_sheets.py

Creates visual review sheets for Model 1 locked-test errors:

- 88 false positives
- 20 false negatives

This script:
- Does not train or change the model.
- Does not change the decision threshold.
- Does not modify, move, rename, or delete screenshots.
- Only reads the error CSV files and creates review images.
"""

import csv
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(
    r"C:\Users\oshan\Desktop\model 1 login pp2"
)

PREPARED_DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "prepared_login_384x160"
)

INPUT_REPORT_DIR = (
    PROJECT_DIR
    / "reports"
    / "07_locked_test_evaluation"
)

FALSE_POSITIVE_CSV = (
    INPUT_REPORT_DIR
    / "false_positives.csv"
)

FALSE_NEGATIVE_CSV = (
    INPUT_REPORT_DIR
    / "false_negatives.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "reports"
    / "08_test_error_review_sheets"
)

FALSE_POSITIVE_OUTPUT_DIR = (
    OUTPUT_DIR
    / "false_positives"
)

FALSE_NEGATIVE_OUTPUT_DIR = (
    OUTPUT_DIR
    / "false_negatives"
)

FALSE_POSITIVE_REVIEW_CSV = (
    OUTPUT_DIR
    / "false_positive_manual_review.csv"
)

FALSE_NEGATIVE_REVIEW_CSV = (
    OUTPUT_DIR
    / "false_negative_manual_review.csv"
)


# Expected counts from the locked test evaluation.

EXPECTED_FALSE_POSITIVES = 88
EXPECTED_FALSE_NEGATIVES = 20


# Review-sheet layout.

GRID_COLUMNS = 4
GRID_ROWS = 4
IMAGES_PER_SHEET = GRID_COLUMNS * GRID_ROWS

THUMBNAIL_WIDTH = 384
THUMBNAIL_HEIGHT = 160

TILE_PADDING = 12
TEXT_AREA_HEIGHT = 100

TILE_WIDTH = (
    THUMBNAIL_WIDTH
    + (TILE_PADDING * 2)
)

TILE_HEIGHT = (
    THUMBNAIL_HEIGHT
    + TEXT_AREA_HEIGHT
    + (TILE_PADDING * 2)
)

SHEET_HEADER_HEIGHT = 80

SHEET_WIDTH = GRID_COLUMNS * TILE_WIDTH

SHEET_HEIGHT = (
    SHEET_HEADER_HEIGHT
    + (GRID_ROWS * TILE_HEIGHT)
)


# Colours.

BACKGROUND_COLOUR = (240, 242, 245)
TILE_BACKGROUND = (255, 255, 255)
BORDER_COLOUR = (150, 150, 150)
HEADER_BACKGROUND = (32, 42, 56)
HEADER_TEXT_COLOUR = (255, 255, 255)
NORMAL_TEXT_COLOUR = (25, 25, 25)

FALSE_POSITIVE_COLOUR = (190, 40, 40)
FALSE_NEGATIVE_COLOUR = (210, 120, 0)

IMAGE_PLACEHOLDER_COLOUR = (225, 225, 225)
IMAGE_ERROR_TEXT_COLOUR = (170, 20, 20)


# =============================================================================
# FONT FUNCTIONS
# =============================================================================

def load_font(size, bold=False):
    """Load a readable Windows font with a safe fallback."""

    possible_fonts = []

    if bold:
        possible_fonts.extend(
            [
                Path(r"C:\Windows\Fonts\arialbd.ttf"),
                Path(r"C:\Windows\Fonts\calibrib.ttf"),
                Path(r"C:\Windows\Fonts\segoeuib.ttf"),
            ]
        )

    else:
        possible_fonts.extend(
            [
                Path(r"C:\Windows\Fonts\arial.ttf"),
                Path(r"C:\Windows\Fonts\calibri.ttf"),
                Path(r"C:\Windows\Fonts\segoeui.ttf"),
            ]
        )

    for font_path in possible_fonts:
        if font_path.exists():
            return ImageFont.truetype(
                str(font_path),
                size=size,
            )

    return ImageFont.load_default()


TITLE_FONT = load_font(25, bold=True)
SUBTITLE_FONT = load_font(16, bold=False)
TILE_TITLE_FONT = load_font(17, bold=True)
TILE_TEXT_FONT = load_font(14, bold=False)
TILE_SMALL_FONT = load_font(12, bold=False)


# =============================================================================
# INPUT FUNCTIONS
# =============================================================================

def read_error_csv(csv_path, expected_error_type):
    """Read and validate a locked-test error CSV."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Required error CSV was not found:\n"
            f"{csv_path}"
        )

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        fieldnames = reader.fieldnames or []
        rows = list(reader)

    required_columns = {
        "test_index",
        "prepared_relative_path",
        "prepared_image_path",
        "original_relative_path",
        "viewport_action",
        "actual_numeric_label",
        "actual_class",
        "malicious_probability",
        "decision_threshold",
        "predicted_numeric_label",
        "predicted_class",
        "error_type",
    }

    missing_columns = required_columns.difference(
        fieldnames
    )

    if missing_columns:
        raise RuntimeError(
            f"The following columns are missing from:\n"
            f"{csv_path}\n\n"
            f"Missing columns: {sorted(missing_columns)}"
        )

    validated_rows = []

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        error_type = (
            row["error_type"]
            .strip()
            .upper()
        )

        if error_type != expected_error_type:
            raise RuntimeError(
                f"Unexpected error type at CSV row "
                f"{row_number}.\n"
                f"Expected: {expected_error_type}\n"
                f"Found   : {error_type}"
            )

        try:
            probability = float(
                row["malicious_probability"]
            )

            test_index = int(
                row["test_index"]
            )

        except ValueError as error:
            raise RuntimeError(
                f"Invalid number at CSV row "
                f"{row_number} in:\n{csv_path}"
            ) from error

        row["_probability_number"] = probability
        row["_test_index_number"] = test_index

        validated_rows.append(row)

    return validated_rows


def resolve_image_path(row):
    """Resolve the prepared screenshot path safely."""

    absolute_path_text = (
        row.get(
            "prepared_image_path",
            "",
        )
        .strip()
    )

    if absolute_path_text:
        absolute_path = Path(
            absolute_path_text
        )

        if absolute_path.exists():
            return absolute_path

    relative_path_text = (
        row["prepared_relative_path"]
        .strip()
        .replace("\\", "/")
    )

    relative_path = Path(
        relative_path_text
    )

    fallback_path = (
        PREPARED_DATASET_DIR
        / relative_path
    )

    return fallback_path


# =============================================================================
# IMAGE FUNCTIONS
# =============================================================================

def create_fitted_thumbnail(image_path):
    """
    Load a screenshot and fit it into the thumbnail area.

    The source screenshot is only read. It is never modified.
    """

    with Image.open(image_path) as source_image:
        source_image = source_image.convert("RGB")

        source_image.thumbnail(
            (
                THUMBNAIL_WIDTH,
                THUMBNAIL_HEIGHT,
            ),
            Image.Resampling.LANCZOS,
        )

        thumbnail = Image.new(
            "RGB",
            (
                THUMBNAIL_WIDTH,
                THUMBNAIL_HEIGHT,
            ),
            color=(245, 245, 245),
        )

        paste_x = (
            THUMBNAIL_WIDTH
            - source_image.width
        ) // 2

        paste_y = (
            THUMBNAIL_HEIGHT
            - source_image.height
        ) // 2

        thumbnail.paste(
            source_image,
            (
                paste_x,
                paste_y,
            ),
        )

    return thumbnail


def shorten_text(text, maximum_length):
    """Shorten long text so it fits inside a review tile."""

    text = str(text).strip()

    if len(text) <= maximum_length:
        return text

    return (
        text[: maximum_length - 3]
        + "..."
    )


def draw_multiline_text(
    draw,
    position,
    lines,
    font,
    fill,
    line_spacing,
):
    """Draw a small group of text lines."""

    x_position, y_position = position

    for line in lines:
        draw.text(
            (
                x_position,
                y_position,
            ),
            line,
            font=font,
            fill=fill,
        )

        y_position += line_spacing


# =============================================================================
# REVIEW-SHEET CREATION
# =============================================================================

def create_review_sheet(
    rows,
    error_type,
    sheet_number,
    total_sheets,
    output_path,
):
    """Create one visual error-review sheet."""

    if error_type == "FALSE_POSITIVE":
        error_colour = FALSE_POSITIVE_COLOUR
        short_name = "FP"
        explanation = (
            "Actual NOT_MALICIOUS, "
            "predicted MALICIOUS"
        )

    else:
        error_colour = FALSE_NEGATIVE_COLOUR
        short_name = "FN"
        explanation = (
            "Actual MALICIOUS, "
            "predicted NOT_MALICIOUS"
        )

    sheet = Image.new(
        "RGB",
        (
            SHEET_WIDTH,
            SHEET_HEIGHT,
        ),
        color=BACKGROUND_COLOUR,
    )

    draw = ImageDraw.Draw(sheet)

    draw.rectangle(
        [
            0,
            0,
            SHEET_WIDTH,
            SHEET_HEADER_HEIGHT,
        ],
        fill=HEADER_BACKGROUND,
    )

    title = (
        f"Model 1 Locked Test "
        f"{error_type.replace('_', ' ')} Review"
    )

    subtitle = (
        f"Sheet {sheet_number}/{total_sheets} | "
        f"{explanation} | "
        f"Threshold = 0.1900"
    )

    draw.text(
        (18, 12),
        title,
        font=TITLE_FONT,
        fill=HEADER_TEXT_COLOUR,
    )

    draw.text(
        (18, 47),
        subtitle,
        font=SUBTITLE_FONT,
        fill=HEADER_TEXT_COLOUR,
    )

    for local_index, row in enumerate(rows):
        grid_row = (
            local_index
            // GRID_COLUMNS
        )

        grid_column = (
            local_index
            % GRID_COLUMNS
        )

        tile_x = grid_column * TILE_WIDTH

        tile_y = (
            SHEET_HEADER_HEIGHT
            + (grid_row * TILE_HEIGHT)
        )

        tile_left = tile_x + 5
        tile_top = tile_y + 5

        tile_right = (
            tile_x
            + TILE_WIDTH
            - 5
        )

        tile_bottom = (
            tile_y
            + TILE_HEIGHT
            - 5
        )

        draw.rounded_rectangle(
            [
                tile_left,
                tile_top,
                tile_right,
                tile_bottom,
            ],
            radius=8,
            fill=TILE_BACKGROUND,
            outline=BORDER_COLOUR,
            width=2,
        )

        image_x = (
            tile_x
            + TILE_PADDING
        )

        image_y = (
            tile_y
            + TILE_PADDING
        )

        image_path = resolve_image_path(
            row
        )

        if image_path.exists():
            try:
                thumbnail = create_fitted_thumbnail(
                    image_path
                )

                sheet.paste(
                    thumbnail,
                    (
                        image_x,
                        image_y,
                    ),
                )

            except Exception as image_error:
                draw.rectangle(
                    [
                        image_x,
                        image_y,
                        image_x + THUMBNAIL_WIDTH,
                        image_y + THUMBNAIL_HEIGHT,
                    ],
                    fill=IMAGE_PLACEHOLDER_COLOUR,
                )

                draw.text(
                    (
                        image_x + 10,
                        image_y + 65,
                    ),
                    shorten_text(
                        f"Image error: {image_error}",
                        45,
                    ),
                    font=TILE_TEXT_FONT,
                    fill=IMAGE_ERROR_TEXT_COLOUR,
                )

        else:
            draw.rectangle(
                [
                    image_x,
                    image_y,
                    image_x + THUMBNAIL_WIDTH,
                    image_y + THUMBNAIL_HEIGHT,
                ],
                fill=IMAGE_PLACEHOLDER_COLOUR,
            )

            draw.text(
                (
                    image_x + 10,
                    image_y + 65,
                ),
                "IMAGE NOT FOUND",
                font=TILE_TITLE_FONT,
                fill=IMAGE_ERROR_TEXT_COLOUR,
            )

        text_x = image_x

        text_y = (
            image_y
            + THUMBNAIL_HEIGHT
            + 7
        )

        review_number = (
            ((sheet_number - 1) * IMAGES_PER_SHEET)
            + local_index
            + 1
        )

        probability = row[
            "_probability_number"
        ]

        filename = Path(
            row["prepared_relative_path"]
        ).name

        viewport_action = (
            row["viewport_action"]
            .strip()
        )

        title_line = (
            f"{short_name}-{review_number:03d} | "
            f"Test #{row['_test_index_number']}"
        )

        draw.text(
            (
                text_x,
                text_y,
            ),
            title_line,
            font=TILE_TITLE_FONT,
            fill=error_colour,
        )

        text_lines = [
            (
                f"Malicious probability: "
                f"{probability:.4f}"
            ),
            (
                f"Actual: {row['actual_class']} | "
                f"Predicted: {row['predicted_class']}"
            ),
            (
                f"Crop: {viewport_action}"
            ),
            (
                f"File: "
                f"{shorten_text(filename, 42)}"
            ),
        ]

        draw_multiline_text(
            draw=draw,
            position=(
                text_x,
                text_y + 24,
            ),
            lines=text_lines,
            font=TILE_SMALL_FONT,
            fill=NORMAL_TEXT_COLOUR,
            line_spacing=17,
        )

    sheet.save(
        output_path,
        format="PNG",
        optimize=True,
    )


def create_all_review_sheets(
    rows,
    error_type,
    output_directory,
):
    """Create every review sheet for one error type."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        print(
            f"No {error_type} rows were found. "
            f"No sheets were created."
        )

        return 0

    total_sheets = math.ceil(
        len(rows)
        / IMAGES_PER_SHEET
    )

    if error_type == "FALSE_POSITIVE":
        file_prefix = "FP_sheet"

    else:
        file_prefix = "FN_sheet"

    for sheet_index in range(
        total_sheets
    ):
        start_index = (
            sheet_index
            * IMAGES_PER_SHEET
        )

        end_index = (
            start_index
            + IMAGES_PER_SHEET
        )

        sheet_rows = rows[
            start_index:end_index
        ]

        sheet_number = (
            sheet_index + 1
        )

        output_path = (
            output_directory
            / (
                f"{file_prefix}_"
                f"{sheet_number:02d}.png"
            )
        )

        create_review_sheet(
            rows=sheet_rows,
            error_type=error_type,
            sheet_number=sheet_number,
            total_sheets=total_sheets,
            output_path=output_path,
        )

        print(
            f"Created: {output_path.name} "
            f"({len(sheet_rows)} images)"
        )

    return total_sheets


# =============================================================================
# MANUAL-REVIEW CSV
# =============================================================================

def create_manual_review_csv(
    rows,
    error_type,
    output_path,
):
    """
    Create a CSV where manual error-analysis findings can be entered.
    """

    fieldnames = [
        "review_id",
        "error_type",
        "test_index",
        "prepared_relative_path",
        "prepared_image_path",
        "original_relative_path",
        "viewport_action",
        "actual_class",
        "predicted_class",
        "malicious_probability",
        "decision_threshold",
        "manual_review_status",
        "page_type",
        "suspected_error_reason",
        "visual_observations",
        "dataset_or_label_issue",
        "recommended_action",
        "reviewer_notes",
    ]

    output_rows = []

    if error_type == "FALSE_POSITIVE":
        prefix = "FP"

    else:
        prefix = "FN"

    for review_index, row in enumerate(
        rows,
        start=1,
    ):
        image_path = resolve_image_path(
            row
        )

        output_rows.append(
            {
                "review_id": (
                    f"{prefix}-{review_index:03d}"
                ),
                "error_type": error_type,
                "test_index": (
                    row["test_index"]
                ),
                "prepared_relative_path": (
                    row["prepared_relative_path"]
                ),
                "prepared_image_path": str(
                    image_path
                ),
                "original_relative_path": (
                    row["original_relative_path"]
                ),
                "viewport_action": (
                    row["viewport_action"]
                ),
                "actual_class": (
                    row["actual_class"]
                ),
                "predicted_class": (
                    row["predicted_class"]
                ),
                "malicious_probability": (
                    row["malicious_probability"]
                ),
                "decision_threshold": (
                    row["decision_threshold"]
                ),
                "manual_review_status": (
                    "PENDING"
                ),
                "page_type": "",
                "suspected_error_reason": "",
                "visual_observations": "",
                "dataset_or_label_issue": "",
                "recommended_action": "",
                "reviewer_notes": "",
            }
        )

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


# =============================================================================
# SUMMARY
# =============================================================================

def create_summary(
    false_positive_count,
    false_negative_count,
    false_positive_sheets,
    false_negative_sheets,
):
    """Save a short summary of generated review files."""

    summary_lines = [
        "=" * 80,
        "MODEL 1 LOCKED TEST ERROR REVIEW SHEETS COMPLETE",
        "=" * 80,
        "",
        "INPUT",
        "-" * 80,
        f"False-positive CSV : {FALSE_POSITIVE_CSV}",
        f"False-negative CSV : {FALSE_NEGATIVE_CSV}",
        "",
        "RESULTS",
        "-" * 80,
        (
            f"False positives    : "
            f"{false_positive_count}"
        ),
        (
            f"FP review sheets   : "
            f"{false_positive_sheets}"
        ),
        (
            f"False negatives    : "
            f"{false_negative_count}"
        ),
        (
            f"FN review sheets   : "
            f"{false_negative_sheets}"
        ),
        "",
        "MANUAL REVIEW FILES",
        "-" * 80,
        str(FALSE_POSITIVE_REVIEW_CSV),
        str(FALSE_NEGATIVE_REVIEW_CSV),
        "",
        "IMPORTANT",
        "-" * 80,
        "No model training was performed.",
        "The locked threshold was not changed.",
        "No original or prepared screenshot was modified.",
        "These sheets are only for error analysis.",
        "=" * 80,
    ]

    summary_text = "\n".join(
        summary_lines
    )

    summary_path = (
        OUTPUT_DIR
        / "error_review_summary.txt"
    )

    with open(
        summary_path,
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
    print("MODEL 1 LOCKED TEST ERROR REVIEW SHEETS")
    print("=" * 80)

    print(
        f"Input reports       : "
        f"{INPUT_REPORT_DIR}"
    )

    print(
        f"Output directory    : "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Images per sheet    : "
        f"{IMAGES_PER_SHEET}"
    )

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Reading false positives...")

    false_positive_rows = read_error_csv(
        csv_path=FALSE_POSITIVE_CSV,
        expected_error_type="FALSE_POSITIVE",
    )

    print(
        f"False positives     : "
        f"{len(false_positive_rows)}"
    )

    print("Reading false negatives...")

    false_negative_rows = read_error_csv(
        csv_path=FALSE_NEGATIVE_CSV,
        expected_error_type="FALSE_NEGATIVE",
    )

    print(
        f"False negatives     : "
        f"{len(false_negative_rows)}"
    )

    if (
        len(false_positive_rows)
        != EXPECTED_FALSE_POSITIVES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_FALSE_POSITIVES} "
            f"false positives, but found "
            f"{len(false_positive_rows)}."
        )

    if (
        len(false_negative_rows)
        != EXPECTED_FALSE_NEGATIVES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_FALSE_NEGATIVES} "
            f"false negatives, but found "
            f"{len(false_negative_rows)}."
        )

    # Show the strongest false positives first.

    false_positive_rows.sort(
        key=lambda row: row[
            "_probability_number"
        ],
        reverse=True,
    )

    # Show the lowest-probability false negatives first.

    false_negative_rows.sort(
        key=lambda row: row[
            "_probability_number"
        ],
    )

    print()
    print("Creating false-positive review sheets...")

    false_positive_sheets = (
        create_all_review_sheets(
            rows=false_positive_rows,
            error_type="FALSE_POSITIVE",
            output_directory=(
                FALSE_POSITIVE_OUTPUT_DIR
            ),
        )
    )

    print()
    print("Creating false-negative review sheets...")

    false_negative_sheets = (
        create_all_review_sheets(
            rows=false_negative_rows,
            error_type="FALSE_NEGATIVE",
            output_directory=(
                FALSE_NEGATIVE_OUTPUT_DIR
            ),
        )
    )

    print()
    print("Creating manual-review CSV files...")

    create_manual_review_csv(
        rows=false_positive_rows,
        error_type="FALSE_POSITIVE",
        output_path=FALSE_POSITIVE_REVIEW_CSV,
    )

    create_manual_review_csv(
        rows=false_negative_rows,
        error_type="FALSE_NEGATIVE",
        output_path=FALSE_NEGATIVE_REVIEW_CSV,
    )

    summary_text = create_summary(
        false_positive_count=len(
            false_positive_rows
        ),
        false_negative_count=len(
            false_negative_rows
        ),
        false_positive_sheets=(
            false_positive_sheets
        ),
        false_negative_sheets=(
            false_negative_sheets
        ),
    )

    print()
    print(summary_text)

    print()
    print(
        f"All reports saved to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print()
        print("=" * 80)
        print("ERROR REVIEW SHEET CREATION FAILED")
        print("=" * 80)
        print(error)
        print()
        print(
            "No screenshot, model, threshold, "
            "or dataset split was modified."
        )

        sys.exit(1)