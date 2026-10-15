from __future__ import annotations

import csv
import html
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\oshan\Desktop\new model 2 pp2"
)

FINAL_TEST_FOLDER = (
    PROJECT_ROOT
    / "reports"
    / "model2_effnetv2b0"
    / "final_test"
)

FALSE_POSITIVES_CSV = (
    FINAL_TEST_FOLDER
    / "final_test_false_positives.csv"
)

FALSE_NEGATIVES_CSV = (
    FINAL_TEST_FOLDER
    / "final_test_false_negatives.csv"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "reports"
    / "model2_effnetv2b0"
    / "error_analysis"
)

FALSE_POSITIVE_FOLDER = (
    OUTPUT_FOLDER
    / "false_positive_sheets"
)

FALSE_NEGATIVE_FOLDER = (
    OUTPUT_FOLDER
    / "false_negative_sheets"
)

DECISIONS_CSV = (
    OUTPUT_FOLDER
    / "error_review_decisions.csv"
)

HTML_REPORT = (
    OUTPUT_FOLDER
    / "error_review_index.html"
)

SUMMARY_FILE = (
    OUTPUT_FOLDER
    / "error_review_summary.txt"
)

EXPECTED_FALSE_POSITIVES = 184
EXPECTED_FALSE_NEGATIVES = 55


# =============================================================================
# CONTACT-SHEET SETTINGS
# =============================================================================

TILE_WIDTH = 620
TILE_HEIGHT = 410
HEADER_HEIGHT = 100
COLUMNS = 2
ROWS_PER_SHEET = 4
IMAGES_PER_SHEET = COLUMNS * ROWS_PER_SHEET

try:
    RESAMPLE_METHOD = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_METHOD = Image.LANCZOS


def normalize_header(value: str) -> str:
    return "".join(
        character
        for character in str(value).lower()
        if character.isalnum()
    )


def find_column(
    fieldnames: list[str],
    possible_names: list[str],
    description: str,
) -> str:
    normalized = {
        normalize_header(name): name
        for name in fieldnames
    }

    for possible_name in possible_names:
        key = normalize_header(possible_name)

        if key in normalized:
            return normalized[key]

    raise SystemExit(
        f"\nERROR: Could not find the {description} column.\n"
        f"Available columns:\n{fieldnames}"
    )


def load_font(size: int) -> ImageFont.ImageFont:
    possible_fonts = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    ]

    for font_path in possible_fonts:
        if font_path.is_file():
            try:
                return ImageFont.truetype(
                    str(font_path),
                    size=size,
                )
            except OSError:
                pass

    return ImageFont.load_default()


TITLE_FONT = load_font(27)
LABEL_FONT = load_font(19)
SMALL_FONT = load_font(16)


def shorten(value: str, maximum: int) -> str:
    value = str(value)

    if len(value) <= maximum:
        return value

    return value[: maximum - 3] + "..."


def load_error_csv(
    csv_path: Path,
    expected_count: int,
    error_type: str,
) -> tuple[
    list[dict[str, str]],
    list[str],
    dict[str, str],
]:
    if not csv_path.is_file():
        raise SystemExit(
            f"\nERROR: {error_type} report was not found:\n"
            f"{csv_path}"
        )

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise SystemExit(
                f"\nERROR: {error_type} report has no header."
            )

        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if len(rows) != expected_count:
        raise SystemExit(
            f"\nERROR: Unexpected {error_type} count.\n"
            f"Expected: {expected_count:,}\n"
            f"Found: {len(rows):,}"
        )

    columns = {
        "path": find_column(
            fieldnames,
            [
                "screenshot_path",
                "image_path",
                "file_path",
                "path",
            ],
            "screenshot path",
        ),
        "sample": find_column(
            fieldnames,
            [
                "sample_folder",
                "sample_name",
                "sample_id",
                "sample",
                "folder_name",
            ],
            "sample name",
        ),
        "probability": find_column(
            fieldnames,
            ["phishing_probability", "probability", "score"],
            "phishing probability",
        ),
        "true_label": find_column(
            fieldnames,
            ["label", "class", "class_label", "dataset_label"],
            "true label",
        ),
        "predicted_label": find_column(
            fieldnames,
            ["predicted_label", "prediction"],
            "predicted label",
        ),
    }

    for row in rows:
        try:
            float(row[columns["probability"]])
        except ValueError as error:
            raise SystemExit(
                f"\nERROR: Invalid probability in {csv_path}:\n"
                f"{row[columns['probability']]}"
            ) from error

    if error_type == "FALSE_POSITIVE":
        # Most confidently wrong false alarms first.
        rows.sort(
            key=lambda row: float(
                row[columns["probability"]]
            ),
            reverse=True,
        )
    else:
        # Most confidently missed phishing pages first.
        rows.sort(
            key=lambda row: float(
                row[columns["probability"]]
            )
        )

    return rows, fieldnames, columns


def draw_error_tile(
    canvas: Image.Image,
    drawing: ImageDraw.ImageDraw,
    row: dict[str, str],
    columns: dict[str, str],
    error_type: str,
    overall_review_number: int,
    tile_left: int,
    tile_top: int,
) -> None:
    tile_right = tile_left + TILE_WIDTH
    tile_bottom = tile_top + TILE_HEIGHT

    drawing.rectangle(
        (
            tile_left + 5,
            tile_top + 5,
            tile_right - 5,
            tile_bottom - 5,
        ),
        fill=(255, 255, 255),
        outline=(105, 115, 130),
        width=3,
    )

    if error_type == "FALSE_POSITIVE":
        heading_colour = (190, 80, 20)
    else:
        heading_colour = (185, 30, 45)

    probability = float(row[columns["probability"]])
    sample = row[columns["sample"]].strip()
    true_label = row[columns["true_label"]].strip().upper()
    predicted_label = (
        row[columns["predicted_label"]].strip().upper()
    )

    drawing.text(
        (tile_left + 18, tile_top + 15),
        f"Review {overall_review_number:03d} | {error_type}",
        fill=heading_colour,
        font=LABEL_FONT,
    )

    drawing.text(
        (tile_left + 18, tile_top + 45),
        shorten(sample, 58),
        fill=(25, 25, 25),
        font=LABEL_FONT,
    )

    details = (
        f"True: {true_label} | Predicted: {predicted_label} | "
        f"Phishing probability: {probability:.4f}"
    )

    drawing.text(
        (tile_left + 18, tile_top + 74),
        details,
        fill=(65, 65, 65),
        font=SMALL_FONT,
    )

    image_area_left = tile_left + 15
    image_area_top = tile_top + 105
    image_area_width = TILE_WIDTH - 30
    image_area_height = TILE_HEIGHT - 125

    screenshot_path = Path(row[columns["path"]])

    try:
        with Image.open(screenshot_path) as screenshot:
            screenshot = ImageOps.exif_transpose(screenshot)
            screenshot = screenshot.convert("RGB")

            screenshot.thumbnail(
                (image_area_width, image_area_height),
                RESAMPLE_METHOD,
            )

            paste_left = (
                image_area_left
                + (image_area_width - screenshot.width) // 2
            )

            paste_top = (
                image_area_top
                + (image_area_height - screenshot.height) // 2
            )

            canvas.paste(
                screenshot,
                (paste_left, paste_top),
            )

    except Exception as error:
        drawing.text(
            (image_area_left + 20, image_area_top + 70),
            (
                "IMAGE OPEN ERROR:\n"
                f"{type(error).__name__}: {error}"
            ),
            fill=(190, 30, 45),
            font=LABEL_FONT,
        )


def create_sheets(
    rows: list[dict[str, str]],
    columns: dict[str, str],
    error_type: str,
    output_folder: Path,
) -> list[str]:
    output_folder.mkdir(parents=True, exist_ok=True)

    number_of_sheets = math.ceil(
        len(rows) / IMAGES_PER_SHEET
    )

    sheet_names: list[str] = []

    for sheet_number in range(1, number_of_sheets + 1):
        start_index = (
            sheet_number - 1
        ) * IMAGES_PER_SHEET

        sheet_rows = rows[
            start_index:
            start_index + IMAGES_PER_SHEET
        ]

        number_of_tile_rows = math.ceil(
            len(sheet_rows) / COLUMNS
        )

        canvas_width = COLUMNS * TILE_WIDTH
        canvas_height = (
            HEADER_HEIGHT
            + number_of_tile_rows * TILE_HEIGHT
        )

        canvas = Image.new(
            "RGB",
            (canvas_width, canvas_height),
            color=(235, 238, 242),
        )

        drawing = ImageDraw.Draw(canvas)

        header_colour = (
            (125, 65, 20)
            if error_type == "FALSE_POSITIVE"
            else (135, 25, 40)
        )

        drawing.rectangle(
            (0, 0, canvas_width, HEADER_HEIGHT),
            fill=header_colour,
        )

        header_text = (
            f"Model 2 {error_type.replace('_', ' ').title()} "
            f"Review | Sheet {sheet_number}/{number_of_sheets} | "
            f"Sorted by strongest wrong confidence"
        )

        drawing.text(
            (25, 32),
            header_text,
            fill=(255, 255, 255),
            font=TITLE_FONT,
        )

        for local_index, row in enumerate(sheet_rows):
            column_number = local_index % COLUMNS
            row_number = local_index // COLUMNS

            draw_error_tile(
                canvas=canvas,
                drawing=drawing,
                row=row,
                columns=columns,
                error_type=error_type,
                overall_review_number=start_index + local_index + 1,
                tile_left=column_number * TILE_WIDTH,
                tile_top=(
                    HEADER_HEIGHT + row_number * TILE_HEIGHT
                ),
            )

        filename = (
            f"{error_type.lower()}_sheet_{sheet_number:03d}.jpg"
        )

        canvas.save(
            output_folder / filename,
            format="JPEG",
            quality=92,
            optimize=True,
        )

        sheet_names.append(filename)

        print(
            f"Created {error_type} sheet "
            f"{sheet_number:,}/{number_of_sheets:,}"
        )

    return sheet_names


def create_decisions_csv(
    false_positive_rows: list[dict[str, str]],
    false_positive_columns: dict[str, str],
    false_negative_rows: list[dict[str, str]],
    false_negative_columns: dict[str, str],
) -> None:
    decision_rows: list[dict[str, str]] = []

    def add_rows(
        rows: list[dict[str, str]],
        columns: dict[str, str],
        error_type: str,
    ) -> None:
        for review_number, row in enumerate(rows, start=1):
            sheet_number = (
                (review_number - 1) // IMAGES_PER_SHEET
            ) + 1

            decision_rows.append(
                {
                    "error_type": error_type,
                    "review_number": str(review_number),
                    "sheet_number": str(sheet_number),
                    "sample": row[columns["sample"]],
                    "true_label": row[columns["true_label"]],
                    "predicted_label": row[
                        columns["predicted_label"]
                    ],
                    "phishing_probability": row[
                        columns["probability"]
                    ],
                    "screenshot_path": row[columns["path"]],
                    "error_category": "REVIEW_REQUIRED",
                    "review_notes": "",
                }
            )

    add_rows(
        false_positive_rows,
        false_positive_columns,
        "FALSE_POSITIVE",
    )

    add_rows(
        false_negative_rows,
        false_negative_columns,
        "FALSE_NEGATIVE",
    )

    fieldnames = [
        "error_type",
        "review_number",
        "sheet_number",
        "sample",
        "true_label",
        "predicted_label",
        "phishing_probability",
        "screenshot_path",
        "error_category",
        "review_notes",
    ]

    with DECISIONS_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(decision_rows)


def create_html_report(
    false_positive_sheets: list[str],
    false_negative_sheets: list[str],
) -> None:
    sections: list[str] = []

    for title, folder_name, sheet_names in [
        (
            "False Positives — Legitimate Predicted as Phishing",
            FALSE_POSITIVE_FOLDER.name,
            false_positive_sheets,
        ),
        (
            "False Negatives — Phishing Predicted as Legitimate",
            FALSE_NEGATIVE_FOLDER.name,
            false_negative_sheets,
        ),
    ]:
        images = "".join(
            f"""
            <section class="sheet">
                <h3>{html.escape(sheet_name)}</h3>
                <img
                    src="{html.escape(folder_name)}/{html.escape(sheet_name)}"
                    alt="{html.escape(sheet_name)}"
                >
            </section>
            """
            for sheet_name in sheet_names
        )

        sections.append(
            f"""
            <h2>{html.escape(title)}</h2>
            {images}
            """
        )

    content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Model 2 Error Review</title>
        <style>
            body {{
                background: #eef1f5;
                color: #202735;
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 30px;
            }}
            .instructions, .sheet {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.12);
                margin-bottom: 30px;
                padding: 20px;
            }}
            .instructions {{
                border-left: 6px solid #2457a7;
            }}
            img {{
                border: 1px solid #aab1bd;
                display: block;
                height: auto;
                max-width: 100%;
            }}
            h1, h2 {{ color: #183c72; }}
            code {{
                background: #e7eaf0;
                padding: 3px 6px;
            }}
        </style>
    </head>
    <body>
        <div class="instructions">
            <h1>Model 2 Fixed-Test Error Review</h1>
            <p>
                Review the strongest errors first and record a category in
                <code>error_review_decisions.csv</code>.
            </p>
            <p>Suggested categories:</p>
            <ul>
                <li>GENERIC_LOGIN_OR_FORM_LAYOUT</li>
                <li>BRAND_OR_TEMPLATE_SIMILARITY</li>
                <li>TEXT_HEAVY_OR_CONTENT_PAGE</li>
                <li>BLANK_PARTIAL_OR_LOADING_PAGE</li>
                <li>IMAGE_QUALITY_OR_CROPPING</li>
                <li>POSSIBLE_LABEL_CONCERN</li>
                <li>OTHER_VISUAL_PATTERN</li>
            </ul>
            <p>
                Do not retrain the finalized model using these test errors.
            </p>
        </div>
        {''.join(sections)}
    </body>
    </html>
    """

    HTML_REPORT.write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 80)
    print("MODEL 2 FIXED-TEST ERROR REVIEW SHEET CREATOR")
    print("=" * 80)
    print("This script creates review reports only.")
    print("It does not train or modify the finalized model.")
    print("It does not modify original screenshots.")
    print()

    (
        false_positive_rows,
        _,
        false_positive_columns,
    ) = load_error_csv(
        FALSE_POSITIVES_CSV,
        EXPECTED_FALSE_POSITIVES,
        "FALSE_POSITIVE",
    )

    (
        false_negative_rows,
        _,
        false_negative_columns,
    ) = load_error_csv(
        FALSE_NEGATIVES_CSV,
        EXPECTED_FALSE_NEGATIVES,
        "FALSE_NEGATIVE",
    )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    false_positive_sheets = create_sheets(
        false_positive_rows,
        false_positive_columns,
        "FALSE_POSITIVE",
        FALSE_POSITIVE_FOLDER,
    )

    false_negative_sheets = create_sheets(
        false_negative_rows,
        false_negative_columns,
        "FALSE_NEGATIVE",
        FALSE_NEGATIVE_FOLDER,
    )

    create_decisions_csv(
        false_positive_rows,
        false_positive_columns,
        false_negative_rows,
        false_negative_columns,
    )

    create_html_report(
        false_positive_sheets,
        false_negative_sheets,
    )

    summary_lines = [
        "=" * 80,
        "MODEL 2 ERROR REVIEW SUMMARY",
        "=" * 80,
        "",
        f"False positives: {len(false_positive_rows):,}",
        f"False-positive sheets: {len(false_positive_sheets):,}",
        f"False negatives: {len(false_negative_rows):,}",
        f"False-negative sheets: {len(false_negative_sheets):,}",
        f"Images per full sheet: {IMAGES_PER_SHEET}",
        "",
        "REPORT FILES",
        "-" * 80,
        str(HTML_REPORT),
        str(DECISIONS_CSV),
        str(FALSE_POSITIVE_FOLDER),
        str(FALSE_NEGATIVE_FOLDER),
        str(SUMMARY_FILE),
        "",
        "IMPORTANT",
        "-" * 80,
        "No original screenshot or model file was modified.",
        "Do not retrain using fixed-test errors.",
        "=" * 80,
    ]

    summary_text = "\n".join(summary_lines)

    SUMMARY_FILE.write_text(
        summary_text,
        encoding="utf-8",
    )

    print()
    print(summary_text)


if __name__ == "__main__":
    main()
