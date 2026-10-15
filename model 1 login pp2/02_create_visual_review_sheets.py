from __future__ import annotations

import csv
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


DATASET_ROOT = Path(
    r"C:\Users\oshan\Desktop\Research project\datasets\dataset\login dataset"
)

MANIFEST_PATH = Path(
    r"C:\Users\oshan\Desktop\model 1 login pp2"
    r"\reports\01_dataset_audit\image_manifest.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\oshan\Desktop\model 1 login pp2"
    r"\reports\02_visual_review"
)

RANDOM_SEED = 20260826
SAMPLE_SIZE = 12

COLUMNS = 3
ROWS = 4
CELL_WIDTH = 520
CELL_HEIGHT = 310
HEADER_HEIGHT = 55
IMAGE_AREA_HEIGHT = 235


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ["arial.ttf", "calibri.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue

    return ImageFont.load_default()


TITLE_FONT = load_font(24)
LABEL_FONT = load_font(15)


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        return [
            row
            for row in csv.DictReader(csv_file)
            if row["status"] == "VALID"
        ]


def select_sample(
    rows: list[dict[str, str]],
    sample_size: int,
    random_generator: random.Random,
) -> list[dict[str, str]]:
    if len(rows) <= sample_size:
        return list(rows)

    return random_generator.sample(rows, sample_size)


def shorten_text(value: str, width: int = 62) -> str:
    return textwrap.shorten(
        value,
        width=width,
        placeholder="...",
    )


def create_contact_sheet(
    selected_rows: list[dict[str, str]],
    title: str,
    output_name: str,
) -> None:
    if not selected_rows:
        print(f"No images found for: {title}")
        return

    images_per_sheet = COLUMNS * ROWS

    for page_start in range(0, len(selected_rows), images_per_sheet):
        page_rows = selected_rows[
            page_start:page_start + images_per_sheet
        ]

        page_number = (page_start // images_per_sheet) + 1

        canvas_width = COLUMNS * CELL_WIDTH
        canvas_height = HEADER_HEIGHT + ROWS * CELL_HEIGHT

        canvas = Image.new(
            "RGB",
            (canvas_width, canvas_height),
            "white",
        )

        draw = ImageDraw.Draw(canvas)

        draw.rectangle(
            (0, 0, canvas_width, HEADER_HEIGHT),
            fill=(35, 45, 65),
        )

        draw.text(
            (15, 13),
            f"{title} - Page {page_number}",
            fill="white",
            font=TITLE_FONT,
        )

        for index, row in enumerate(page_rows):
            column = index % COLUMNS
            sheet_row = index // COLUMNS

            x = column * CELL_WIDTH
            y = HEADER_HEIGHT + sheet_row * CELL_HEIGHT

            class_label = row["class_label"]

            if class_label == "MALICIOUS":
                border_color = (190, 50, 50)
            else:
                border_color = (40, 140, 70)

            draw.rectangle(
                (
                    x + 3,
                    y + 3,
                    x + CELL_WIDTH - 3,
                    y + CELL_HEIGHT - 3,
                ),
                outline=border_color,
                width=3,
            )

            image_path = DATASET_ROOT / row["relative_path"]

            try:
                with Image.open(image_path) as image:
                    image = ImageOps.exif_transpose(image)
                    image = image.convert("RGB")

                    image.thumbnail(
                        (
                            CELL_WIDTH - 20,
                            IMAGE_AREA_HEIGHT - 15,
                        ),
                        Image.Resampling.LANCZOS,
                    )

                    paste_x = x + (CELL_WIDTH - image.width) // 2
                    paste_y = y + 10 + (
                        IMAGE_AREA_HEIGHT - image.height
                    ) // 2

                    canvas.paste(
                        image,
                        (paste_x, paste_y),
                    )

            except Exception as error:
                draw.text(
                    (x + 15, y + 40),
                    f"IMAGE ERROR: {error}",
                    fill="red",
                    font=LABEL_FONT,
                )

            label_y = y + IMAGE_AREA_HEIGHT + 7

            label_lines = [
                (
                    f"{class_label} | "
                    f"{row['width']}x{row['height']}"
                ),
                shorten_text(row["filename"]),
                shorten_text(
                    row["review_reasons"]
                    if row["review_reasons"]
                    else "No audit warning"
                ),
            ]

            for line_number, line in enumerate(label_lines):
                draw.text(
                    (
                        x + 10,
                        label_y + line_number * 20,
                    ),
                    line,
                    fill=(20, 20, 20),
                    font=LABEL_FONT,
                )

        output_path = OUTPUT_DIR / (
            f"{output_name}_page_{page_number:02d}.png"
        )

        canvas.save(output_path, format="PNG")
        print(f"Created: {output_path.name}")


def main() -> None:
    print("=" * 80)
    print("MODEL 1 VISUAL REVIEW SHEET CREATION")
    print("=" * 80)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest was not found:\n{MANIFEST_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_manifest()
    random_generator = random.Random(RANDOM_SEED)

    for class_label in ["MALICIOUS", "NOT_MALICIOUS"]:
        class_rows = [
            row
            for row in rows
            if row["class_label"] == class_label
        ]

        random_sample = select_sample(
            class_rows,
            SAMPLE_SIZE,
            random_generator,
        )

        create_contact_sheet(
            random_sample,
            f"Random {class_label} samples",
            f"random_{class_label.lower()}",
        )

        mostly_white_rows = [
            row
            for row in class_rows
            if "MOSTLY_WHITE" in row["review_reasons"]
        ]

        mostly_white_sample = select_sample(
            mostly_white_rows,
            SAMPLE_SIZE,
            random_generator,
        )

        create_contact_sheet(
            mostly_white_sample,
            f"Mostly-white {class_label} samples",
            f"mostly_white_{class_label.lower()}",
        )

        low_entropy_rows = [
            row
            for row in class_rows
            if "LOW_ENTROPY" in row["review_reasons"]
        ]

        low_entropy_sample = select_sample(
            low_entropy_rows,
            SAMPLE_SIZE,
            random_generator,
        )

        create_contact_sheet(
            low_entropy_sample,
            f"Low-entropy {class_label} samples",
            f"low_entropy_{class_label.lower()}",
        )

    severe_rows = [
        row
        for row in rows
        if (
            "LOW_PIXEL_VARIATION" in row["review_reasons"]
            or "MOSTLY_BLACK" in row["review_reasons"]
        )
    ]

    create_contact_sheet(
        severe_rows,
        "All severe audit warnings",
        "severe_audit_warnings",
    )

    print()
    print("=" * 80)
    print("VISUAL REVIEW SHEETS CREATED")
    print("=" * 80)
    print(f"Output folder: {OUTPUT_DIR}")
    print()
    print("No original screenshots were modified.")
    print("=" * 80)


if __name__ == "__main__":
    main()