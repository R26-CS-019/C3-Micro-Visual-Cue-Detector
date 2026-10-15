from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\oshan\Desktop\new model 2 pp2"
)

REPORTS_FOLDER = PROJECT_ROOT / "reports"

NEAR_DUPLICATE_GROUPS_CSV = (
    REPORTS_FOLDER
    / "phish360_near_duplicate_groups.csv"
)

OUTPUT_FOLDER = (
    REPORTS_FOLDER
    / "cross_split_review"
)

DECISIONS_CSV = (
    OUTPUT_FOLDER
    / "cross_split_review_decisions.csv"
)

HTML_REPORT = (
    OUTPUT_FOLDER
    / "cross_split_review_index.html"
)


# ============================================================================
# IMAGE SETTINGS
# ============================================================================

TILE_WIDTH = 620
TILE_HEIGHT = 520
HEADER_HEIGHT = 90
LABEL_HEIGHT = 100
COLUMNS = 2

try:
    RESAMPLE_METHOD = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_METHOD = Image.LANCZOS


def is_true(value: str) -> bool:
    """Convert a CSV value into a Boolean value."""
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def shorten_text(
    text: str,
    maximum_characters: int,
) -> str:
    """Shorten long text so it fits on the review sheet."""
    text = str(text)

    if len(text) <= maximum_characters:
        return text

    return text[: maximum_characters - 3] + "..."


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a readable Windows font."""
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


TITLE_FONT = load_font(26)
LABEL_FONT = load_font(19)
SMALL_FONT = load_font(16)


def load_cross_split_groups() -> dict[str, list[dict[str, str]]]:
    """Load only groups crossing trainval and test."""
    if not NEAR_DUPLICATE_GROUPS_CSV.is_file():
        raise SystemExit(
            "\nERROR: Near-duplicate group report was not found.\n"
            f"Expected file:\n{NEAR_DUPLICATE_GROUPS_CSV}\n\n"
            "Run audit_phish360_near_duplicates.py first."
        )

    groups: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    with NEAR_DUPLICATE_GROUPS_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if is_true(row.get("cross_split", "")):
                group_number = row[
                    "near_duplicate_group"
                ]

                groups[group_number].append(row)

    return dict(groups)


def load_existing_decisions() -> dict[str, dict[str, str]]:
    """Preserve decisions if the script is run again."""
    existing_decisions: dict[
        str,
        dict[str, str],
    ] = {}

    if not DECISIONS_CSV.is_file():
        return existing_decisions

    with DECISIONS_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            group_number = row.get(
                "near_duplicate_group",
                "",
            )

            if group_number:
                existing_decisions[group_number] = row

    return existing_decisions


def create_contact_sheet(
    group_number: str,
    group_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    """Create one contact sheet for a near-duplicate group."""
    group_size = len(group_rows)

    number_of_columns = min(
        COLUMNS,
        group_size,
    )

    number_of_rows = math.ceil(
        group_size / number_of_columns
    )

    canvas_width = (
        number_of_columns * TILE_WIDTH
    )

    canvas_height = (
        HEADER_HEIGHT
        + number_of_rows * TILE_HEIGHT
    )

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        color=(235, 238, 242),
    )

    drawing = ImageDraw.Draw(canvas)

    group_splits = sorted(
        {
            row["split"]
            for row in group_rows
        }
    )

    group_labels = sorted(
        {
            row["label"]
            for row in group_rows
        }
    )

    header_text = (
        f"Near-Duplicate Group {group_number}  |  "
        f"Images: {group_size}  |  "
        f"Splits: {' / '.join(group_splits)}  |  "
        f"Labels: {' / '.join(group_labels)}"
    )

    drawing.rectangle(
        (
            0,
            0,
            canvas_width,
            HEADER_HEIGHT,
        ),
        fill=(25, 45, 75),
    )

    drawing.text(
        (25, 28),
        header_text,
        fill=(255, 255, 255),
        font=TITLE_FONT,
    )

    for image_number, row in enumerate(group_rows):
        column_number = (
            image_number % number_of_columns
        )

        row_number = (
            image_number // number_of_columns
        )

        tile_left = (
            column_number * TILE_WIDTH
        )

        tile_top = (
            HEADER_HEIGHT
            + row_number * TILE_HEIGHT
        )

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
            outline=(100, 110, 125),
            width=3,
        )

        split_label_text = (
            f"{row['split'].upper()} | "
            f"{row['label']}"
        )

        if row["label"] == "PHISHING":
            label_colour = (190, 30, 45)
        else:
            label_colour = (20, 115, 70)

        drawing.text(
            (
                tile_left + 18,
                tile_top + 15,
            ),
            split_label_text,
            fill=label_colour,
            font=LABEL_FONT,
        )

        drawing.text(
            (
                tile_left + 18,
                tile_top + 45,
            ),
            shorten_text(
                row["sample_folder"],
                55,
            ),
            fill=(25, 25, 25),
            font=LABEL_FONT,
        )

        drawing.text(
            (
                tile_left + 18,
                tile_top + 73,
            ),
            (
                "Brand: "
                + shorten_text(
                    row.get("brand", ""),
                    55,
                )
            ),
            fill=(70, 70, 70),
            font=SMALL_FONT,
        )

        image_area_left = tile_left + 15
        image_area_top = (
            tile_top + LABEL_HEIGHT
        )

        image_area_width = TILE_WIDTH - 30
        image_area_height = (
            TILE_HEIGHT
            - LABEL_HEIGHT
            - 20
        )

        screenshot_path = Path(
            row["screenshot_path"]
        )

        try:
            with Image.open(
                screenshot_path
            ) as screenshot:

                screenshot = ImageOps.exif_transpose(
                    screenshot
                )

                screenshot = screenshot.convert("RGB")

                screenshot.thumbnail(
                    (
                        image_area_width,
                        image_area_height,
                    ),
                    RESAMPLE_METHOD,
                )

                paste_left = (
                    image_area_left
                    + (
                        image_area_width
                        - screenshot.width
                    )
                    // 2
                )

                paste_top = (
                    image_area_top
                    + (
                        image_area_height
                        - screenshot.height
                    )
                    // 2
                )

                canvas.paste(
                    screenshot,
                    (
                        paste_left,
                        paste_top,
                    ),
                )

        except Exception as error:
            drawing.text(
                (
                    image_area_left + 20,
                    image_area_top + 80,
                ),
                (
                    "IMAGE OPEN ERROR:\n"
                    f"{type(error).__name__}: {error}"
                ),
                fill=(190, 30, 45),
                font=LABEL_FONT,
            )

    canvas.save(
        output_path,
        format="JPEG",
        quality=92,
        optimize=True,
    )


def write_decisions_csv(
    groups: dict[str, list[dict[str, str]]],
    contact_sheet_names: dict[str, str],
) -> list[dict[str, str]]:
    """Create the manual-review decision file."""
    existing_decisions = load_existing_decisions()
    decision_rows: list[dict[str, str]] = []

    sorted_group_numbers = sorted(
        groups.keys(),
        key=lambda value: int(value),
    )

    for group_number in sorted_group_numbers:
        group_rows = groups[group_number]

        group_splits = sorted(
            {
                row["split"]
                for row in group_rows
            }
        )

        group_labels = sorted(
            {
                row["label"]
                for row in group_rows
            }
        )

        previous = existing_decisions.get(
            group_number,
            {},
        )

        if previous:
            decision = previous.get(
                "decision",
                "REVIEW_REQUIRED",
            )

            notes = previous.get(
                "notes",
                "",
            )

        elif group_number == "144":
            decision = "FALSE_MATCH_KEEP_ALL"

            notes = (
                "Different Caracol Television and "
                "Americanas login pages. Generic login "
                "layout caused the hash match."
            )

        else:
            decision = "REVIEW_REQUIRED"
            notes = ""

        decision_rows.append(
            {
                "near_duplicate_group": (
                    group_number
                ),
                "group_size": str(
                    len(group_rows)
                ),
                "group_splits": "|".join(
                    group_splits
                ),
                "group_labels": "|".join(
                    group_labels
                ),
                "contact_sheet": (
                    contact_sheet_names[
                        group_number
                    ]
                ),
                "decision": decision,
                "notes": notes,
            }
        )

    fieldnames = [
        "near_duplicate_group",
        "group_size",
        "group_splits",
        "group_labels",
        "contact_sheet",
        "decision",
        "notes",
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

    return decision_rows


def create_html_report(
    decision_rows: list[dict[str, str]],
) -> None:
    """Create an HTML gallery containing every contact sheet."""
    sections: list[str] = []

    for row in decision_rows:
        group_number = html.escape(
            row["near_duplicate_group"]
        )

        group_size = html.escape(
            row["group_size"]
        )

        group_labels = html.escape(
            row["group_labels"]
        )

        decision = html.escape(
            row["decision"]
        )

        notes = html.escape(
            row["notes"]
        )

        contact_sheet = html.escape(
            row["contact_sheet"]
        )

        sections.append(
            f"""
            <section class="group">
                <h2>Group {group_number}</h2>

                <p>
                    <strong>Images:</strong> {group_size}
                    &nbsp; | &nbsp;
                    <strong>Labels:</strong> {group_labels}
                    &nbsp; | &nbsp;
                    <strong>Current decision:</strong>
                    {decision}
                </p>

                <p><strong>Notes:</strong> {notes}</p>

                <img
                    src="{contact_sheet}"
                    alt="Near-duplicate group {group_number}"
                >
            </section>
            """
        )

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">

        <title>
            Phish360 Cross-Split Near-Duplicate Review
        </title>

        <style>
            body {{
                background: #eef1f5;
                color: #202735;
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 30px;
            }}

            .instructions {{
                background: white;
                border-left: 6px solid #2457a7;
                margin-bottom: 30px;
                padding: 20px 25px;
            }}

            .group {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px
                    rgba(0, 0, 0, 0.12);
                margin-bottom: 35px;
                padding: 20px;
            }}

            img {{
                border: 1px solid #aab1bd;
                display: block;
                height: auto;
                margin-top: 15px;
                max-width: 100%;
            }}

            h1, h2 {{
                color: #183c72;
            }}

            code {{
                background: #e7eaf0;
                padding: 3px 6px;
            }}
        </style>
    </head>

    <body>
        <div class="instructions">
            <h1>
                Phish360 Cross-Split Near-Duplicate Review
            </h1>

            <p>
                Review each group and record one of these
                decisions in
                <code>cross_split_review_decisions.csv</code>:
            </p>

            <ul>
                <li>
                    <strong>CONFIRMED_NEAR_DUPLICATE</strong>:
                    Same webpage or almost identical copies.
                </li>

                <li>
                    <strong>FALSE_MATCH_KEEP_ALL</strong>:
                    Different webpages that only have a
                    similar general layout.
                </li>

                <li>
                    <strong>REVIEW_REQUIRED</strong>:
                    The decision has not been made.
                </li>
            </ul>

            <p>
                Do not delete or modify original dataset images.
            </p>
        </div>

        {''.join(sections)}
    </body>
    </html>
    """

    HTML_REPORT.write_text(
        html_content,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 80)
    print("PHISH360 CROSS-SPLIT REVIEW SHEET CREATOR")
    print("=" * 80)

    print(
        "This script creates review images and reports only."
    )

    print(
        "It does not modify or delete original dataset files."
    )

    print()

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    groups = load_cross_split_groups()

    print(
        f"Cross-split groups found: {len(groups):,}"
    )

    if not groups:
        raise SystemExit(
            "No cross-split groups were found."
        )

    contact_sheet_names: dict[str, str] = {}

    sorted_group_numbers = sorted(
        groups.keys(),
        key=lambda value: int(value),
    )

    for current_number, group_number in enumerate(
        sorted_group_numbers,
        start=1,
    ):
        output_filename = (
            f"group_{int(group_number):04d}.jpg"
        )

        output_path = (
            OUTPUT_FOLDER / output_filename
        )

        create_contact_sheet(
            group_number,
            groups[group_number],
            output_path,
        )

        contact_sheet_names[
            group_number
        ] = output_filename

        print(
            f"Created {current_number:,}/"
            f"{len(groups):,}: "
            f"Group {group_number}"
        )

    decision_rows = write_decisions_csv(
        groups,
        contact_sheet_names,
    )

    create_html_report(decision_rows)

    print()
    print("=" * 80)
    print("REVIEW SHEETS CREATED")
    print("=" * 80)

    print(f"Groups: {len(groups):,}")
    print(f"Review folder: {OUTPUT_FOLDER}")
    print(f"HTML gallery: {HTML_REPORT}")
    print(f"Decision file: {DECISIONS_CSV}")

    print()
    print(
        "No original screenshots were modified."
    )


if __name__ == "__main__":
    main()