from __future__ import annotations

import csv
import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(
    r"C:\Users\oshan\Desktop\new model 2 pp2"
)

REPORTS_FOLDER = PROJECT_ROOT / "reports"

PAIRS_CSV = (
    REPORTS_FOLDER
    / "phish360_near_duplicate_pairs.csv"
)

OUTPUT_FOLDER = (
    REPORTS_FOLDER
    / "cross_split_pair_review"
)

DECISIONS_CSV = (
    OUTPUT_FOLDER
    / "cross_split_pair_decisions.csv"
)

HTML_REPORT = (
    OUTPUT_FOLDER
    / "cross_split_pair_review.html"
)

TILE_WIDTH = 620
TILE_HEIGHT = 530
HEADER_HEIGHT = 100

try:
    RESAMPLE_METHOD = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_METHOD = Image.LANCZOS


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
                    size,
                )
            except OSError:
                pass

    return ImageFont.load_default()


TITLE_FONT = load_font(26)
LABEL_FONT = load_font(20)
SMALL_FONT = load_font(16)


def is_true(value: str) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def shorten(text: str, length: int = 55) -> str:
    text = str(text)

    if len(text) <= length:
        return text

    return text[: length - 3] + "..."


def load_cross_split_pairs() -> list[dict[str, str]]:
    if not PAIRS_CSV.is_file():
        raise SystemExit(
            "\nERROR: Pair report was not found:\n"
            f"{PAIRS_CSV}"
        )

    pairs: list[dict[str, str]] = []

    with PAIRS_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if is_true(row.get("cross_split", "")):
                pairs.append(row)

    pairs.sort(
        key=lambda row: int(row["pair_number"])
    )

    return pairs


def load_existing_decisions() -> dict[str, dict[str, str]]:
    existing: dict[str, dict[str, str]] = {}

    if not DECISIONS_CSV.is_file():
        return existing

    with DECISIONS_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            pair_number = row.get("pair_number", "")

            if pair_number:
                existing[pair_number] = row

    return existing


def paste_screenshot(
    canvas: Image.Image,
    drawing: ImageDraw.ImageDraw,
    image_path: Path,
    left: int,
    top: int,
    split_name: str,
    label: str,
    sample_name: str,
) -> None:

    tile_right = left + TILE_WIDTH
    tile_bottom = top + TILE_HEIGHT

    drawing.rectangle(
        (
            left + 5,
            top + 5,
            tile_right - 5,
            tile_bottom - 5,
        ),
        fill=(255, 255, 255),
        outline=(105, 115, 130),
        width=3,
    )

    if label == "PHISHING":
        label_colour = (190, 30, 45)
    else:
        label_colour = (20, 115, 70)

    drawing.text(
        (left + 20, top + 18),
        f"{split_name.upper()} | {label}",
        fill=label_colour,
        font=LABEL_FONT,
    )

    drawing.text(
        (left + 20, top + 50),
        shorten(sample_name),
        fill=(25, 25, 25),
        font=LABEL_FONT,
    )

    image_area_width = TILE_WIDTH - 40
    image_area_height = TILE_HEIGHT - 115

    try:
        with Image.open(image_path) as screenshot:
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
                left
                + 20
                + (
                    image_area_width
                    - screenshot.width
                )
                // 2
            )

            paste_top = (
                top
                + 90
                + (
                    image_area_height
                    - screenshot.height
                )
                // 2
            )

            canvas.paste(
                screenshot,
                (paste_left, paste_top),
            )

    except Exception as error:
        drawing.text(
            (left + 25, top + 160),
            (
                "IMAGE ERROR:\n"
                f"{type(error).__name__}: {error}"
            ),
            fill=(190, 30, 45),
            font=LABEL_FONT,
        )


def create_pair_sheet(
    pair: dict[str, str],
    output_path: Path,
) -> None:

    canvas_width = TILE_WIDTH * 2
    canvas_height = HEADER_HEIGHT + TILE_HEIGHT

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        color=(235, 238, 242),
    )

    drawing = ImageDraw.Draw(canvas)

    drawing.rectangle(
        (
            0,
            0,
            canvas_width,
            HEADER_HEIGHT,
        ),
        fill=(25, 45, 75),
    )

    title = (
        f"Pair {pair['pair_number']} | "
        f"pHash distance: {pair['phash_distance']} | "
        f"dHash distance: {pair['dhash_distance']}"
    )

    drawing.text(
        (25, 20),
        title,
        fill=(255, 255, 255),
        font=TITLE_FONT,
    )

    instruction = (
        "Decide whether these two screenshots are genuine "
        "near-duplicates or a false match."
    )

    drawing.text(
        (25, 60),
        instruction,
        fill=(220, 230, 245),
        font=SMALL_FONT,
    )

    paste_screenshot(
        canvas=canvas,
        drawing=drawing,
        image_path=Path(pair["first_path"]),
        left=0,
        top=HEADER_HEIGHT,
        split_name=pair["first_split"],
        label=pair["first_label"],
        sample_name=pair["first_sample"],
    )

    paste_screenshot(
        canvas=canvas,
        drawing=drawing,
        image_path=Path(pair["second_path"]),
        left=TILE_WIDTH,
        top=HEADER_HEIGHT,
        split_name=pair["second_split"],
        label=pair["second_label"],
        sample_name=pair["second_sample"],
    )

    canvas.save(
        output_path,
        format="JPEG",
        quality=92,
        optimize=True,
    )


def default_decision(
    pair: dict[str, str],
) -> tuple[str, str]:

    sample_names = {
        pair["first_sample"],
        pair["second_sample"],
    }

    known_false_pair = {
        "L06075_legitimate",
        "P02289_phish",
    }

    if sample_names == known_false_pair:
        return (
            "FALSE_MATCH_KEEP_BOTH",
            (
                "Different Caracol Television and "
                "Americanas login pages."
            ),
        )

    return "REVIEW_REQUIRED", ""


def create_decisions_file(
    pairs: list[dict[str, str]],
    sheet_names: dict[str, str],
) -> list[dict[str, str]]:

    existing = load_existing_decisions()
    decision_rows: list[dict[str, str]] = []

    for review_number, pair in enumerate(
        pairs,
        start=1,
    ):
        pair_number = pair["pair_number"]

        if pair_number in existing:
            decision = existing[pair_number].get(
                "decision",
                "REVIEW_REQUIRED",
            )

            notes = existing[pair_number].get(
                "notes",
                "",
            )

        else:
            decision, notes = default_decision(pair)

        decision_rows.append(
            {
                "review_number": str(review_number),
                "pair_number": pair_number,
                "first_split": pair["first_split"],
                "first_label": pair["first_label"],
                "first_sample": pair["first_sample"],
                "second_split": pair["second_split"],
                "second_label": pair["second_label"],
                "second_sample": pair["second_sample"],
                "phash_distance": pair[
                    "phash_distance"
                ],
                "dhash_distance": pair[
                    "dhash_distance"
                ],
                "contact_sheet": sheet_names[
                    pair_number
                ],
                "decision": decision,
                "notes": notes,
            }
        )

    fieldnames = [
        "review_number",
        "pair_number",
        "first_split",
        "first_label",
        "first_sample",
        "second_split",
        "second_label",
        "second_sample",
        "phash_distance",
        "dhash_distance",
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


def create_html(
    decision_rows: list[dict[str, str]],
) -> None:

    sections: list[str] = []

    for row in decision_rows:
        sections.append(
            f"""
            <section>
                <h2>
                    Review {html.escape(row['review_number'])}
                    — Pair {html.escape(row['pair_number'])}
                </h2>

                <p>
                    <strong>Current decision:</strong>
                    {html.escape(row['decision'])}
                </p>

                <img
                    src="{html.escape(row['contact_sheet'])}"
                    alt="Pair review"
                >
            </section>
            """
        )

    content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">

        <title>Cross-Split Pair Review</title>

        <style>
            body {{
                background: #eef1f5;
                color: #202735;
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 30px;
            }}

            .instructions,
            section {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px
                    rgba(0, 0, 0, 0.12);
                margin-bottom: 30px;
                padding: 20px;
            }}

            .instructions {{
                border-left: 6px solid #2457a7;
            }}

            img {{
                border: 1px solid #aab1bd;
                height: auto;
                max-width: 100%;
            }}

            h1, h2 {{
                color: #183c72;
            }}

            code {{
                background: #e6e9ef;
                padding: 3px 6px;
            }}
        </style>
    </head>

    <body>
        <div class="instructions">
            <h1>Cross-Split Pair Review</h1>

            <p>
                Make one decision for each pair:
            </p>

            <ul>
                <li>
                    <code>CONFIRMED_NEAR_DUPLICATE</code> —
                    same webpage or almost identical copies.
                </li>

                <li>
                    <code>FALSE_MATCH_KEEP_BOTH</code> —
                    different webpages with only a similar
                    general layout.
                </li>
            </ul>
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
    print("PHISH360 CROSS-SPLIT PAIR REVIEW CREATOR")
    print("=" * 80)

    print(
        "This script creates pair-review reports only."
    )

    print(
        "It does not modify or delete dataset images."
    )

    print()

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs = load_cross_split_pairs()

    print(f"Cross-split pairs found: {len(pairs):,}")

    sheet_names: dict[str, str] = {}

    for review_number, pair in enumerate(
        pairs,
        start=1,
    ):
        pair_number = pair["pair_number"]

        filename = (
            f"review_{review_number:03d}"
            f"_pair_{int(pair_number):04d}.jpg"
        )

        create_pair_sheet(
            pair,
            OUTPUT_FOLDER / filename,
        )

        sheet_names[pair_number] = filename

        print(
            f"Created {review_number:,}/"
            f"{len(pairs):,}: "
            f"Pair {pair_number}"
        )

    decisions = create_decisions_file(
        pairs,
        sheet_names,
    )

    create_html(decisions)

    print()
    print("=" * 80)
    print("PAIR REVIEW CREATED")
    print("=" * 80)
    print(f"Pairs: {len(pairs):,}")
    print(f"HTML report: {HTML_REPORT}")
    print(f"Decision CSV: {DECISIONS_CSV}")
    print()
    print("No original dataset files were modified.")


if __name__ == "__main__":
    main()