from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


# =============================================================================
# PATHS AND SETTINGS
# =============================================================================

DATASET_ROOT = Path(
    r"C:\Users\oshan\Desktop\Research project\datasets\dataset\login dataset"
)

PROJECT_ROOT = Path(
    r"C:\Users\oshan\Desktop\model 1 login pp2"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "01_dataset_audit"
    / "image_manifest.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "prepared_login_384x160"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "04_normalization"
)

CLASS_FOLDERS = {
    "MALICIOUS": DATASET_ROOT / "screenshots_MALICIOUS",
    "NOT_MALICIOUS": DATASET_ROOT / "screenshots_NOT_MALICIOUS",
}

# Common virtual viewport used before final resizing.
REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 802

# Remove a small, equal border from every screenshot.
# This reduces scrollbar and capture-edge shortcuts.
BORDER_CROP = 8

# Final MobileNetV3Large input size.
TARGET_WIDTH = 384
TARGET_HEIGHT = 160


# =============================================================================
# HELPERS
# =============================================================================

def sha256_file(file_path: Path) -> str:
    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)

    return hasher.hexdigest()


def estimate_background_color(
    image: Image.Image,
) -> tuple[int, int, int]:
    width, height = image.size

    corner_pixels = np.array(
        [
            image.getpixel((0, 0)),
            image.getpixel((width - 1, 0)),
            image.getpixel((0, height - 1)),
            image.getpixel((width - 1, height - 1)),
        ],
        dtype=np.float32,
    )

    median_color = np.median(
        corner_pixels,
        axis=0,
    ).astype(np.uint8)

    return tuple(int(value) for value in median_color)


def normalize_image(
    source_path: Path,
) -> tuple[
    Image.Image,
    int,
    int,
    int,
    int,
    str,
]:
    with Image.open(source_path) as source_image:
        source_image = ImageOps.exif_transpose(source_image)
        source_image = source_image.convert("RGB")

        original_width, original_height = source_image.size

        # Scale every screenshot to the same width while
        # preserving its original aspect ratio.
        scale = REFERENCE_WIDTH / original_width

        scaled_height = max(
            1,
            round(original_height * scale),
        )

        scaled_image = source_image.resize(
            (REFERENCE_WIDTH, scaled_height),
            Image.Resampling.LANCZOS,
        )

        # Create the common 1920x802 virtual viewport.
        if scaled_height > REFERENCE_HEIGHT:
            viewport_image = scaled_image.crop(
                (
                    0,
                    0,
                    REFERENCE_WIDTH,
                    REFERENCE_HEIGHT,
                )
            )

            viewport_action = "TOP_CROP"

        elif scaled_height < REFERENCE_HEIGHT:
            background_color = estimate_background_color(
                scaled_image
            )

            viewport_image = Image.new(
                "RGB",
                (
                    REFERENCE_WIDTH,
                    REFERENCE_HEIGHT,
                ),
                background_color,
            )

            viewport_image.paste(
                scaled_image,
                (0, 0),
            )

            viewport_action = "BOTTOM_PAD"

        else:
            viewport_image = scaled_image
            viewport_action = "NONE"

        # Apply the same small border crop to every image.
        viewport_image = viewport_image.crop(
            (
                BORDER_CROP,
                BORDER_CROP,
                REFERENCE_WIDTH - BORDER_CROP,
                REFERENCE_HEIGHT - BORDER_CROP,
            )
        )

        # Convert to the final model input resolution.
        final_image = viewport_image.resize(
            (
                TARGET_WIDTH,
                TARGET_HEIGHT,
            ),
            Image.Resampling.LANCZOS,
        )

    return (
        final_image,
        original_width,
        original_height,
        REFERENCE_WIDTH,
        scaled_height,
        viewport_action,
    )


def read_valid_manifest() -> list[dict[str, str]]:
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


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("MODEL 1 DATASET NORMALIZATION")
    print("=" * 80)
    print(f"Reference viewport : {REFERENCE_WIDTH}x{REFERENCE_HEIGHT}")
    print(f"Border crop        : {BORDER_CROP} pixels")
    print(f"Final image size   : {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest was not found:\n{MANIFEST_PATH}"
        )

    # Avoid silently overwriting an earlier prepared dataset.
    if OUTPUT_ROOT.exists():
        existing_images = list(
            OUTPUT_ROOT.rglob("*.png")
        )

        if existing_images:
            raise FileExistsError(
                "The prepared dataset already contains images:\n"
                f"{OUTPUT_ROOT}\n\n"
                "Nothing was overwritten."
            )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_valid_manifest()

    normalization_rows: list[dict[str, object]] = []
    action_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()

    hashes_to_rows: dict[
        str,
        list[dict[str, object]],
    ] = defaultdict(list)

    total_images = len(rows)

    for index, row in enumerate(rows, start=1):
        class_label = row["class_label"]
        source_path = DATASET_ROOT / row["relative_path"]

        class_source_root = CLASS_FOLDERS[class_label]

        relative_inside_class = source_path.relative_to(
            class_source_root
        )

        output_path = (
            OUTPUT_ROOT
            / class_label
            / relative_inside_class
        ).with_suffix(".png")

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            normalized_image,
            original_width,
            original_height,
            scaled_width,
            scaled_height,
            viewport_action,
        ) = normalize_image(source_path)

        normalized_image.save(
            output_path,
            format="PNG",
            optimize=True,
            compress_level=6,
        )

        # Reopen the output to verify its final dimensions.
        with Image.open(output_path) as verification_image:
            final_width, final_height = (
                verification_image.size
            )

        if (
            final_width != TARGET_WIDTH
            or final_height != TARGET_HEIGHT
        ):
            raise ValueError(
                "Unexpected output size for:\n"
                f"{output_path}\n"
                f"Received: {final_width}x{final_height}"
            )

        output_hash = sha256_file(output_path)

        normalization_row: dict[str, object] = {
            "class_label": class_label,
            "original_relative_path": row["relative_path"],
            "original_width": original_width,
            "original_height": original_height,
            "scaled_width": scaled_width,
            "scaled_height": scaled_height,
            "viewport_action": viewport_action,
            "border_crop": BORDER_CROP,
            "output_relative_path": str(
                output_path.relative_to(OUTPUT_ROOT)
            ),
            "final_width": final_width,
            "final_height": final_height,
            "output_sha256": output_hash,
        }

        normalization_rows.append(normalization_row)
        hashes_to_rows[output_hash].append(
            normalization_row
        )

        action_counts[viewport_action] += 1
        class_counts[class_label] += 1

        if index % 250 == 0 or index == total_images:
            print(
                f"Normalized {index}/{total_images} images"
            )

    # =========================================================================
    # NORMALIZED MANIFEST
    # =========================================================================

    manifest_columns = [
        "class_label",
        "original_relative_path",
        "original_width",
        "original_height",
        "scaled_width",
        "scaled_height",
        "viewport_action",
        "border_crop",
        "output_relative_path",
        "final_width",
        "final_height",
        "output_sha256",
    ]

    normalized_manifest_path = (
        REPORT_DIR / "normalized_manifest.csv"
    )

    with normalized_manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=manifest_columns,
        )

        writer.writeheader()
        writer.writerows(normalization_rows)

    # =========================================================================
    # CHECK WHETHER NORMALIZATION CREATED EXACT DUPLICATES
    # =========================================================================

    duplicate_groups = [
        group
        for group in hashes_to_rows.values()
        if len(group) > 1
    ]

    duplicate_columns = [
        "duplicate_group_id",
        "group_size",
        "cross_class",
        "class_label",
        "output_relative_path",
        "output_sha256",
    ]

    duplicate_output_rows: list[dict[str, object]] = []
    cross_class_duplicate_groups = 0

    for group_number, group in enumerate(
        duplicate_groups,
        start=1,
    ):
        group_classes = {
            str(item["class_label"])
            for item in group
        }

        is_cross_class = len(group_classes) > 1

        if is_cross_class:
            cross_class_duplicate_groups += 1

        for item in group:
            duplicate_output_rows.append(
                {
                    "duplicate_group_id": (
                        f"NORMALIZED_DUP_{group_number:04d}"
                    ),
                    "group_size": len(group),
                    "cross_class": is_cross_class,
                    "class_label": item["class_label"],
                    "output_relative_path": (
                        item["output_relative_path"]
                    ),
                    "output_sha256": (
                        item["output_sha256"]
                    ),
                }
            )

    duplicate_report_path = (
        REPORT_DIR
        / "normalized_exact_duplicate_groups.csv"
    )

    with duplicate_report_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=duplicate_columns,
        )

        writer.writeheader()
        writer.writerows(duplicate_output_rows)

    summary_lines = [
        "=" * 80,
        "MODEL 1 NORMALIZATION SUMMARY",
        "=" * 80,
        "",
        f"Original images                : {total_images}",
        f"Final output size               : {TARGET_WIDTH}x{TARGET_HEIGHT}",
        f"Prepared dataset                : {OUTPUT_ROOT}",
        "",
        "CLASS COUNTS",
        "-" * 80,
        (
            f"MALICIOUS                      : "
            f"{class_counts['MALICIOUS']}"
        ),
        (
            f"NOT_MALICIOUS                  : "
            f"{class_counts['NOT_MALICIOUS']}"
        ),
        "",
        "VIEWPORT ACTIONS",
        "-" * 80,
        (
            f"Top cropped                    : "
            f"{action_counts['TOP_CROP']}"
        ),
        (
            f"Bottom padded                  : "
            f"{action_counts['BOTTOM_PAD']}"
        ),
        (
            f"No viewport change             : "
            f"{action_counts['NONE']}"
        ),
        "",
        "POST-NORMALIZATION DUPLICATES",
        "-" * 80,
        (
            f"Exact duplicate groups         : "
            f"{len(duplicate_groups)}"
        ),
        (
            f"Cross-class duplicate groups   : "
            f"{cross_class_duplicate_groups}"
        ),
        "",
        "IMPORTANT:",
        "All output images are newly created normalized copies.",
        "No original screenshot was modified, moved, renamed,",
        "resized, or deleted.",
        "=" * 80,
    ]

    summary_text = "\n".join(summary_lines)

    summary_path = (
        REPORT_DIR / "normalization_summary.txt"
    )

    summary_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    print()
    print(summary_text)
    print()
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()