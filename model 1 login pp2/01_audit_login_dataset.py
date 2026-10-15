from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


# =============================================================================
# PATHS
# =============================================================================

DATASET_ROOT = Path(
    r"C:\Users\oshan\Desktop\Research project\datasets\dataset\login dataset"
)

PROJECT_ROOT = Path(
    r"C:\Users\oshan\Desktop\model 1 login pp2"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "01_dataset_audit"

CLASS_FOLDERS = {
    "MALICIOUS": DATASET_ROOT / "screenshots_MALICIOUS",
    "NOT_MALICIOUS": DATASET_ROOT / "screenshots_NOT_MALICIOUS",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}

MANIFEST_COLUMNS = [
    "class_label",
    "relative_path",
    "filename",
    "extension",
    "file_size_bytes",
    "width",
    "height",
    "aspect_ratio",
    "image_mode",
    "image_format",
    "mean_brightness",
    "pixel_std",
    "entropy",
    "sha256",
    "status",
    "review_reasons",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sha256_file(file_path: Path) -> str:
    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)

    return hasher.hexdigest()


def inspect_image(file_path: Path, class_label: str) -> dict[str, object]:
    row: dict[str, object] = {
        "class_label": class_label,
        "relative_path": str(file_path.relative_to(DATASET_ROOT)),
        "filename": file_path.name,
        "extension": file_path.suffix.lower(),
        "file_size_bytes": file_path.stat().st_size,
        "width": "",
        "height": "",
        "aspect_ratio": "",
        "image_mode": "",
        "image_format": "",
        "mean_brightness": "",
        "pixel_std": "",
        "entropy": "",
        "sha256": "",
        "status": "VALID",
        "review_reasons": "",
    }

    review_reasons: list[str] = []

    try:
        row["sha256"] = sha256_file(file_path)

        # First pass checks whether Pillow can verify the file.
        with Image.open(file_path) as image:
            image.verify()

        # Reopen after verify() to inspect the pixels.
        with Image.open(file_path) as image:
            image_format = image.format or "UNKNOWN"
            image = ImageOps.exif_transpose(image)
            image.load()

            width, height = image.size
            image_mode = image.mode

            row["width"] = width
            row["height"] = height
            row["aspect_ratio"] = round(width / height, 4) if height else ""
            row["image_mode"] = image_mode
            row["image_format"] = image_format

            # Use a thumbnail to calculate quick visual statistics.
            sample = image.convert("RGB")
            sample.thumbnail((256, 256), Image.Resampling.LANCZOS)

            pixel_array = np.asarray(sample, dtype=np.float32)

            mean_brightness = float(pixel_array.mean())
            pixel_std = float(pixel_array.std())

            grayscale = sample.convert("L")
            entropy = float(grayscale.entropy())

            row["mean_brightness"] = round(mean_brightness, 4)
            row["pixel_std"] = round(pixel_std, 4)
            row["entropy"] = round(entropy, 4)

            # These images are not automatically removed.
            # They are only marked for later visual review.
            if width < 300 or height < 150:
                review_reasons.append("SMALL_DIMENSIONS")

            if height > 0:
                aspect_ratio = width / height

                if aspect_ratio < 0.30 or aspect_ratio > 5.00:
                    review_reasons.append("UNUSUAL_ASPECT_RATIO")

            if file_path.stat().st_size < 5_000:
                review_reasons.append("VERY_SMALL_FILE")

            if pixel_std < 5.0:
                review_reasons.append("LOW_PIXEL_VARIATION")

            if entropy < 1.0:
                review_reasons.append("LOW_ENTROPY")

            if mean_brightness < 5.0:
                review_reasons.append("MOSTLY_BLACK")

            if mean_brightness > 250.0:
                review_reasons.append("MOSTLY_WHITE")

    except Exception as error:
        row["status"] = "INVALID"
        review_reasons.append(
            f"{type(error).__name__}: {str(error)}"
        )

    row["review_reasons"] = "; ".join(review_reasons)

    return row


def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# MAIN AUDIT
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("MODEL 1 LOGIN DATASET AUDIT")
    print("=" * 80)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folder was not found:\n{DATASET_ROOT}"
        )

    for class_label, class_folder in CLASS_FOLDERS.items():
        if not class_folder.exists():
            raise FileNotFoundError(
                f"Class folder was not found:\n{class_folder}"
            )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    image_entries: list[tuple[Path, str]] = []

    for class_label, class_folder in CLASS_FOLDERS.items():
        class_images = sorted(
            file_path
            for file_path in class_folder.rglob("*")
            if file_path.is_file()
            and file_path.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"{class_label:<20}: {len(class_images)} images")

        image_entries.extend(
            (file_path, class_label)
            for file_path in class_images
        )

    print("-" * 80)
    print(f"Total images          : {len(image_entries)}")
    print()
    print("Inspecting images...")

    manifest_rows: list[dict[str, object]] = []

    for index, (file_path, class_label) in enumerate(
        image_entries,
        start=1,
    ):
        manifest_rows.append(
            inspect_image(file_path, class_label)
        )

        if index % 250 == 0 or index == len(image_entries):
            print(
                f"Processed {index}/{len(image_entries)} images"
            )

    valid_rows = [
        row
        for row in manifest_rows
        if row["status"] == "VALID"
    ]

    invalid_rows = [
        row
        for row in manifest_rows
        if row["status"] == "INVALID"
    ]

    review_rows = [
        row
        for row in manifest_rows
        if row["status"] == "VALID"
        and row["review_reasons"]
    ]

    # =========================================================================
    # EXACT-DUPLICATE DETECTION
    # =========================================================================

    rows_by_hash: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in valid_rows:
        file_hash = str(row["sha256"])

        if file_hash:
            rows_by_hash[file_hash].append(row)

    duplicate_hash_groups = [
        duplicate_rows
        for duplicate_rows in rows_by_hash.values()
        if len(duplicate_rows) > 1
    ]

    duplicate_hash_groups.sort(
        key=lambda group: (
            -len(group),
            str(group[0]["sha256"]),
        )
    )

    duplicate_rows_output: list[dict[str, object]] = []
    cross_class_duplicate_rows: list[dict[str, object]] = []

    duplicate_extra_images = 0
    cross_class_group_count = 0

    duplicate_columns = [
        "duplicate_group_id",
        "group_size",
        "cross_class",
        "class_label",
        "relative_path",
        "sha256",
    ]

    for group_number, duplicate_group in enumerate(
        duplicate_hash_groups,
        start=1,
    ):
        group_id = f"EXACT_DUP_{group_number:04d}"
        group_classes = {
            str(row["class_label"])
            for row in duplicate_group
        }

        is_cross_class = len(group_classes) > 1

        duplicate_extra_images += len(duplicate_group) - 1

        if is_cross_class:
            cross_class_group_count += 1

        for row in duplicate_group:
            output_row = {
                "duplicate_group_id": group_id,
                "group_size": len(duplicate_group),
                "cross_class": is_cross_class,
                "class_label": row["class_label"],
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
            }

            duplicate_rows_output.append(output_row)

            if is_cross_class:
                cross_class_duplicate_rows.append(output_row)

    # =========================================================================
    # REPORT FILES
    # =========================================================================

    write_csv(
        REPORT_DIR / "image_manifest.csv",
        manifest_rows,
        MANIFEST_COLUMNS,
    )

    write_csv(
        REPORT_DIR / "invalid_images.csv",
        invalid_rows,
        MANIFEST_COLUMNS,
    )

    write_csv(
        REPORT_DIR / "review_required_images.csv",
        review_rows,
        MANIFEST_COLUMNS,
    )

    write_csv(
        REPORT_DIR / "exact_duplicate_groups.csv",
        duplicate_rows_output,
        duplicate_columns,
    )

    write_csv(
        REPORT_DIR / "cross_class_exact_duplicates.csv",
        cross_class_duplicate_rows,
        duplicate_columns,
    )

    class_counts = Counter(
        str(row["class_label"])
        for row in manifest_rows
    )

    valid_class_counts = Counter(
        str(row["class_label"])
        for row in valid_rows
    )

    extension_counts = Counter(
        str(row["extension"])
        for row in manifest_rows
    )

    dimension_counts = Counter(
        f"{row['width']}x{row['height']}"
        for row in valid_rows
    )

    review_reason_counts: Counter[str] = Counter()

    for row in review_rows:
        reasons = str(row["review_reasons"]).split("; ")

        for reason in reasons:
            if reason:
                review_reason_counts[reason] += 1

    summary_lines: list[str] = [
        "=" * 80,
        "MODEL 1 LOGIN DATASET AUDIT SUMMARY",
        "=" * 80,
        "",
        f"Dataset path                    : {DATASET_ROOT}",
        f"Total discovered images         : {len(manifest_rows)}",
        f"Valid images                    : {len(valid_rows)}",
        f"Invalid/corrupt images           : {len(invalid_rows)}",
        f"Images requiring visual review  : {len(review_rows)}",
        "",
        "CLASS COUNTS",
        "-" * 80,
    ]

    for class_label in CLASS_FOLDERS:
        summary_lines.append(
            f"{class_label:<30}: "
            f"{class_counts[class_label]} total, "
            f"{valid_class_counts[class_label]} valid"
        )

    summary_lines.extend(
        [
            "",
            "EXACT DUPLICATES",
            "-" * 80,
            f"Exact duplicate groups          : {len(duplicate_hash_groups)}",
            f"Extra duplicate images          : {duplicate_extra_images}",
            f"Cross-class duplicate groups    : {cross_class_group_count}",
            "",
            "FILE EXTENSIONS",
            "-" * 80,
        ]
    )

    for extension, count in sorted(extension_counts.items()):
        summary_lines.append(
            f"{extension:<30}: {count}"
        )

    summary_lines.extend(
        [
            "",
            "REVIEW REASONS",
            "-" * 80,
        ]
    )

    if review_reason_counts:
        for reason, count in review_reason_counts.most_common():
            summary_lines.append(
                f"{reason:<30}: {count}"
            )
    else:
        summary_lines.append("No images were marked for review.")

    summary_lines.extend(
        [
            "",
            "MOST COMMON IMAGE DIMENSIONS",
            "-" * 80,
        ]
    )

    for dimensions, count in dimension_counts.most_common(20):
        summary_lines.append(
            f"{dimensions:<30}: {count}"
        )

    summary_lines.extend(
        [
            "",
            "=" * 80,
            "IMPORTANT",
            "=" * 80,
            "This audit did not delete, rename, resize, move, or modify",
            "any original screenshot.",
            "",
            "Near-duplicate detection is not included in this first audit.",
            "It will be performed after reviewing this report.",
            "=" * 80,
        ]
    )

    summary_text = "\n".join(summary_lines)

    summary_path = REPORT_DIR / "audit_summary.txt"
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