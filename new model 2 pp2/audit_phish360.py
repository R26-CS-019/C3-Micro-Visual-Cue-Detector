from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageStat, UnidentifiedImageError


# ============================================================================
# PHISH360 DATASET PATH
# ============================================================================

DATASET_ROOT = Path(
    r"C:\Users\oshan\Desktop\Research project\datasets\dataset\phish360\Phish360"
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}

LOW_VARIANCE_THRESHOLD = 5.0


def normalize_sample_name(name: str) -> str:
    """Remove spaces, underscores and hyphens before the sample ID."""
    return name.lstrip("_ -").strip()


def infer_label(sample_name: str) -> str:
    """
    Determine the sample label using the folder name.

    P = Phishing
    L = Legitimate
    """
    normalized_name = normalize_sample_name(sample_name).upper()

    if normalized_name.startswith("P"):
        return "PHISHING"

    if normalized_name.startswith("L"):
        return "LEGITIMATE"

    return "UNKNOWN"


def infer_brand(sample_name: str, label: str) -> str:
    """
    Extract the target brand from a phishing folder name.

    Example:
    P16026_facebook -> facebook
    """
    normalized_name = normalize_sample_name(sample_name)
    parts = normalized_name.split("_", 1)

    if label == "PHISHING" and len(parts) == 2:
        brand = parts[1].strip()

        if brand:
            return brand

    if label == "LEGITIMATE":
        return "LEGITIMATE"

    return "UNKNOWN"


def find_screenshot(sample_folder: Path) -> Path | None:
    """Find the screenshot belonging to one Phish360 sample."""
    screenshot_folder = sample_folder / "SCREEN-SHOT"

    expected_screenshot = screenshot_folder / "screen_shoot.png"

    if expected_screenshot.is_file():
        return expected_screenshot

    if not screenshot_folder.is_dir():
        return None

    alternative_images = sorted(
        file_path
        for file_path in screenshot_folder.rglob("*")
        if (
            file_path.is_file()
            and file_path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )

    if alternative_images:
        return alternative_images[0]

    return None


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 hash of an image."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def inspect_image(image_path: Path) -> dict[str, object]:
    """
    Check whether an image is readable and collect image information.
    """
    result: dict[str, object] = {
        "status": "OK",
        "error": "",
        "width": "",
        "height": "",
        "mode": "",
        "format": "",
        "file_size_bytes": "",
        "gray_mean": "",
        "gray_stddev": "",
        "low_variance_flag": False,
        "sha256": "",
    }

    try:
        result["file_size_bytes"] = image_path.stat().st_size
        result["sha256"] = calculate_sha256(image_path)

        # Verify that the image file is not damaged.
        with Image.open(image_path) as image:
            image.verify()

        # Reopen the image after verify().
        with Image.open(image_path) as image:
            image.load()

            width, height = image.size

            result["width"] = width
            result["height"] = height
            result["mode"] = image.mode
            result["format"] = (
                image.format
                or image_path.suffix.lstrip(".").upper()
            )

            # Create a small grayscale preview for blank-page checking.
            preview = image.convert("L")
            preview.thumbnail((128, 128))

            statistics = ImageStat.Stat(preview)

            gray_mean = float(statistics.mean[0])
            gray_stddev = float(statistics.stddev[0])

            result["gray_mean"] = round(gray_mean, 4)
            result["gray_stddev"] = round(gray_stddev, 4)

            result["low_variance_flag"] = (
                gray_stddev < LOW_VARIANCE_THRESHOLD
            )

    except (UnidentifiedImageError, OSError, ValueError) as error:
        result["status"] = "CORRUPT"
        result["error"] = f"{type(error).__name__}: {error}"

    return result


def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    """Write report rows to a CSV file."""
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("=" * 80)
    print("PHISH360 DATASET AUDIT")
    print("=" * 80)

    print(f"Dataset path: {DATASET_ROOT}")
    print()

    print(
        "This script only reads the dataset. "
        "It does not delete, move, rename or modify dataset files."
    )
    print()

    trainval_folder = DATASET_ROOT / "trainval"
    test_folder = DATASET_ROOT / "test"

    if not DATASET_ROOT.is_dir():
        raise SystemExit(
            "\nERROR: The Phish360 dataset folder was not found.\n"
            f"Checked path:\n{DATASET_ROOT}"
        )

    if not trainval_folder.is_dir():
        raise SystemExit(
            "\nERROR: The 'trainval' folder was not found.\n"
            f"Checked path:\n{trainval_folder}"
        )

    if not test_folder.is_dir():
        raise SystemExit(
            "\nERROR: The 'test' folder was not found.\n"
            f"Checked path:\n{test_folder}"
        )

    # Reports are created beside this Python script.
    script_folder = Path(__file__).resolve().parent
    reports_folder = script_folder / "reports"
    reports_folder.mkdir(parents=True, exist_ok=True)

    all_sample_folders: list[tuple[str, Path]] = []

    for split_name, split_folder in (
        ("trainval", trainval_folder),
        ("test", test_folder),
    ):
        sample_folders = sorted(
            folder
            for folder in split_folder.iterdir()
            if folder.is_dir()
        )

        print(
            f"Found {len(sample_folders):,} "
            f"sample folders in {split_name}."
        )

        for sample_folder in sample_folders:
            all_sample_folders.append(
                (split_name, sample_folder)
            )

    total_samples = len(all_sample_folders)

    print(f"Total sample folders: {total_samples:,}")
    print()
    print("Starting screenshot inspection...")
    print()

    audit_rows: list[dict[str, object]] = []

    for sample_number, (
        split_name,
        sample_folder,
    ) in enumerate(all_sample_folders, start=1):

        label = infer_label(sample_folder.name)
        brand = infer_brand(sample_folder.name, label)
        screenshot_path = find_screenshot(sample_folder)

        row: dict[str, object] = {
            "split": split_name,
            "sample_folder": sample_folder.name,
            "label": label,
            "brand": brand,
            "screenshot_path": "",
            "status": "",
            "error": "",
            "width": "",
            "height": "",
            "mode": "",
            "format": "",
            "file_size_bytes": "",
            "gray_mean": "",
            "gray_stddev": "",
            "low_variance_flag": False,
            "sha256": "",
        }

        if screenshot_path is None:
            row["status"] = "MISSING_SCREENSHOT"
        else:
            row["screenshot_path"] = str(screenshot_path)
            row.update(inspect_image(screenshot_path))

        audit_rows.append(row)

        if (
            sample_number % 500 == 0
            or sample_number == total_samples
        ):
            print(
                f"Processed "
                f"{sample_number:,}/{total_samples:,} samples"
            )

    # ========================================================================
    # SAVE MAIN AUDIT REPORT
    # ========================================================================

    audit_fieldnames = [
        "split",
        "sample_folder",
        "label",
        "brand",
        "screenshot_path",
        "status",
        "error",
        "width",
        "height",
        "mode",
        "format",
        "file_size_bytes",
        "gray_mean",
        "gray_stddev",
        "low_variance_flag",
        "sha256",
    ]

    audit_report_path = (
        reports_folder / "phish360_dataset_audit.csv"
    )

    write_csv(
        audit_report_path,
        audit_rows,
        audit_fieldnames,
    )

    # ========================================================================
    # FIND EXACT DUPLICATES
    # ========================================================================

    valid_rows = [
        row
        for row in audit_rows
        if row["status"] == "OK" and row["sha256"]
    ]

    hash_groups: dict[
        str,
        list[dict[str, object]],
    ] = defaultdict(list)

    for row in valid_rows:
        image_hash = str(row["sha256"])
        hash_groups[image_hash].append(row)

    duplicate_groups = [
        group
        for group in hash_groups.values()
        if len(group) > 1
    ]

    duplicate_rows: list[dict[str, object]] = []

    for group_number, group in enumerate(
        duplicate_groups,
        start=1,
    ):
        group_splits = sorted(
            {
                str(row["split"])
                for row in group
            }
        )

        group_labels = sorted(
            {
                str(row["label"])
                for row in group
            }
        )

        for row in group:
            duplicate_rows.append(
                {
                    "duplicate_group": group_number,
                    "sha256": row["sha256"],
                    "group_size": len(group),
                    "group_splits": "|".join(group_splits),
                    "group_labels": "|".join(group_labels),
                    "split": row["split"],
                    "label": row["label"],
                    "sample_folder": row["sample_folder"],
                    "screenshot_path": row["screenshot_path"],
                }
            )

    duplicate_fieldnames = [
        "duplicate_group",
        "sha256",
        "group_size",
        "group_splits",
        "group_labels",
        "split",
        "label",
        "sample_folder",
        "screenshot_path",
    ]

    duplicate_report_path = (
        reports_folder
        / "phish360_exact_duplicate_groups.csv"
    )

    write_csv(
        duplicate_report_path,
        duplicate_rows,
        duplicate_fieldnames,
    )

    # ========================================================================
    # CALCULATE SUMMARY RESULTS
    # ========================================================================

    split_label_counts = Counter(
        (
            str(row["split"]),
            str(row["label"]),
        )
        for row in audit_rows
    )

    status_counts = Counter(
        str(row["status"])
        for row in audit_rows
    )

    dimension_counts = Counter(
        (
            int(row["width"]),
            int(row["height"]),
        )
        for row in valid_rows
        if (
            row["width"] != ""
            and row["height"] != ""
        )
    )

    low_variance_count = sum(
        bool(row["low_variance_flag"])
        for row in valid_rows
    )

    duplicate_images_count = sum(
        len(group)
        for group in duplicate_groups
    )

    extra_duplicate_copies = sum(
        len(group) - 1
        for group in duplicate_groups
    )

    cross_split_duplicate_groups = sum(
        1
        for group in duplicate_groups
        if len(
            {
                str(row["split"])
                for row in group
            }
        ) > 1
    )

    cross_label_duplicate_groups = sum(
        1
        for group in duplicate_groups
        if len(
            {
                str(row["label"])
                for row in group
            }
        ) > 1
    )

    summary_lines = [
        "=" * 80,
        "PHISH360 DATASET AUDIT SUMMARY",
        "=" * 80,
        (
            "Audit time: "
            + datetime.now().isoformat(
                timespec="seconds"
            )
        ),
        f"Dataset root: {DATASET_ROOT}",
        "",
        "SAMPLE COUNTS",
        "-" * 80,
    ]

    for split_name in ("trainval", "test"):
        for label in (
            "LEGITIMATE",
            "PHISHING",
            "UNKNOWN",
        ):
            count = split_label_counts[
                (split_name, label)
            ]

            summary_lines.append(
                f"{split_name:10s} "
                f"{label:12s}: "
                f"{count:,}"
            )

    summary_lines.extend(
        [
            f"Total sample folders: {len(audit_rows):,}",
            "",
            "IMAGE STATUS",
            "-" * 80,
            (
                "Valid screenshots: "
                f"{status_counts['OK']:,}"
            ),
            (
                "Missing screenshots: "
                f"{status_counts['MISSING_SCREENSHOT']:,}"
            ),
            (
                "Corrupt screenshots: "
                f"{status_counts['CORRUPT']:,}"
            ),
            (
                "Low-variance warnings: "
                f"{low_variance_count:,}"
            ),
            "",
            "EXACT DUPLICATES",
            "-" * 80,
            (
                "Exact duplicate groups: "
                f"{len(duplicate_groups):,}"
            ),
            (
                "Images inside duplicate groups: "
                f"{duplicate_images_count:,}"
            ),
            (
                "Extra duplicate copies: "
                f"{extra_duplicate_copies:,}"
            ),
            (
                "Groups crossing trainval/test: "
                f"{cross_split_duplicate_groups:,}"
            ),
            (
                "Groups crossing phishing/legitimate labels: "
                f"{cross_label_duplicate_groups:,}"
            ),
            "",
            "MOST COMMON IMAGE DIMENSIONS",
            "-" * 80,
        ]
    )

    if dimension_counts:
        for (
            width,
            height,
        ), count in dimension_counts.most_common(10):
            summary_lines.append(
                f"{width} x {height}: {count:,}"
            )
    else:
        summary_lines.append(
            "No valid image dimensions were found."
        )

    summary_lines.extend(
        [
            "",
            "REPORT FILES",
            "-" * 80,
            str(audit_report_path),
            str(duplicate_report_path),
            "",
            "IMPORTANT",
            "-" * 80,
            (
                "No screenshots or original dataset files were "
                "deleted, moved, renamed or modified."
            ),
            (
                "Do not remove warning or duplicate samples "
                "until the reports have been reviewed."
            ),
            "=" * 80,
        ]
    )

    summary_text = "\n".join(summary_lines)

    summary_report_path = (
        reports_folder / "phish360_audit_summary.txt"
    )

    summary_report_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    print()
    print(summary_text)
    print()
    print(f"Summary saved to: {summary_report_path}")


if __name__ == "__main__":
    main()