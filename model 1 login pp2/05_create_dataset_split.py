from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path


# =============================================================================
# PATHS AND SETTINGS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\oshan\Desktop\model 1 login pp2"
)

PREPARED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "prepared_login_384x160"
)

NORMALIZED_MANIFEST = (
    PROJECT_ROOT
    / "reports"
    / "04_normalization"
    / "normalized_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "05_dataset_split"
)

RANDOM_SEED = 20260826

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

LABEL_MAPPING = {
    "NOT_MALICIOUS": 0,
    "MALICIOUS": 1,
}

OUTPUT_COLUMNS = [
    "split",
    "class_label",
    "numeric_label",
    "prepared_relative_path",
    "original_relative_path",
    "viewport_action",
    "output_sha256",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def read_normalized_manifest() -> list[dict[str, str]]:
    with NORMALIZED_MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=OUTPUT_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(rows)


def prepare_output_row(
    row: dict[str, str],
    split_name: str,
) -> dict[str, object]:
    class_label = row["class_label"]

    return {
        "split": split_name,
        "class_label": class_label,
        "numeric_label": LABEL_MAPPING[class_label],
        "prepared_relative_path": (
            row["output_relative_path"]
        ),
        "original_relative_path": (
            row["original_relative_path"]
        ),
        "viewport_action": row["viewport_action"],
        "output_sha256": row["output_sha256"],
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("MODEL 1 REPRODUCIBLE DATASET SPLIT")
    print("=" * 80)
    print(f"Random seed       : {RANDOM_SEED}")
    print(f"Train ratio       : {TRAIN_RATIO:.0%}")
    print(f"Validation ratio  : {VALIDATION_RATIO:.0%}")
    print(
        f"Test ratio        : "
        f"{1 - TRAIN_RATIO - VALIDATION_RATIO:.0%}"
    )
    print()

    if not NORMALIZED_MANIFEST.exists():
        raise FileNotFoundError(
            f"Normalized manifest not found:\n"
            f"{NORMALIZED_MANIFEST}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_normalized_manifest()

    if not rows:
        raise ValueError(
            "The normalized manifest is empty."
        )

    unknown_classes = {
        row["class_label"]
        for row in rows
    } - set(LABEL_MAPPING)

    if unknown_classes:
        raise ValueError(
            f"Unexpected class labels: "
            f"{sorted(unknown_classes)}"
        )

    split_rows: dict[
        str,
        list[dict[str, object]],
    ] = {
        "TRAIN": [],
        "VALIDATION": [],
        "TEST": [],
    }

    # =========================================================================
    # SPLIT EACH CLASS SEPARATELY
    # =========================================================================

    for class_offset, class_label in enumerate(
        LABEL_MAPPING,
        start=1,
    ):
        class_rows = sorted(
            [
                row
                for row in rows
                if row["class_label"] == class_label
            ],
            key=lambda row: row["output_relative_path"],
        )

        if not class_rows:
            raise ValueError(
                f"No images found for class: "
                f"{class_label}"
            )

        class_random = random.Random(
            RANDOM_SEED + class_offset
        )

        class_random.shuffle(class_rows)

        total_class_images = len(class_rows)

        train_count = int(
            total_class_images * TRAIN_RATIO
        )

        validation_count = int(
            total_class_images * VALIDATION_RATIO
        )

        test_count = (
            total_class_images
            - train_count
            - validation_count
        )

        train_class_rows = class_rows[
            :train_count
        ]

        validation_class_rows = class_rows[
            train_count:
            train_count + validation_count
        ]

        test_class_rows = class_rows[
            train_count + validation_count:
        ]

        if len(test_class_rows) != test_count:
            raise ValueError(
                f"Unexpected test count for "
                f"{class_label}"
            )

        split_rows["TRAIN"].extend(
            prepare_output_row(
                row,
                "TRAIN",
            )
            for row in train_class_rows
        )

        split_rows["VALIDATION"].extend(
            prepare_output_row(
                row,
                "VALIDATION",
            )
            for row in validation_class_rows
        )

        split_rows["TEST"].extend(
            prepare_output_row(
                row,
                "TEST",
            )
            for row in test_class_rows
        )

    # Shuffle the two classes together inside each split.
    for split_offset, split_name in enumerate(
        [
            "TRAIN",
            "VALIDATION",
            "TEST",
        ],
        start=100,
    ):
        split_random = random.Random(
            RANDOM_SEED + split_offset
        )

        split_random.shuffle(
            split_rows[split_name]
        )

    # =========================================================================
    # CHECK THAT EVERY PREPARED IMAGE EXISTS
    # =========================================================================

    missing_files: list[str] = []

    for current_rows in split_rows.values():
        for row in current_rows:
            image_path = (
                PREPARED_ROOT
                / str(
                    row["prepared_relative_path"]
                )
            )

            if not image_path.exists():
                missing_files.append(
                    str(image_path)
                )

    if missing_files:
        raise FileNotFoundError(
            "Prepared images are missing:\n"
            + "\n".join(missing_files[:20])
        )

    # =========================================================================
    # CHECK FOR PATH LEAKAGE
    # =========================================================================

    path_sets = {
        split_name: {
            str(
                row["prepared_relative_path"]
            )
            for row in current_rows
        }
        for split_name, current_rows
        in split_rows.items()
    }

    if (
        path_sets["TRAIN"]
        & path_sets["VALIDATION"]
    ):
        raise ValueError(
            "Path leakage between "
            "TRAIN and VALIDATION"
        )

    if (
        path_sets["TRAIN"]
        & path_sets["TEST"]
    ):
        raise ValueError(
            "Path leakage between "
            "TRAIN and TEST"
        )

    if (
        path_sets["VALIDATION"]
        & path_sets["TEST"]
    ):
        raise ValueError(
            "Path leakage between "
            "VALIDATION and TEST"
        )

    # =========================================================================
    # CHECK FOR EXACT-IMAGE HASH LEAKAGE
    # =========================================================================

    hash_sets = {
        split_name: {
            str(row["output_sha256"])
            for row in current_rows
        }
        for split_name, current_rows
        in split_rows.items()
    }

    if (
        hash_sets["TRAIN"]
        & hash_sets["VALIDATION"]
    ):
        raise ValueError(
            "Exact-image leakage between "
            "TRAIN and VALIDATION"
        )

    if (
        hash_sets["TRAIN"]
        & hash_sets["TEST"]
    ):
        raise ValueError(
            "Exact-image leakage between "
            "TRAIN and TEST"
        )

    if (
        hash_sets["VALIDATION"]
        & hash_sets["TEST"]
    ):
        raise ValueError(
            "Exact-image leakage between "
            "VALIDATION and TEST"
        )

    # =========================================================================
    # WRITE THE SPLIT MANIFESTS
    # =========================================================================

    write_csv(
        OUTPUT_DIR / "train_manifest.csv",
        split_rows["TRAIN"],
    )

    write_csv(
        OUTPUT_DIR
        / "validation_manifest.csv",
        split_rows["VALIDATION"],
    )

    write_csv(
        OUTPUT_DIR / "test_manifest.csv",
        split_rows["TEST"],
    )

    combined_rows: list[
        dict[str, object]
    ] = []

    for split_name in [
        "TRAIN",
        "VALIDATION",
        "TEST",
    ]:
        combined_rows.extend(
            split_rows[split_name]
        )

    write_csv(
        OUTPUT_DIR
        / "complete_split_manifest.csv",
        combined_rows,
    )

    # =========================================================================
    # CREATE SUMMARY
    # =========================================================================

    summary_lines = [
        "=" * 80,
        "MODEL 1 DATASET SPLIT SUMMARY",
        "=" * 80,
        "",
        f"Random seed: {RANDOM_SEED}",
        "",
    ]

    for split_name in [
        "TRAIN",
        "VALIDATION",
        "TEST",
    ]:
        current_rows = split_rows[
            split_name
        ]

        class_counts = Counter(
            str(row["class_label"])
            for row in current_rows
        )

        action_counts = Counter(
            str(row["viewport_action"])
            for row in current_rows
        )

        summary_lines.extend(
            [
                split_name,
                "-" * 80,
                (
                    "Total images          : "
                    f"{len(current_rows)}"
                ),
                (
                    "MALICIOUS              : "
                    f"{class_counts['MALICIOUS']}"
                ),
                (
                    "NOT_MALICIOUS          : "
                    f"{class_counts['NOT_MALICIOUS']}"
                ),
                (
                    "Top-cropped images     : "
                    f"{action_counts['TOP_CROP']}"
                ),
                (
                    "Uncropped images       : "
                    f"{action_counts['NONE']}"
                ),
                "",
            ]
        )

    summary_lines.extend(
        [
            "LEAKAGE CHECKS",
            "-" * 80,
            (
                "Cross-split path leakage     : 0"
            ),
            (
                "Cross-split exact leakage    : 0"
            ),
            "",
            (
                "The TEST manifest must not be "
                "used during"
            ),
            (
                "model selection or threshold "
                "tuning."
            ),
            "",
            (
                "No image was moved, deleted "
                "or modified."
            ),
            "=" * 80,
        ]
    )

    summary_text = "\n".join(
        summary_lines
    )

    summary_path = (
        OUTPUT_DIR
        / "dataset_split_summary.txt"
    )

    summary_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    print(summary_text)
    print()
    print(
        f"Reports saved to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()