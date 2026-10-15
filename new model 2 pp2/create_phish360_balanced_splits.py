from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


# =============================================================================
# PROJECT SETTINGS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\oshan\Desktop\new model 2 pp2"
)

REPORTS_FOLDER = PROJECT_ROOT / "reports"

CLEAN_MANIFEST = (
    REPORTS_FOLDER
    / "phish360_clean_manifest.csv"
)

NEAR_DUPLICATE_GROUPS = (
    REPORTS_FOLDER
    / "phish360_near_duplicate_groups.csv"
)

OUTPUT_FOLDER = (
    REPORTS_FOLDER
    / "balanced_split"
)

FULL_SPLIT_MANIFEST = (
    OUTPUT_FOLDER
    / "phish360_balanced_split_manifest.csv"
)

TRAIN_MANIFEST = (
    OUTPUT_FOLDER
    / "phish360_train_manifest.csv"
)

VALIDATION_MANIFEST = (
    OUTPUT_FOLDER
    / "phish360_validation_manifest.csv"
)

TEST_MANIFEST = (
    OUTPUT_FOLDER
    / "phish360_test_manifest.csv"
)

UNUSED_MANIFEST = (
    OUTPUT_FOLDER
    / "phish360_unused_trainval_manifest.csv"
)

SUMMARY_FILE = (
    OUTPUT_FOLDER
    / "phish360_balanced_split_summary.txt"
)

RANDOM_SEED = 42
VALIDATION_FRACTION = 0.20

EXPECTED_CLEAN_IMAGES = 10_715
EXPECTED_TRAINVAL_IMAGES = 7_950
EXPECTED_TEST_IMAGES = 2_765


# =============================================================================
# GENERAL HELPERS
# =============================================================================

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


def normalize_split(value: str) -> str:
    value = str(value).strip().lower()

    if value in {"train_val", "train-val", "train val"}:
        return "trainval"

    return value


def read_csv_file(
    csv_path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    if not csv_path.is_file():
        raise SystemExit(
            f"\nERROR: Required file was not found:\n{csv_path}"
        )

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise SystemExit(
                f"\nERROR: CSV has no header:\n{csv_path}"
            )

        return list(reader), list(reader.fieldnames)


def write_csv_file(
    csv_path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# DISJOINT-SET STRUCTURE
# Keeps exact and near-duplicate samples in one indivisible group.
# =============================================================================

class DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {
            value: value
            for value in values
        }

        self.rank = {
            value: 0
            for value in values
        }

    def find(self, value: str) -> str:
        parent = self.parent[value]

        if parent != value:
            self.parent[value] = self.find(parent)

        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)

        if first_root == second_root:
            return

        first_rank = self.rank[first_root]
        second_rank = self.rank[second_root]

        if first_rank < second_rank:
            self.parent[first_root] = second_root
        elif first_rank > second_rank:
            self.parent[second_root] = first_root
        else:
            self.parent[second_root] = first_root
            self.rank[first_root] += 1


# =============================================================================
# EXACT-SIZE GROUP SELECTION
# =============================================================================

def select_units_exactly(
    units: list[dict[str, object]],
    target_images: int,
    random_seed: int,
    selection_name: str,
) -> list[dict[str, object]]:
    """
    Select complete groups whose combined image count exactly matches
    target_images. Dynamic programming prevents a group from being split.
    """
    if target_images < 0:
        raise SystemExit(
            f"\nERROR: Negative target for {selection_name}."
        )

    if target_images == 0:
        return []

    available_images = sum(
        int(unit["size"])
        for unit in units
    )

    if target_images > available_images:
        raise SystemExit(
            f"\nERROR: {selection_name} requests "
            f"{target_images:,} images, but only "
            f"{available_images:,} are available."
        )

    shuffled_units = list(units)
    random.Random(random_seed).shuffle(shuffled_units)

    reachable = [False] * (target_images + 1)
    previous_total = [-1] * (target_images + 1)
    previous_unit = [-1] * (target_images + 1)

    reachable[0] = True

    for unit_index, unit in enumerate(shuffled_units):
        unit_size = int(unit["size"])

        if unit_size <= 0:
            raise SystemExit(
                f"\nERROR: Invalid group size in {selection_name}."
            )

        if unit_size > target_images:
            continue

        for current_total in range(
            target_images - unit_size,
            -1,
            -1,
        ):
            new_total = current_total + unit_size

            if (
                reachable[current_total]
                and not reachable[new_total]
            ):
                reachable[new_total] = True
                previous_total[new_total] = current_total
                previous_unit[new_total] = unit_index

        if reachable[target_images]:
            break

    if not reachable[target_images]:
        raise SystemExit(
            f"\nERROR: Could not create an exact group-aware "
            f"selection of {target_images:,} images for "
            f"{selection_name}."
        )

    selected_indexes: set[int] = set()
    current_total = target_images

    while current_total > 0:
        unit_index = previous_unit[current_total]

        if unit_index < 0:
            raise SystemExit(
                f"\nERROR: Selection backtracking failed for "
                f"{selection_name}."
            )

        selected_indexes.add(unit_index)
        current_total = previous_total[current_total]

    selected_units = [
        shuffled_units[index]
        for index in sorted(selected_indexes)
    ]

    selected_image_count = sum(
        int(unit["size"])
        for unit in selected_units
    )

    if selected_image_count != target_images:
        raise SystemExit(
            f"\nERROR: Incorrect selection total for "
            f"{selection_name}."
        )

    return selected_units


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("PHISH360 BALANCED GROUP-AWARE SPLIT CREATOR")
    print("=" * 80)
    print(
        "This script creates manifest CSV files only."
    )
    print(
        "It does not delete, move, copy, rename, resize "
        "or modify screenshots."
    )
    print()

    clean_rows, clean_fieldnames = read_csv_file(
        CLEAN_MANIFEST
    )

    split_column = find_column(
        clean_fieldnames,
        ["split", "dataset_split"],
        "dataset split",
    )

    label_column = find_column(
        clean_fieldnames,
        ["label", "class", "class_label", "dataset_label"],
        "class label",
    )

    sample_column = find_column(
        clean_fieldnames,
        [
            "sample_folder",
            "sample_name",
            "sample_id",
            "sample",
            "folder_name",
        ],
        "sample folder",
    )

    sha256_column = find_column(
        clean_fieldnames,
        ["sha256", "sha_256", "file_sha256"],
        "SHA-256",
    )

    if len(clean_rows) != EXPECTED_CLEAN_IMAGES:
        raise SystemExit(
            "\nERROR: Clean-manifest image count changed.\n"
            f"Expected: {EXPECTED_CLEAN_IMAGES:,}\n"
            f"Found: {len(clean_rows):,}"
        )

    trainval_rows: list[dict[str, str]] = []
    test_rows_original: list[dict[str, str]] = []

    for row in clean_rows:
        split = normalize_split(row[split_column])

        if split == "trainval":
            trainval_rows.append(row)
        elif split == "test":
            test_rows_original.append(row)
        else:
            raise SystemExit(
                f"\nERROR: Unexpected dataset split: {split}"
            )

    if len(trainval_rows) != EXPECTED_TRAINVAL_IMAGES:
        raise SystemExit(
            "\nERROR: Trainval count changed.\n"
            f"Expected: {EXPECTED_TRAINVAL_IMAGES:,}\n"
            f"Found: {len(trainval_rows):,}"
        )

    if len(test_rows_original) != EXPECTED_TEST_IMAGES:
        raise SystemExit(
            "\nERROR: Test count changed.\n"
            f"Expected: {EXPECTED_TEST_IMAGES:,}\n"
            f"Found: {len(test_rows_original):,}"
        )

    row_by_sample: dict[str, dict[str, str]] = {}

    for row in trainval_rows:
        sample = row[sample_column].strip()

        if not sample:
            raise SystemExit(
                "\nERROR: Empty trainval sample name found."
            )

        if sample in row_by_sample:
            raise SystemExit(
                "\nERROR: Duplicate trainval sample name:\n"
                f"{sample}"
            )

        row_by_sample[sample] = row

    trainval_samples = sorted(row_by_sample)
    disjoint_set = DisjointSet(trainval_samples)

    # -------------------------------------------------------------------------
    # Join samples belonging to the same perceptual near-duplicate group.
    # -------------------------------------------------------------------------

    near_rows, near_fieldnames = read_csv_file(
        NEAR_DUPLICATE_GROUPS
    )

    near_group_column = find_column(
        near_fieldnames,
        ["near_duplicate_group", "group_number", "group_id"],
        "near-duplicate group",
    )

    near_split_column = find_column(
        near_fieldnames,
        ["split", "dataset_split"],
        "near-duplicate split",
    )

    near_sample_column = find_column(
        near_fieldnames,
        [
            "sample_folder",
            "sample_name",
            "sample_id",
            "sample",
            "folder_name",
        ],
        "near-duplicate sample",
    )

    near_group_members: dict[str, list[str]] = defaultdict(list)

    for row in near_rows:
        if normalize_split(row[near_split_column]) != "trainval":
            continue

        sample = row[near_sample_column].strip()

        if sample not in row_by_sample:
            # The sample may have been excluded by the clean manifest.
            continue

        group_number = row[near_group_column].strip()

        if group_number:
            near_group_members[group_number].append(sample)

    for members in near_group_members.values():
        unique_members = sorted(set(members))

        if len(unique_members) < 2:
            continue

        first_member = unique_members[0]

        for member in unique_members[1:]:
            disjoint_set.union(first_member, member)

    # -------------------------------------------------------------------------
    # Also join samples with the same SHA-256 inside trainval.
    # -------------------------------------------------------------------------

    sha256_members: dict[str, list[str]] = defaultdict(list)

    for sample, row in row_by_sample.items():
        sha256 = row[sha256_column].strip().lower()

        if sha256:
            sha256_members[sha256].append(sample)

    for members in sha256_members.values():
        if len(members) < 2:
            continue

        first_member = members[0]

        for member in members[1:]:
            disjoint_set.union(first_member, member)

    root_members: dict[str, list[str]] = defaultdict(list)

    for sample in trainval_samples:
        root = disjoint_set.find(sample)
        root_members[root].append(sample)

    sorted_member_groups = sorted(
        (
            sorted(members)
            for members in root_members.values()
        ),
        key=lambda members: members[0],
    )

    sample_to_group_id: dict[str, str] = {}
    units_by_label: dict[
        str,
        list[dict[str, object]],
    ] = defaultdict(list)

    for group_number, members in enumerate(
        sorted_member_groups,
        start=1,
    ):
        group_id = f"G{group_number:05d}"

        labels = {
            row_by_sample[sample][label_column].strip().upper()
            for sample in members
        }

        if len(labels) != 1:
            raise SystemExit(
                "\nERROR: A trainval duplicate group contains "
                "multiple labels.\n"
                f"Group: {group_id}\n"
                f"Labels: {sorted(labels)}\n"
                f"Samples: {members}"
            )

        label = next(iter(labels))

        if label not in {"LEGITIMATE", "PHISHING"}:
            raise SystemExit(
                f"\nERROR: Unexpected class label: {label}"
            )

        for sample in members:
            sample_to_group_id[sample] = group_id

        units_by_label[label].append(
            {
                "group_id": group_id,
                "samples": members,
                "size": len(members),
                "label": label,
            }
        )

    trainval_counts = Counter(
        row[label_column].strip().upper()
        for row in trainval_rows
    )

    required_labels = {"LEGITIMATE", "PHISHING"}

    if set(trainval_counts) != required_labels:
        raise SystemExit(
            "\nERROR: Expected exactly LEGITIMATE and PHISHING "
            f"classes, but found: {sorted(trainval_counts)}"
        )

    target_per_class = min(
        trainval_counts["LEGITIMATE"],
        trainval_counts["PHISHING"],
    )

    validation_target_per_class = round(
        target_per_class * VALIDATION_FRACTION
    )

    training_target_per_class = (
        target_per_class
        - validation_target_per_class
    )

    print("CLEAN INPUT COUNTS")
    print("-" * 80)
    print(f"Trainval images: {len(trainval_rows):,}")
    print(f"Fixed test images: {len(test_rows_original):,}")
    print(
        "Trainval LEGITIMATE: "
        f"{trainval_counts['LEGITIMATE']:,}"
    )
    print(
        "Trainval PHISHING: "
        f"{trainval_counts['PHISHING']:,}"
    )
    print(f"Duplicate-aware trainval groups: {len(root_members):,}")
    print()

    selected_units_by_label: dict[
        str,
        list[dict[str, object]],
    ] = {}

    validation_units_by_label: dict[
        str,
        list[dict[str, object]],
    ] = {}

    for label_index, label in enumerate(
        ["LEGITIMATE", "PHISHING"],
        start=1,
    ):
        selected_units = select_units_exactly(
            units=units_by_label[label],
            target_images=target_per_class,
            random_seed=RANDOM_SEED + label_index * 100,
            selection_name=f"balanced {label} selection",
        )

        validation_units = select_units_exactly(
            units=selected_units,
            target_images=validation_target_per_class,
            random_seed=RANDOM_SEED + label_index * 1000,
            selection_name=f"{label} validation split",
        )

        selected_units_by_label[label] = selected_units
        validation_units_by_label[label] = validation_units

    selected_group_ids = {
        str(unit["group_id"])
        for units in selected_units_by_label.values()
        for unit in units
    }

    validation_group_ids = {
        str(unit["group_id"])
        for units in validation_units_by_label.values()
        for unit in units
    }

    training_group_ids = (
        selected_group_ids
        - validation_group_ids
    )

    if training_group_ids & validation_group_ids:
        raise SystemExit(
            "\nERROR: Duplicate group leakage exists between "
            "training and validation."
        )

    additional_columns = [
        "model_split",
        "balanced_selection",
        "leakage_group_id",
        "split_random_seed",
    ]

    output_fieldnames = list(clean_fieldnames)

    for column in additional_columns:
        if column not in output_fieldnames:
            output_fieldnames.append(column)

    full_output_rows: list[dict[str, str]] = []
    train_output_rows: list[dict[str, str]] = []
    validation_output_rows: list[dict[str, str]] = []
    test_output_rows: list[dict[str, str]] = []
    unused_output_rows: list[dict[str, str]] = []

    for original_row in clean_rows:
        row = dict(original_row)
        original_split = normalize_split(row[split_column])
        sample = row[sample_column].strip()

        row["split_random_seed"] = str(RANDOM_SEED)

        if original_split == "test":
            row["model_split"] = "TEST"
            row["balanced_selection"] = "FIXED_TEST_KEEP"
            row["leakage_group_id"] = ""
            test_output_rows.append(row)

        else:
            group_id = sample_to_group_id[sample]
            row["leakage_group_id"] = group_id

            if group_id in validation_group_ids:
                row["model_split"] = "VALIDATION"
                row["balanced_selection"] = "SELECTED"
                validation_output_rows.append(row)
            elif group_id in training_group_ids:
                row["model_split"] = "TRAIN"
                row["balanced_selection"] = "SELECTED"
                train_output_rows.append(row)
            else:
                row["model_split"] = "UNUSED"
                row["balanced_selection"] = (
                    "NOT_SELECTED_FOR_CLASS_BALANCE"
                )
                unused_output_rows.append(row)

        full_output_rows.append(row)

    def count_labels(
        rows: list[dict[str, str]],
    ) -> Counter[str]:
        return Counter(
            row[label_column].strip().upper()
            for row in rows
        )

    train_counts = count_labels(train_output_rows)
    validation_counts = count_labels(validation_output_rows)
    test_counts = count_labels(test_output_rows)
    unused_counts = count_labels(unused_output_rows)

    for label in ["LEGITIMATE", "PHISHING"]:
        if train_counts[label] != training_target_per_class:
            raise SystemExit(
                f"\nERROR: Incorrect TRAIN {label} count."
            )

        if validation_counts[label] != validation_target_per_class:
            raise SystemExit(
                f"\nERROR: Incorrect VALIDATION {label} count."
            )

    train_samples = {
        row[sample_column].strip()
        for row in train_output_rows
    }

    validation_samples = {
        row[sample_column].strip()
        for row in validation_output_rows
    }

    if train_samples & validation_samples:
        raise SystemExit(
            "\nERROR: Sample leakage exists between training "
            "and validation."
        )

    train_sha256 = {
        row[sha256_column].strip().lower()
        for row in train_output_rows
        if row[sha256_column].strip()
    }

    validation_sha256 = {
        row[sha256_column].strip().lower()
        for row in validation_output_rows
        if row[sha256_column].strip()
    }

    if train_sha256 & validation_sha256:
        raise SystemExit(
            "\nERROR: Exact SHA-256 leakage exists between "
            "training and validation."
        )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv_file(
        FULL_SPLIT_MANIFEST,
        full_output_rows,
        output_fieldnames,
    )

    write_csv_file(
        TRAIN_MANIFEST,
        train_output_rows,
        output_fieldnames,
    )

    write_csv_file(
        VALIDATION_MANIFEST,
        validation_output_rows,
        output_fieldnames,
    )

    write_csv_file(
        TEST_MANIFEST,
        test_output_rows,
        output_fieldnames,
    )

    write_csv_file(
        UNUSED_MANIFEST,
        unused_output_rows,
        output_fieldnames,
    )

    summary_lines = [
        "=" * 80,
        "PHISH360 BALANCED GROUP-AWARE SPLIT SUMMARY",
        "=" * 80,
        "",
        "SETTINGS",
        "-" * 80,
        f"Random seed: {RANDOM_SEED}",
        f"Validation fraction: {VALIDATION_FRACTION:.0%}",
        f"Balanced target per class: {target_per_class:,}",
        "Fixed test set rebalanced: NO",
        "",
        "FINAL SPLITS",
        "-" * 80,
        (
            "TRAIN       | LEGITIMATE: "
            f"{train_counts['LEGITIMATE']:,} | PHISHING: "
            f"{train_counts['PHISHING']:,} | TOTAL: "
            f"{len(train_output_rows):,}"
        ),
        (
            "VALIDATION  | LEGITIMATE: "
            f"{validation_counts['LEGITIMATE']:,} | PHISHING: "
            f"{validation_counts['PHISHING']:,} | TOTAL: "
            f"{len(validation_output_rows):,}"
        ),
        (
            "TEST         | LEGITIMATE: "
            f"{test_counts['LEGITIMATE']:,} | PHISHING: "
            f"{test_counts['PHISHING']:,} | TOTAL: "
            f"{len(test_output_rows):,}"
        ),
        (
            "UNUSED       | LEGITIMATE: "
            f"{unused_counts['LEGITIMATE']:,} | PHISHING: "
            f"{unused_counts['PHISHING']:,} | TOTAL: "
            f"{len(unused_output_rows):,}"
        ),
        "",
        "LEAKAGE VERIFICATION",
        "-" * 80,
        "Samples crossing train/validation: 0",
        "Duplicate groups crossing train/validation: 0",
        "Exact SHA-256 values crossing train/validation: 0",
        "Test samples modified or removed: 0",
        "",
        "REPORT FILES",
        "-" * 80,
        str(FULL_SPLIT_MANIFEST),
        str(TRAIN_MANIFEST),
        str(VALIDATION_MANIFEST),
        str(TEST_MANIFEST),
        str(UNUSED_MANIFEST),
        str(SUMMARY_FILE),
        "",
        "IMPORTANT",
        "-" * 80,
        (
            "No original screenshots were deleted, moved, copied, "
            "renamed, resized or modified."
        ),
        "=" * 80,
    ]

    summary_text = "\n".join(summary_lines)

    SUMMARY_FILE.write_text(
        summary_text,
        encoding="utf-8",
    )

    print(summary_text)


if __name__ == "__main__":
    main()
