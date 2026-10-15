from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\oshan\Desktop\new model 2 pp2"
)

REPORTS_FOLDER = PROJECT_ROOT / "reports"

HASHES_CSV = (
    REPORTS_FOLDER
    / "phish360_perceptual_hashes.csv"
)

PAIR_DECISIONS_CSV = (
    REPORTS_FOLDER
    / "cross_split_pair_review"
    / "cross_split_pair_decisions.csv"
)

ALL_DECISIONS_MANIFEST = (
    REPORTS_FOLDER
    / "phish360_manifest_with_cleaning_decisions.csv"
)

CLEAN_MANIFEST = (
    REPORTS_FOLDER
    / "phish360_clean_manifest.csv"
)

EXCLUDED_SAMPLES_CSV = (
    REPORTS_FOLDER
    / "phish360_excluded_samples.csv"
)

SUMMARY_FILE = (
    REPORTS_FOLDER
    / "phish360_clean_manifest_summary.txt"
)


# Expected results from the completed manual review
EXPECTED_PAIR_COUNT = 45
EXPECTED_CONFIRMED_PAIRS = 17
EXPECTED_FALSE_PAIRS = 28
EXPECTED_UNIQUE_NEAR_DUPLICATE_TRAINVAL = 16

ALLOWED_DECISIONS = {
    "CONFIRMED_NEAR_DUPLICATE",
    "FALSE_MATCH_KEEP_BOTH",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_header(value: str) -> str:
    """Normalize a CSV column name for flexible matching."""
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
    """Find a CSV column using several possible names."""
    normalized_columns = {
        normalize_header(name): name
        for name in fieldnames
    }

    for possible_name in possible_names:
        normalized_name = normalize_header(possible_name)

        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    raise SystemExit(
        f"\nERROR: Could not find the {description} column.\n"
        f"Available columns:\n{fieldnames}"
    )


def normalize_split(value: str) -> str:
    """Normalize dataset split values."""
    value = str(value).strip().lower()

    if value in {"train_val", "train-val", "train val"}:
        return "trainval"

    return value


def normalize_decision(value: str) -> str:
    """Normalize manual decision values."""
    return str(value).strip().upper()


def read_csv_file(
    csv_path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    """Read a CSV file and return rows and field names."""
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

        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    return rows, fieldnames


def pair_sort_key(value: str) -> tuple[int, str]:
    """Sort numeric pair numbers correctly."""
    try:
        return int(value), value
    except ValueError:
        return 999999999, value


# =============================================================================
# LOAD AND VALIDATE MANUAL PAIR DECISIONS
# =============================================================================

def load_and_validate_pair_decisions():
    decision_rows, fieldnames = read_csv_file(
        PAIR_DECISIONS_CSV
    )

    pair_number_column = find_column(
        fieldnames,
        ["pair_number"],
        "pair number",
    )

    decision_column = find_column(
        fieldnames,
        ["decision"],
        "decision",
    )

    first_split_column = find_column(
        fieldnames,
        ["first_split"],
        "first split",
    )

    first_sample_column = find_column(
        fieldnames,
        ["first_sample"],
        "first sample",
    )

    second_split_column = find_column(
        fieldnames,
        ["second_split"],
        "second split",
    )

    second_sample_column = find_column(
        fieldnames,
        ["second_sample"],
        "second sample",
    )

    if len(decision_rows) != EXPECTED_PAIR_COUNT:
        raise SystemExit(
            "\nERROR: Unexpected number of decision rows.\n"
            f"Expected: {EXPECTED_PAIR_COUNT}\n"
            f"Found: {len(decision_rows)}"
        )

    seen_pair_numbers: set[str] = set()
    decision_counts: Counter[str] = Counter()

    confirmed_trainval_samples: dict[
        tuple[str, str],
        set[str],
    ] = defaultdict(set)

    confirmed_pairs: list[dict[str, str]] = []

    for row_number, row in enumerate(
        decision_rows,
        start=2,
    ):
        pair_number = row[pair_number_column].strip()
        decision = normalize_decision(
            row[decision_column]
        )

        if not pair_number:
            raise SystemExit(
                f"\nERROR: Missing pair number on CSV row {row_number}."
            )

        if pair_number in seen_pair_numbers:
            raise SystemExit(
                f"\nERROR: Duplicate pair number: {pair_number}"
            )

        seen_pair_numbers.add(pair_number)

        if decision not in ALLOWED_DECISIONS:
            raise SystemExit(
                "\nERROR: Invalid or incomplete decision.\n"
                f"CSV row: {row_number}\n"
                f"Pair: {pair_number}\n"
                f"Decision: {decision}\n\n"
                "Every decision must be either:\n"
                "CONFIRMED_NEAR_DUPLICATE\n"
                "FALSE_MATCH_KEEP_BOTH"
            )

        decision_counts[decision] += 1

        first_split = normalize_split(
            row[first_split_column]
        )

        second_split = normalize_split(
            row[second_split_column]
        )

        first_sample = row[first_sample_column].strip()
        second_sample = row[second_sample_column].strip()

        split_set = {
            first_split,
            second_split,
        }

        if split_set != {"trainval", "test"}:
            raise SystemExit(
                "\nERROR: A reviewed pair is not a "
                "trainval/test pair.\n"
                f"Pair: {pair_number}\n"
                f"Splits: {first_split}, {second_split}"
            )

        if decision == "CONFIRMED_NEAR_DUPLICATE":
            confirmed_pairs.append(
                {
                    "pair_number": pair_number,
                    "first_split": first_split,
                    "first_sample": first_sample,
                    "second_split": second_split,
                    "second_sample": second_sample,
                }
            )

            if first_split == "trainval":
                confirmed_trainval_samples[
                    ("trainval", first_sample)
                ].add(pair_number)

            if second_split == "trainval":
                confirmed_trainval_samples[
                    ("trainval", second_sample)
                ].add(pair_number)

    confirmed_count = decision_counts[
        "CONFIRMED_NEAR_DUPLICATE"
    ]

    false_count = decision_counts[
        "FALSE_MATCH_KEEP_BOTH"
    ]

    if confirmed_count != EXPECTED_CONFIRMED_PAIRS:
        raise SystemExit(
            "\nERROR: Confirmed-pair count does not match.\n"
            f"Expected: {EXPECTED_CONFIRMED_PAIRS}\n"
            f"Found: {confirmed_count}"
        )

    if false_count != EXPECTED_FALSE_PAIRS:
        raise SystemExit(
            "\nERROR: False-match count does not match.\n"
            f"Expected: {EXPECTED_FALSE_PAIRS}\n"
            f"Found: {false_count}"
        )

    unique_trainval_count = len(
        confirmed_trainval_samples
    )

    if (
        unique_trainval_count
        != EXPECTED_UNIQUE_NEAR_DUPLICATE_TRAINVAL
    ):
        raise SystemExit(
            "\nERROR: Unexpected number of unique trainval "
            "near-duplicate samples.\n"
            "Expected: "
            f"{EXPECTED_UNIQUE_NEAR_DUPLICATE_TRAINVAL}\n"
            f"Found: {unique_trainval_count}"
        )

    return (
        decision_counts,
        confirmed_pairs,
        confirmed_trainval_samples,
    )


# =============================================================================
# CREATE CLEAN MANIFEST
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("PHISH360 CLEAN MANIFEST CREATOR")
    print("=" * 80)

    print(
        "This script creates CSV reports only."
    )

    print(
        "It does not delete, move, rename, resize "
        "or modify dataset screenshots."
    )

    print()

    (
        decision_counts,
        confirmed_pairs,
        confirmed_trainval_samples,
    ) = load_and_validate_pair_decisions()

    print("PAIR DECISION VALIDATION")
    print("-" * 80)
    print(
        "Total reviewed pairs: "
        f"{sum(decision_counts.values()):,}"
    )
    print(
        "Confirmed near-duplicate pairs: "
        f"{decision_counts['CONFIRMED_NEAR_DUPLICATE']:,}"
    )
    print(
        "False matches kept: "
        f"{decision_counts['FALSE_MATCH_KEEP_BOTH']:,}"
    )
    print(
        "Unique trainval samples from confirmed pairs: "
        f"{len(confirmed_trainval_samples):,}"
    )
    print()

    hash_rows, hash_fieldnames = read_csv_file(
        HASHES_CSV
    )

    split_column = find_column(
        hash_fieldnames,
        ["split", "dataset_split"],
        "dataset split",
    )

    label_column = find_column(
        hash_fieldnames,
        [
            "label",
            "class",
            "class_label",
            "dataset_label",
        ],
        "class label",
    )

    sample_column = find_column(
        hash_fieldnames,
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
        hash_fieldnames,
        [
            "sha256",
            "sha_256",
            "file_sha256",
        ],
        "SHA-256",
    )

    print("DATASET MANIFEST")
    print("-" * 80)
    print(
        f"Successfully hashed images loaded: "
        f"{len(hash_rows):,}"
    )

    # Create a dataset index.
    dataset_index: dict[
        tuple[str, str],
        list[dict[str, str]],
    ] = defaultdict(list)

    sha256_splits: dict[str, set[str]] = defaultdict(set)

    for row in hash_rows:
        split = normalize_split(
            row[split_column]
        )

        sample = row[sample_column].strip()
        sha256 = row[sha256_column].strip().lower()

        dataset_index[(split, sample)].append(row)

        if sha256:
            sha256_splits[sha256].add(split)

    # Confirm that every reviewed sample exists.
    reviewed_keys: set[tuple[str, str]] = set()

    for pair in confirmed_pairs:
        reviewed_keys.add(
            (
                pair["first_split"],
                pair["first_sample"],
            )
        )

        reviewed_keys.add(
            (
                pair["second_split"],
                pair["second_sample"],
            )
        )

    missing_samples = sorted(
        key
        for key in reviewed_keys
        if key not in dataset_index
    )

    if missing_samples:
        formatted_missing = "\n".join(
            f"{split} | {sample}"
            for split, sample in missing_samples
        )

        raise SystemExit(
            "\nERROR: Some reviewed samples were not found "
            "in the perceptual-hash manifest:\n"
            f"{formatted_missing}"
        )

    # Detect objective exact duplicates crossing trainval/test.
    cross_split_exact_hashes = {
        sha256
        for sha256, splits in sha256_splits.items()
        if "trainval" in splits and "test" in splits
    }

    print(
        "Cross-split exact SHA-256 groups found: "
        f"{len(cross_split_exact_hashes):,}"
    )

    additional_columns = [
        "include_in_clean_dataset",
        "clean_status",
        "exclusion_reason",
        "matched_confirmed_pair_numbers",
        "cross_split_exact_duplicate",
    ]

    output_fieldnames = list(hash_fieldnames)

    for column in additional_columns:
        if column not in output_fieldnames:
            output_fieldnames.append(column)

    all_output_rows: list[dict[str, str]] = []
    clean_rows: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []

    original_counts: Counter[
        tuple[str, str]
    ] = Counter()

    kept_counts: Counter[
        tuple[str, str]
    ] = Counter()

    excluded_counts: Counter[
        tuple[str, str]
    ] = Counter()

    for original_row in hash_rows:
        row = dict(original_row)

        split = normalize_split(
            row[split_column]
        )

        label = row[label_column].strip().upper()
        sample = row[sample_column].strip()
        sha256 = row[sha256_column].strip().lower()

        key = (split, sample)
        original_counts[(split, label)] += 1

        reasons: list[str] = []

        confirmed_pair_numbers = sorted(
            confirmed_trainval_samples.get(
                key,
                set(),
            ),
            key=pair_sort_key,
        )

        is_cross_split_exact = (
            bool(sha256)
            and sha256 in cross_split_exact_hashes
        )

        # Only training-side samples are excluded.
        # The fixed test set is always preserved.
        if split == "trainval":
            if confirmed_pair_numbers:
                reasons.append(
                    "CONFIRMED_CROSS_SPLIT_NEAR_DUPLICATE"
                )

            if is_cross_split_exact:
                reasons.append(
                    "CROSS_SPLIT_EXACT_DUPLICATE"
                )

        if reasons:
            row["include_in_clean_dataset"] = "FALSE"
            row["clean_status"] = "EXCLUDED"
            row["exclusion_reason"] = "|".join(reasons)

            excluded_counts[(split, label)] += 1
            excluded_rows.append(row)
        else:
            row["include_in_clean_dataset"] = "TRUE"
            row["clean_status"] = "KEEP"
            row["exclusion_reason"] = ""

            kept_counts[(split, label)] += 1
            clean_rows.append(row)

        row["matched_confirmed_pair_numbers"] = "|".join(
            confirmed_pair_numbers
        )

        row["cross_split_exact_duplicate"] = (
            "TRUE"
            if is_cross_split_exact
            else "FALSE"
        )

        all_output_rows.append(row)

    test_exclusions = [
        row
        for row in excluded_rows
        if normalize_split(row[split_column]) == "test"
    ]

    if test_exclusions:
        raise SystemExit(
            "\nERROR: Test samples were selected for exclusion. "
            "No output files were written."
        )

    # Verify confirmed pairs no longer cross the kept data.
    kept_keys = {
        (
            normalize_split(row[split_column]),
            row[sample_column].strip(),
        )
        for row in clean_rows
    }

    remaining_confirmed_leakage: list[str] = []

    for pair in confirmed_pairs:
        first_key = (
            pair["first_split"],
            pair["first_sample"],
        )

        second_key = (
            pair["second_split"],
            pair["second_sample"],
        )

        if first_key in kept_keys and second_key in kept_keys:
            remaining_confirmed_leakage.append(
                pair["pair_number"]
            )

    if remaining_confirmed_leakage:
        raise SystemExit(
            "\nERROR: Confirmed cross-split near-duplicate "
            "leakage remains for pairs:\n"
            + ", ".join(remaining_confirmed_leakage)
        )

    # Verify no exact SHA-256 duplicate remains across trainval/test.
    kept_sha_splits: dict[str, set[str]] = defaultdict(set)

    for row in clean_rows:
        sha256 = row[sha256_column].strip().lower()
        split = normalize_split(row[split_column])

        if sha256:
            kept_sha_splits[sha256].add(split)

    remaining_exact_cross_split = [
        sha256
        for sha256, splits in kept_sha_splits.items()
        if "trainval" in splits and "test" in splits
    ]

    if remaining_exact_cross_split:
        raise SystemExit(
            "\nERROR: Exact duplicates still cross "
            "trainval and test after cleaning."
        )

    REPORTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ALL_DECISIONS_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=output_fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(all_output_rows)

    with CLEAN_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=output_fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(clean_rows)

    with EXCLUDED_SAMPLES_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=output_fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(excluded_rows)

    summary_lines = [
        "=" * 80,
        "PHISH360 CLEAN MANIFEST SUMMARY",
        "=" * 80,
        "",
        "PAIR REVIEW",
        "-" * 80,
        (
            "Reviewed pairs: "
            f"{sum(decision_counts.values()):,}"
        ),
        (
            "Confirmed near-duplicate pairs: "
            f"{decision_counts['CONFIRMED_NEAR_DUPLICATE']:,}"
        ),
        (
            "False-match pairs kept: "
            f"{decision_counts['FALSE_MATCH_KEEP_BOTH']:,}"
        ),
        (
            "Unique confirmed trainval samples: "
            f"{len(confirmed_trainval_samples):,}"
        ),
        "",
        "CLEANING RESULTS",
        "-" * 80,
        f"Original manifest images: {len(hash_rows):,}",
        f"Images kept: {len(clean_rows):,}",
        f"Trainval images excluded: {len(excluded_rows):,}",
        "Test images excluded: 0",
        (
            "Cross-split exact SHA-256 groups: "
            f"{len(cross_split_exact_hashes):,}"
        ),
        "Remaining confirmed pair leakage: 0",
        "Remaining exact cross-split leakage: 0",
        "",
        "COUNTS BY SPLIT AND LABEL",
        "-" * 80,
    ]

    all_count_keys = sorted(
        set(original_counts)
        | set(kept_counts)
        | set(excluded_counts)
    )

    for split, label in all_count_keys:
        summary_lines.append(
            f"{split.upper():8s} | "
            f"{label:10s} | "
            f"Original: {original_counts[(split, label)]:,} | "
            f"Kept: {kept_counts[(split, label)]:,} | "
            f"Excluded: {excluded_counts[(split, label)]:,}"
        )

    summary_lines.extend(
        [
            "",
            "REPORT FILES",
            "-" * 80,
            str(ALL_DECISIONS_MANIFEST),
            str(CLEAN_MANIFEST),
            str(EXCLUDED_SAMPLES_CSV),
            str(SUMMARY_FILE),
            "",
            "IMPORTANT",
            "-" * 80,
            (
                "No original screenshots were deleted, moved, "
                "renamed, resized or modified."
            ),
            "=" * 80,
        ]
    )

    summary_text = "\n".join(summary_lines)

    SUMMARY_FILE.write_text(
        summary_text,
        encoding="utf-8",
    )

    print()
    print(summary_text)


if __name__ == "__main__":
    main()