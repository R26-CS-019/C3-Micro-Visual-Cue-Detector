from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import imagehash
from PIL import Image, ImageOps


# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\oshan\Desktop\new model 2 pp2"
)

AUDIT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "phish360_dataset_audit.csv"
)

REPORTS_FOLDER = PROJECT_ROOT / "reports"

# Conservative near-duplicate thresholds.
# Smaller values mean stricter matching.
PHASH_DISTANCE_THRESHOLD = 6
DHASH_DISTANCE_THRESHOLD = 10


def hamming_distance(first_hash: int, second_hash: int) -> int:
    """Calculate the Hamming distance between two integer hashes."""
    return (first_hash ^ second_hash).bit_count()


class BKNode:
    """One node in a BK-tree."""

    def __init__(self, value: int, record_index: int) -> None:
        self.value = value
        self.record_indexes = [record_index]
        self.children: dict[int, BKNode] = {}


class BKTree:
    """
    BK-tree for efficiently finding perceptual hashes with a small
    Hamming distance.
    """

    def __init__(self) -> None:
        self.root: BKNode | None = None

    def add(self, value: int, record_index: int) -> None:
        if self.root is None:
            self.root = BKNode(value, record_index)
            return

        current_node = self.root

        while True:
            distance = hamming_distance(
                value,
                current_node.value,
            )

            if distance == 0:
                current_node.record_indexes.append(record_index)
                return

            if distance in current_node.children:
                current_node = current_node.children[distance]
            else:
                current_node.children[distance] = BKNode(
                    value,
                    record_index,
                )
                return

    def query(
        self,
        value: int,
        maximum_distance: int,
    ) -> list[tuple[int, int]]:
        """
        Return record indexes with hashes inside the selected distance.
        """
        if self.root is None:
            return []

        results: list[tuple[int, int]] = []
        nodes_to_check = [self.root]

        while nodes_to_check:
            current_node = nodes_to_check.pop()

            distance = hamming_distance(
                value,
                current_node.value,
            )

            if distance <= maximum_distance:
                for record_index in current_node.record_indexes:
                    results.append(
                        (record_index, distance)
                    )

            minimum_edge = max(
                0,
                distance - maximum_distance,
            )

            maximum_edge = (
                distance + maximum_distance
            )

            for edge_distance, child_node in (
                current_node.children.items()
            ):
                if (
                    minimum_edge
                    <= edge_distance
                    <= maximum_edge
                ):
                    nodes_to_check.append(child_node)

        return results


class UnionFind:
    """Join connected near-duplicate images into groups."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[
                self.parent[item]
            ]
            item = self.parent[item]

        return item

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)

        if first_root == second_root:
            return

        if self.rank[first_root] < self.rank[second_root]:
            self.parent[first_root] = second_root

        elif self.rank[first_root] > self.rank[second_root]:
            self.parent[second_root] = first_root

        else:
            self.parent[second_root] = first_root
            self.rank[first_root] += 1


def load_valid_audit_rows() -> list[dict[str, str]]:
    """Load successfully audited screenshots."""
    if not AUDIT_REPORT.is_file():
        raise SystemExit(
            "\nERROR: The dataset audit report was not found.\n"
            f"Expected file:\n{AUDIT_REPORT}\n\n"
            "Run audit_phish360.py first."
        )

    valid_rows: list[dict[str, str]] = []

    with AUDIT_REPORT.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if (
                row.get("status") == "OK"
                and row.get("screenshot_path")
            ):
                valid_rows.append(row)

    return valid_rows


def calculate_perceptual_hashes(
    image_path: Path,
) -> tuple[str, str, int, int]:
    """Calculate pHash and dHash for one screenshot."""
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        perceptual_hash = imagehash.phash(
            image,
            hash_size=8,
        )

        difference_hash = imagehash.dhash(
            image,
            hash_size=8,
        )

    perceptual_hash_string = str(perceptual_hash)
    difference_hash_string = str(difference_hash)

    perceptual_hash_integer = int(
        perceptual_hash_string,
        16,
    )

    difference_hash_integer = int(
        difference_hash_string,
        16,
    )

    return (
        perceptual_hash_string,
        difference_hash_string,
        perceptual_hash_integer,
        difference_hash_integer,
    )


def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    """Write a list of dictionaries to a CSV file."""
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
    print("PHISH360 NEAR-DUPLICATE AUDIT")
    print("=" * 80)

    print(f"Project folder: {PROJECT_ROOT}")
    print(f"Audit report: {AUDIT_REPORT}")
    print()

    print(
        "This script only reads screenshots and creates reports."
    )

    print(
        "It does not delete, move, rename, resize or modify "
        "dataset images."
    )

    print()

    REPORTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_rows = load_valid_audit_rows()
    total_images = len(audit_rows)

    print(
        f"Valid screenshots available: {total_images:,}"
    )

    print(
        f"pHash threshold: {PHASH_DISTANCE_THRESHOLD}"
    )

    print(
        f"dHash threshold: {DHASH_DISTANCE_THRESHOLD}"
    )

    print()
    print("Calculating perceptual hashes...")
    print()

    # Allocate enough Union-Find positions.
    union_find = UnionFind(total_images)
    hash_tree = BKTree()

    hash_records: list[dict[str, object]] = []
    hash_failures: list[dict[str, object]] = []

    near_duplicate_pairs_path = (
        REPORTS_FOLDER
        / "phish360_near_duplicate_pairs.csv"
    )

    pair_fieldnames = [
        "pair_number",
        "phash_distance",
        "dhash_distance",
        "first_split",
        "first_label",
        "first_sample",
        "first_sha256",
        "first_path",
        "second_split",
        "second_label",
        "second_sample",
        "second_sha256",
        "second_path",
        "cross_split",
        "cross_label",
    ]

    pair_count = 0
    cross_split_pair_count = 0
    cross_label_pair_count = 0

    with near_duplicate_pairs_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as pair_file:

        pair_writer = csv.DictWriter(
            pair_file,
            fieldnames=pair_fieldnames,
        )

        pair_writer.writeheader()

        for audit_number, audit_row in enumerate(
            audit_rows,
            start=1,
        ):
            image_path = Path(
                audit_row["screenshot_path"]
            )

            try:
                (
                    phash_string,
                    dhash_string,
                    phash_integer,
                    dhash_integer,
                ) = calculate_perceptual_hashes(image_path)

            except Exception as error:
                hash_failures.append(
                    {
                        "split": audit_row.get(
                            "split",
                            "",
                        ),
                        "label": audit_row.get(
                            "label",
                            "",
                        ),
                        "sample_folder": audit_row.get(
                            "sample_folder",
                            "",
                        ),
                        "screenshot_path": str(image_path),
                        "error": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                    }
                )

                continue

            current_record_index = len(hash_records)

            current_record: dict[str, object] = {
                "split": audit_row["split"],
                "label": audit_row["label"],
                "sample_folder": audit_row[
                    "sample_folder"
                ],
                "brand": audit_row.get(
                    "brand",
                    "",
                ),
                "screenshot_path": str(image_path),
                "sha256": audit_row.get(
                    "sha256",
                    "",
                ),
                "phash": phash_string,
                "dhash": dhash_string,
                "phash_integer": phash_integer,
                "dhash_integer": dhash_integer,
            }

            # Find previous images with a similar pHash.
            possible_matches = hash_tree.query(
                phash_integer,
                PHASH_DISTANCE_THRESHOLD,
            )

            for (
                previous_record_index,
                phash_distance,
            ) in possible_matches:

                previous_record = hash_records[
                    previous_record_index
                ]

                dhash_distance = hamming_distance(
                    dhash_integer,
                    int(
                        previous_record[
                            "dhash_integer"
                        ]
                    ),
                )

                # Both hash checks must pass.
                if (
                    dhash_distance
                    > DHASH_DISTANCE_THRESHOLD
                ):
                    continue

                # Exact SHA-256 duplicates were already reported
                # by the first audit.
                if (
                    current_record["sha256"]
                    and current_record["sha256"]
                    == previous_record["sha256"]
                ):
                    continue

                cross_split = (
                    current_record["split"]
                    != previous_record["split"]
                )

                cross_label = (
                    current_record["label"]
                    != previous_record["label"]
                )

                pair_count += 1

                if cross_split:
                    cross_split_pair_count += 1

                if cross_label:
                    cross_label_pair_count += 1

                pair_writer.writerow(
                    {
                        "pair_number": pair_count,
                        "phash_distance": (
                            phash_distance
                        ),
                        "dhash_distance": (
                            dhash_distance
                        ),
                        "first_split": (
                            previous_record["split"]
                        ),
                        "first_label": (
                            previous_record["label"]
                        ),
                        "first_sample": (
                            previous_record[
                                "sample_folder"
                            ]
                        ),
                        "first_sha256": (
                            previous_record["sha256"]
                        ),
                        "first_path": (
                            previous_record[
                                "screenshot_path"
                            ]
                        ),
                        "second_split": (
                            current_record["split"]
                        ),
                        "second_label": (
                            current_record["label"]
                        ),
                        "second_sample": (
                            current_record[
                                "sample_folder"
                            ]
                        ),
                        "second_sha256": (
                            current_record["sha256"]
                        ),
                        "second_path": (
                            current_record[
                                "screenshot_path"
                            ]
                        ),
                        "cross_split": cross_split,
                        "cross_label": cross_label,
                    }
                )

                union_find.union(
                    current_record_index,
                    previous_record_index,
                )

            hash_records.append(current_record)

            hash_tree.add(
                phash_integer,
                current_record_index,
            )

            if (
                audit_number % 500 == 0
                or audit_number == total_images
            ):
                print(
                    f"Processed "
                    f"{audit_number:,}/{total_images:,} images"
                )

    # ========================================================================
    # SAVE CALCULATED HASHES
    # ========================================================================

    hash_report_rows: list[dict[str, object]] = []

    for record in hash_records:
        hash_report_rows.append(
            {
                "split": record["split"],
                "label": record["label"],
                "sample_folder": record[
                    "sample_folder"
                ],
                "brand": record["brand"],
                "screenshot_path": record[
                    "screenshot_path"
                ],
                "sha256": record["sha256"],
                "phash": record["phash"],
                "dhash": record["dhash"],
            }
        )

    hash_report_path = (
        REPORTS_FOLDER
        / "phish360_perceptual_hashes.csv"
    )

    hash_fieldnames = [
        "split",
        "label",
        "sample_folder",
        "brand",
        "screenshot_path",
        "sha256",
        "phash",
        "dhash",
    ]

    write_csv(
        hash_report_path,
        hash_report_rows,
        hash_fieldnames,
    )

    # ========================================================================
    # CREATE CONNECTED NEAR-DUPLICATE GROUPS
    # ========================================================================

    connected_groups: dict[int, list[int]] = defaultdict(list)

    for record_index in range(len(hash_records)):
        root_index = union_find.find(record_index)

        connected_groups[root_index].append(
            record_index
        )

    near_duplicate_groups = [
        member_indexes
        for member_indexes in connected_groups.values()
        if len(member_indexes) > 1
    ]

    near_duplicate_groups.sort(
        key=len,
        reverse=True,
    )

    group_report_rows: list[dict[str, object]] = []

    cross_split_group_count = 0
    cross_label_group_count = 0

    for group_number, member_indexes in enumerate(
        near_duplicate_groups,
        start=1,
    ):
        group_splits = sorted(
            {
                str(hash_records[index]["split"])
                for index in member_indexes
            }
        )

        group_labels = sorted(
            {
                str(hash_records[index]["label"])
                for index in member_indexes
            }
        )

        crosses_splits = len(group_splits) > 1
        crosses_labels = len(group_labels) > 1

        if crosses_splits:
            cross_split_group_count += 1

        if crosses_labels:
            cross_label_group_count += 1

        for record_index in member_indexes:
            record = hash_records[record_index]

            group_report_rows.append(
                {
                    "near_duplicate_group": (
                        group_number
                    ),
                    "group_size": len(
                        member_indexes
                    ),
                    "group_splits": "|".join(
                        group_splits
                    ),
                    "group_labels": "|".join(
                        group_labels
                    ),
                    "cross_split": crosses_splits,
                    "cross_label": crosses_labels,
                    "split": record["split"],
                    "label": record["label"],
                    "sample_folder": record[
                        "sample_folder"
                    ],
                    "brand": record["brand"],
                    "phash": record["phash"],
                    "dhash": record["dhash"],
                    "sha256": record["sha256"],
                    "screenshot_path": record[
                        "screenshot_path"
                    ],
                }
            )

    group_report_path = (
        REPORTS_FOLDER
        / "phish360_near_duplicate_groups.csv"
    )

    group_fieldnames = [
        "near_duplicate_group",
        "group_size",
        "group_splits",
        "group_labels",
        "cross_split",
        "cross_label",
        "split",
        "label",
        "sample_folder",
        "brand",
        "phash",
        "dhash",
        "sha256",
        "screenshot_path",
    ]

    write_csv(
        group_report_path,
        group_report_rows,
        group_fieldnames,
    )

    # ========================================================================
    # SAVE HASH FAILURES
    # ========================================================================

    failure_report_path = (
        REPORTS_FOLDER
        / "phish360_perceptual_hash_failures.csv"
    )

    failure_fieldnames = [
        "split",
        "label",
        "sample_folder",
        "screenshot_path",
        "error",
    ]

    write_csv(
        failure_report_path,
        hash_failures,
        failure_fieldnames,
    )

    images_in_groups = sum(
        len(group)
        for group in near_duplicate_groups
    )

    largest_group_size = max(
        (
            len(group)
            for group in near_duplicate_groups
        ),
        default=0,
    )

    # ========================================================================
    # CREATE SUMMARY
    # ========================================================================

    summary_lines = [
        "=" * 80,
        "PHISH360 NEAR-DUPLICATE AUDIT SUMMARY",
        "=" * 80,
        (
            "Audit time: "
            + datetime.now().isoformat(
                timespec="seconds"
            )
        ),
        f"Valid audit images: {total_images:,}",
        (
            "Successfully hashed images: "
            f"{len(hash_records):,}"
        ),
        (
            "Hash calculation failures: "
            f"{len(hash_failures):,}"
        ),
        "",
        "THRESHOLDS",
        "-" * 80,
        (
            "pHash maximum distance: "
            f"{PHASH_DISTANCE_THRESHOLD}"
        ),
        (
            "dHash maximum distance: "
            f"{DHASH_DISTANCE_THRESHOLD}"
        ),
        "",
        "NEAR-DUPLICATE RESULTS",
        "-" * 80,
        (
            "Near-duplicate pairs: "
            f"{pair_count:,}"
        ),
        (
            "Pairs crossing trainval/test: "
            f"{cross_split_pair_count:,}"
        ),
        (
            "Pairs crossing labels: "
            f"{cross_label_pair_count:,}"
        ),
        (
            "Near-duplicate groups: "
            f"{len(near_duplicate_groups):,}"
        ),
        (
            "Images inside groups: "
            f"{images_in_groups:,}"
        ),
        (
            "Largest group size: "
            f"{largest_group_size:,}"
        ),
        (
            "Groups crossing trainval/test: "
            f"{cross_split_group_count:,}"
        ),
        (
            "Groups crossing labels: "
            f"{cross_label_group_count:,}"
        ),
        "",
        "REPORT FILES",
        "-" * 80,
        str(hash_report_path),
        str(near_duplicate_pairs_path),
        str(group_report_path),
        str(failure_report_path),
        "",
        "IMPORTANT",
        "-" * 80,
        (
            "Near-duplicate results are warnings and must "
            "be reviewed before exclusions are made."
        ),
        (
            "No screenshots or original dataset files were "
            "deleted, moved, renamed, resized or modified."
        ),
        "=" * 80,
    ]

    summary_text = "\n".join(summary_lines)

    summary_report_path = (
        REPORTS_FOLDER
        / "phish360_near_duplicate_summary.txt"
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