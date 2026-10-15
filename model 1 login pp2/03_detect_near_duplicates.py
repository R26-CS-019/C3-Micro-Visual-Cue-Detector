from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


# =============================================================================
# SETTINGS
# =============================================================================

DATASET_ROOT = Path(
    r"C:\Users\oshan\Desktop\Research project\datasets\dataset\login dataset"
)

MANIFEST_PATH = Path(
    r"C:\Users\oshan\Desktop\model 1 login pp2"
    r"\reports\01_dataset_audit\image_manifest.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\oshan\Desktop\model 1 login pp2"
    r"\reports\03_near_duplicates"
)

PHASH_THRESHOLD = 6
DHASH_THRESHOLD = 10

PHASH_SIZE = 8
PHASH_IMAGE_SIZE = 32
DHASH_SIZE = 8


# =============================================================================
# UNION-FIND
# =============================================================================

class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]

        return item

    def union(self, item_a: int, item_b: int) -> None:
        root_a = self.find(item_a)
        root_b = self.find(item_b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


# =============================================================================
# HASH FUNCTIONS
# =============================================================================

def create_dct_matrix(size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.float32)

    for frequency in range(size):
        if frequency == 0:
            scale = np.sqrt(1.0 / size)
        else:
            scale = np.sqrt(2.0 / size)

        for position in range(size):
            matrix[frequency, position] = (
                scale
                * np.cos(
                    (
                        np.pi
                        * (2 * position + 1)
                        * frequency
                    )
                    / (2 * size)
                )
            )

    return matrix


DCT_MATRIX = create_dct_matrix(PHASH_IMAGE_SIZE)


def boolean_array_to_integer(bits: np.ndarray) -> int:
    value = 0

    for bit in bits.reshape(-1):
        value = (value << 1) | int(bool(bit))

    return value


def calculate_phash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize(
        (PHASH_IMAGE_SIZE, PHASH_IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )

    pixel_array = np.asarray(
        grayscale,
        dtype=np.float32,
    )

    dct_result = (
        DCT_MATRIX
        @ pixel_array
        @ DCT_MATRIX.T
    )

    low_frequency = dct_result[
        :PHASH_SIZE,
        :PHASH_SIZE,
    ]

    median_value = np.median(low_frequency)
    hash_bits = low_frequency > median_value

    return boolean_array_to_integer(hash_bits)


def calculate_dhash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize(
        (DHASH_SIZE + 1, DHASH_SIZE),
        Image.Resampling.LANCZOS,
    )

    pixel_array = np.asarray(
        grayscale,
        dtype=np.uint8,
    )

    difference = (
        pixel_array[:, 1:]
        > pixel_array[:, :-1]
    )

    return boolean_array_to_integer(difference)


def hamming_distance(hash_a: int, hash_b: int) -> int:
    return (hash_a ^ hash_b).bit_count()


# =============================================================================
# DATA LOADING
# =============================================================================

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


def calculate_hashes(
    rows: list[dict[str, str]],
) -> tuple[list[int], list[int]]:
    phashes: list[int] = []
    dhashes: list[int] = []

    total_images = len(rows)

    print("Calculating perceptual hashes...")

    for index, row in enumerate(rows, start=1):
        image_path = DATASET_ROOT / row["relative_path"]

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")

            phashes.append(calculate_phash(image))
            dhashes.append(calculate_dhash(image))

        if index % 250 == 0 or index == total_images:
            print(
                f"Hashed {index}/{total_images} images"
            )

    return phashes, dhashes


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("MODEL 1 NEAR-DUPLICATE DETECTION")
    print("=" * 80)
    print(f"pHash threshold : {PHASH_THRESHOLD}")
    print(f"dHash threshold : {DHASH_THRESHOLD}")
    print()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest was not found:\n{MANIFEST_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_manifest()
    total_images = len(rows)

    phashes, dhashes = calculate_hashes(rows)

    hash_manifest_path = OUTPUT_DIR / "perceptual_hash_manifest.csv"

    hash_columns = [
        "class_label",
        "relative_path",
        "phash",
        "dhash",
    ]

    with hash_manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=hash_columns,
        )

        writer.writeheader()

        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "class_label": row["class_label"],
                    "relative_path": row["relative_path"],
                    "phash": f"{phashes[index]:016x}",
                    "dhash": f"{dhashes[index]:016x}",
                }
            )

    total_comparisons = (
        total_images * (total_images - 1)
    ) // 2

    print()
    print(
        f"Comparing {total_comparisons:,} image pairs..."
    )

    union_find = UnionFind(total_images)

    pair_columns = [
        "pair_id",
        "class_a",
        "path_a",
        "class_b",
        "path_b",
        "phash_distance",
        "dhash_distance",
        "cross_class",
    ]

    pairs_path = OUTPUT_DIR / "near_duplicate_pairs.csv"
    cross_class_pairs_path = (
        OUTPUT_DIR / "cross_class_near_duplicate_pairs.csv"
    )

    accepted_pair_count = 0
    same_class_pair_count = 0
    cross_class_pair_count = 0

    comparison_start = time.time()

    with (
        pairs_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as all_pairs_file,
        cross_class_pairs_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as cross_pairs_file,
    ):
        all_pairs_writer = csv.DictWriter(
            all_pairs_file,
            fieldnames=pair_columns,
        )

        cross_pairs_writer = csv.DictWriter(
            cross_pairs_file,
            fieldnames=pair_columns,
        )

        all_pairs_writer.writeheader()
        cross_pairs_writer.writeheader()

        for index_a in range(total_images - 1):
            phash_a = phashes[index_a]
            dhash_a = dhashes[index_a]

            for index_b in range(
                index_a + 1,
                total_images,
            ):
                phash_distance = hamming_distance(
                    phash_a,
                    phashes[index_b],
                )

                if phash_distance > PHASH_THRESHOLD:
                    continue

                dhash_distance = hamming_distance(
                    dhash_a,
                    dhashes[index_b],
                )

                if dhash_distance > DHASH_THRESHOLD:
                    continue

                accepted_pair_count += 1

                class_a = rows[index_a]["class_label"]
                class_b = rows[index_b]["class_label"]

                is_cross_class = class_a != class_b

                if is_cross_class:
                    cross_class_pair_count += 1
                else:
                    same_class_pair_count += 1

                pair_row = {
                    "pair_id": (
                        f"PAIR_{accepted_pair_count:06d}"
                    ),
                    "class_a": class_a,
                    "path_a": rows[index_a]["relative_path"],
                    "class_b": class_b,
                    "path_b": rows[index_b]["relative_path"],
                    "phash_distance": phash_distance,
                    "dhash_distance": dhash_distance,
                    "cross_class": is_cross_class,
                }

                all_pairs_writer.writerow(pair_row)

                if is_cross_class:
                    cross_pairs_writer.writerow(pair_row)

                union_find.union(index_a, index_b)

            if (
                (index_a + 1) % 250 == 0
                or index_a == total_images - 2
            ):
                print(
                    f"Compared base image "
                    f"{index_a + 1}/{total_images - 1}"
                )

    # =========================================================================
    # BUILD CONNECTED GROUPS
    # =========================================================================

    components: dict[int, list[int]] = defaultdict(list)

    for image_index in range(total_images):
        root = union_find.find(image_index)
        components[root].append(image_index)

    groups = [
        members
        for members in components.values()
        if len(members) > 1
    ]

    groups.sort(
        key=lambda members: (
            -len(members),
            rows[members[0]]["relative_path"],
        )
    )

    group_columns = [
        "group_id",
        "group_size",
        "cross_class_group",
        "class_label",
        "relative_path",
        "phash",
        "dhash",
    ]

    group_rows: list[dict[str, object]] = []

    cross_class_group_count = 0
    grouped_image_count = 0
    largest_group_size = 0

    for group_number, members in enumerate(
        groups,
        start=1,
    ):
        group_id = f"NEAR_DUP_GROUP_{group_number:04d}"

        group_classes = {
            rows[index]["class_label"]
            for index in members
        }

        is_cross_class_group = len(group_classes) > 1

        if is_cross_class_group:
            cross_class_group_count += 1

        grouped_image_count += len(members)
        largest_group_size = max(
            largest_group_size,
            len(members),
        )

        for image_index in members:
            group_rows.append(
                {
                    "group_id": group_id,
                    "group_size": len(members),
                    "cross_class_group": is_cross_class_group,
                    "class_label": (
                        rows[image_index]["class_label"]
                    ),
                    "relative_path": (
                        rows[image_index]["relative_path"]
                    ),
                    "phash": (
                        f"{phashes[image_index]:016x}"
                    ),
                    "dhash": (
                        f"{dhashes[image_index]:016x}"
                    ),
                }
            )

    groups_path = OUTPUT_DIR / "near_duplicate_groups.csv"

    with groups_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=group_columns,
        )

        writer.writeheader()
        writer.writerows(group_rows)

    elapsed_seconds = time.time() - comparison_start

    summary_lines = [
        "=" * 80,
        "MODEL 1 NEAR-DUPLICATE SUMMARY",
        "=" * 80,
        "",
        f"Total valid images             : {total_images}",
        f"Total pair comparisons         : {total_comparisons:,}",
        f"pHash threshold                : {PHASH_THRESHOLD}",
        f"dHash threshold                : {DHASH_THRESHOLD}",
        "",
        f"Accepted near-duplicate pairs  : {accepted_pair_count}",
        f"Same-class pairs               : {same_class_pair_count}",
        f"Cross-class pairs              : {cross_class_pair_count}",
        "",
        f"Connected duplicate groups     : {len(groups)}",
        f"Cross-class groups             : {cross_class_group_count}",
        f"Images inside groups           : {grouped_image_count}",
        f"Largest group size             : {largest_group_size}",
        "",
        f"Pair comparison time           : {elapsed_seconds:.2f} seconds",
        "",
        "IMPORTANT:",
        "Candidate pairs are not automatically deleted.",
        "Visual review is required before making decisions.",
        "",
        "No original screenshot was deleted, moved, resized,",
        "renamed, or modified.",
        "=" * 80,
    ]

    summary_text = "\n".join(summary_lines)

    summary_path = OUTPUT_DIR / "near_duplicate_summary.txt"
    summary_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    print()
    print(summary_text)
    print()
    print(f"Reports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()