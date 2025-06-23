import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from collections import defaultdict



def compute_depth(obj):
    """
    Compute max depth of nested JSON object.

    Args:
        obj (dict or list): JSON object.
    Returns:
        int: Depth of the object.
    """
    if isinstance(obj, dict):
        return 1 + max((compute_depth(v) for v in obj.values()), default=0)
    elif isinstance(obj, list):
        return 1 + max((compute_depth(i) for i in obj), default=0)
    else:
        return 0


def compute_size(obj):
    """
    Compute total number of keys in a JSON object.

    Args:
        obj (dict or list): JSON object.
    Returns:
        int: Size of the object.
    """
    if isinstance(obj, dict):
        return len(obj) + sum(compute_size(v) for v in obj.values())
    elif isinstance(obj, list):
        return sum(compute_size(i) for i in obj)
    else:
        return 0


def get_dataset_features(file_path):
    """
    Get features of a JSON file.

    Args:
        file_path (str): Path to the JSON file.
    Returns:
        dict: Dictionary with depth and size.
    """
    depths = []
    sizes = []
    num_docs = 0

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                num_docs += 1
            except json.JSONDecodeError:
                continue  
            depths.append(compute_depth(doc))
            sizes.append(compute_size(doc))

    if not depths:
        return {"avg_depth": 0, "avg_size": 0, "num_docs": 0}
    
    return {
        "avg_depth": round(sum(depths) / len(depths), 2),
        "avg_size": round(sum(sizes) / len(sizes), 2),
        "num_docs": num_docs
    }


def group_dataset(depth, size):
    """
    Assign a dataset to a depth/size group.

    Args:
        depth (int): Depth of the dataset.
        size (int): Size of the dataset.
    Returns:
        str: Group name.
    """

    if depth <= 2:
        depth_bin = "flat"
    elif depth <= 4:
        depth_bin = "nested"
    else:
        depth_bin = "deep"

    if size <= 20:
        size_bin = "small"
    elif size <= 100:
        size_bin = "medium"
    else:
        size_bin = "large"

    return f"{depth_bin}_{size_bin}"


def stratified_sample(metadata, sample_ratio=0.10, seed=42):
    """
    Sample approximately sample_percent of total datasets,
    stratified proportionally across groups defined by depth and size.

    Args:
        metadata (list): List of dicts with "depth" and "size" keys.
        sample_ratio (float): Fraction (0-1) of total datasets to sample.
        seed (int): Random seed for reproducibility.

    Returns:
        list: Sampled dataset metadata entries.
    """
    random.seed(seed)

    groups = defaultdict(list)
    for entry in metadata:
        group = group_dataset(entry["avg_depth"], entry["avg_size"])
        groups[group].append(entry)

    total_items = len(metadata)
    selected = []

    for g, items in groups.items():
        k = max(1, int(len(items) * sample_ratio))
        k = min(k, len(items))
        sampled = random.sample(items, k)
        selected.extend(sampled)

    print(f"Sampled {len(selected)} datasets (target ≈ {int(total_items * sample_ratio)}).")
    return selected


def main():
    start_time = time.time()

    args = sys.argv[1:]

    if len(args) != 2:
        print("Usage: python sample_data.py <original_dir> <sample_ratio>")
        sys.exit(1)

    original_dir = Path(args[0])
    if not original_dir.exists():
        print(f"Error: {original_dir} does not exist.")
        sys.exit(1)

    sample_ratio = float(args[1])
    if not (0 < sample_ratio <= 1):
        print("Error: sample_ratio must be between 0 and 1.")
        sys.exit(1)

    dataset_metadata = []

    for file in sorted(original_dir.glob("*.json")):
        features = get_dataset_features(file)
        dataset_metadata.append({
            "name": file.stem,
            "avg_depth": features["avg_depth"],
            "avg_size": features["avg_size"],
            "num_docs": features["num_docs"],
        })

    sampled = stratified_sample(dataset_metadata, sample_ratio=sample_ratio, seed=42)

    csv_file = "sampled_datasets_metadata.csv"
    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["name", "avg_depth", "avg_size", "num_docs"])
        writer.writeheader()
        for dataset in sorted(sampled, key=lambda x: x["name"]):
            writer.writerow(dataset)


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Sampled {len(sampled)} datasets in {elapsed_time:.2f} seconds. Written to {csv_file}.")


if __name__ == "__main__":
    main()
