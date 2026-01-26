import argparse
import base64
import json
import math
import numpy as np
import pandas as pd
import random
import time
import torch
import torch.nn.functional as F

from collections import defaultdict
from gurobipy import *
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from valentine import valentine_match
from valentine.algorithms import Coma, Cupid, DistributionBased, JaccardDistanceMatcher, SimilarityFlooding

VALENTINE_MATCHERS = {
    "coma": lambda: Coma(java_xmx="4g"),
    "cupid": lambda: Cupid(),
    "jaccard": lambda: JaccardDistanceMatcher(),
    "distribution": lambda: DistributionBased(),
    "similarityflooding": lambda: SimilarityFlooding()
}

# ----------------------------
# Type compatibility functions for Jasper
# ----------------------------
def are_types_compatible(type1, type2):
    """
    Check if two JSON types are semantically compatible.

    Args:
        type1 (str): First type (e.g., 'object', 'string', 'number', 'array', 'boolean', 'null').
        type2 (str): Second type.

    Returns:
        bool: True if compatible, False otherwise.
    """

    # Exact match
    if type1 == type2:
        return True

    # Allow null to match anything
    if type1 == "null" or type2 == "null":
        return True

    # Allow integer ≈ number
    if (type1 == "integer" and type2 == "number") or (type1 == "number" and type2 == "integer"):
        return True

    # Prevent incompatible scalar types
    scalar_incompatibles = {
        ("string", "number"), ("number", "string"),
        ("string", "boolean"), ("boolean", "string"),
        ("number", "boolean"), ("boolean", "number"),
    }
    if (type1, type2) in scalar_incompatibles or (type2, type1) in scalar_incompatibles:
        return False

    # Prevent incompatible structural types
    structural_incompatibles = {
        ("array", "string"),  ("string", "array"),
        ("array", "number"),  ("number", "array"),
        ("array", "boolean"), ("boolean", "array")
    }
    if (type1, type2) in structural_incompatibles or (type2, type1) in structural_incompatibles:
        return False

    return False


# --------------------------------------------
# Linguistic similarity functions for Jasper
# --------------------------------------------
def combined_embedding(path_emb, value_emb, alpha=0.6):
    """
    Combine path and value embeddings using a weighted sum.

    Args:
        path_emb (torch.Tensor): Embedding for the JSON path.
        value_emb (torch.Tensor): Embedding for the value at the path.
        alpha (float): Weight for path embedding.

    Returns:
        torch.Tensor: Combined embedding.
    """
    if path_emb is None and value_emb is None:
        return None
    elif path_emb is None:
        return value_emb
    elif value_emb is None:
        return path_emb
    return (alpha * path_emb + (1-alpha) * value_emb) / \
           (alpha * path_emb + (1-alpha) * value_emb).norm()



# ----------------------------
# Structural similarity functions for Jasper
# ----------------------------
def weighted_depth_similarity(source_path_keys, target_path_keys, penalty_exponent=2):
    """
    Compute depth similarity with a bigger penalty for larger differences.

    Args:
        source_path_keys (list): Source path keys.
        target_path_keys (list): Target path keys.
        penalty_exponent (int): Exponent to increase penalty for depth differences.

    Returns:
        float: Depth similarity score (0 to 1).
    """
    len1, len2 = len(source_path_keys), len(target_path_keys)
    depth_diff = abs(len1 - len2)
    max_depth = max(len1, len2)
    return 1 - (depth_diff / max_depth) ** penalty_exponent

def sibling_similarity(source_num_siblings, target_num_siblings):
    """
    Compare the number of sibling keys at the same level in two paths.

    Args:
        source_num_siblings (int): The number of sibling keys at the same level for the source path.
        target_num_siblings (int): The number of sibling keys at the same level for the target path.

    Returns:
        float: Similarity score (0 to 1).
    """
    diff = abs(source_num_siblings - target_num_siblings)
    return 1 / (1 + diff)

def key_entropy_similarity(source_entropy, target_entropy):
    """
    Compare the key entropy of two paths to measure the variability in their nested structures.
    A lower difference in entropy indicates more similarity in the structure.

    Args:
        source_entropy (float): The key entropy value at the source path, representing the variability of keys.
        target_entropy (float): The key entropy value at the target path, representing the variability of keys.

    Returns:
        float: Similarity score between 0 and 1.
    """
    diff = abs(source_entropy - target_entropy)
    return 1 / (1 + diff)

def get_structural_similarity(s_row, t_row, w_depth=0.25, w_sibling=0.5, w_entropy=0.25):
    """
    Compute structural similarity between two paths based on their statistics.
    
    Args:
        s_row (pd.Series): Statistics for source path.
        t_row (pd.Series): Statistics for target path.
        w_depth, w_sibling, w_entropy (float): Weights for each component.
    Returns:
        float: Structural similarity score between 0 and 1.
    """
    source_path_keys = s_row["path"].split('.')
    target_path_keys = t_row["path"].split('.')

    depth_sim = weighted_depth_similarity(source_path_keys, target_path_keys)

    sibling_sim = sibling_similarity(
        s_row.get("num_siblings", 0),
        t_row.get("num_siblings", 0)
    )

    entropy_sim = key_entropy_similarity(
        s_row.get("key_entropy", 0.0),
        t_row.get("key_entropy", 0.0)
    )

    return (w_depth * depth_sim + w_sibling * sibling_sim + w_entropy * entropy_sim)


# ----------------------------
# Jasper matching functions
# ----------------------------

def decode_embedding(b64_string, dim=None):
    """
    Decode a base64-encoded embedding back to a numpy array.

    Args:
        b64_string (str): Base64 string of float32 array.
        dim (int, optional): dimension of embedding. Needed if you want to reshape.
    Returns:
        np.ndarray
    """
    if not b64_string:
        if dim is not None:
            return np.zeros(dim, dtype=np.float32)
        else:
            return np.array([], dtype=np.float32)
    
    # Decode from base64 to bytes, then interpret as float32
    byte_data = base64.b64decode(b64_string)
    arr = np.frombuffer(byte_data, dtype=np.float32)
    
    if dim is not None:
        arr = arr.reshape(dim)
    
    return arr

def compute_combined_embeddings(df, device):
    """
    Decode and combine path/value embeddings for a DataFrame.
    Returns a normalized tensor of shape [N, d].

    Args:
        df (pd.DataFrame): DataFrame with 'path_emb' and 'values_emb' columns.
        device (str): Device to place the tensor on.
    Returns:
        torch.Tensor: Normalized combined embeddings.
    """
    embeddings = []

    for row in df.itertuples(index=False):
        path_emb = torch.tensor(
            decode_embedding(row.path_emb), dtype=torch.float32
        )
        value_emb = torch.tensor(
            decode_embedding(row.values_emb), dtype=torch.float32
        )
        combined = combined_embedding(path_emb, value_emb)
        embeddings.append(combined)

    emb = torch.stack(embeddings).to(device)
    emb = torch.nn.functional.normalize(emb, dim=1)
    return emb

def match_paths(source_df, target_df, ling_weight=0.3, struct_weight=0.7, min_score=0.7, device="cuda"):
    """
    Match paths from two sets using precomputed embeddings and structural similarity.
    Args:
        source_df (pd.DataFrame): Source dataset.
        target_df (pd.DataFrame): Target dataset.
        ling_weight (float): Weight for linguistic similarity.
        struct_weight (float): Weight for structural similarity.
        min_score (float): Minimum score threshold to consider a match.
        device (str): Device to use for tensor computations.
    Returns:
        dict: {source_path: [(target_path, score), ...]}
    """

    source_df = source_df.copy()
    target_df = target_df.copy()
    source_df["path"] = source_df["path"].astype(str)
    target_df["path"] = target_df["path"].astype(str)

    # Precompute combined embeddings
    source_emb = compute_combined_embeddings(source_df, device)
    target_emb = compute_combined_embeddings(target_df, device)

    # Calculate linguistic similarity matrix
    ling_sim = source_emb @ target_emb.T

    # Structural similarity + type compatibility
    source_rows = list(source_df.itertuples(index=False))
    target_rows = list(target_df.itertuples(index=False))

    matches = defaultdict(list)

    for i, s_row in enumerate(source_rows):
        for j, t_row in enumerate(target_rows):

            # Check type compatibility
            if not are_types_compatible(s_row.types, t_row.types):
                continue

            # Get the linguistic score from precomputed matrix
            ling_score = float(ling_sim[i, j])

            # Get structural similarity score
            struct_score = get_structural_similarity(
                s_row._asdict(),
                t_row._asdict(),
            )

            # Combine scores
            final_score = ling_weight * ling_score + struct_weight * struct_score
            if final_score >= min_score:
                matches[s_row.path].append((t_row.path, final_score))

    return matches

def prune_top_k_candidates(candidate_matches, top_k=5):
    """
    Keep only the top-k scoring targets per source path.

    Args:
        candidate_matches (dict): {source_path: [(target_path, score), ...]}
        top_k (int): Number of candidates to keep per source.

    Returns:
        dict: Pruned candidate matches.
    """
    pruned = {}
    for s_path, targets in candidate_matches.items():
        pruned[s_path] = sorted(targets, key=lambda x: -x[1])[:top_k]
    return pruned

def parent_path(path):
    if "." not in path:
        return None
    return path.rsplit(".", 1)[0]

def refine_scores(match_dict, alpha=0.2):
    """
    Adjust scores with nesting consistency.

    Args:
        match_dict: {source: [(target, score)]}
        alpha: bonus for parent match

    Returns:
        {(source, target): adjusted_score}
    """
    # Compute best target per source
    best_target = {}
    for s, tgts in match_dict.items():
        if tgts:
            best_target[s] = max(tgts, key=lambda x: x[1])[0]

    refined = {}

    for s, tgts in match_dict.items():
        s_parent = parent_path(s)

        for t, score in tgts:
            t_parent = parent_path(t)
            bonus = 0.0
            if s_parent and t_parent and best_target.get(s_parent) == t_parent:
                bonus = alpha

            refined[(s, t)] = score + bonus

    return refined

def select_top_k_matches(refined_pairs, top_k=1):
    """
    Select up to top-k targets for each source.

    Args:
        refined_pairs: {(source, target): score}
        top_k: number of top targets per source

    Returns:
        {(source, target): score}
    """

    per_source = defaultdict(list)

    # group by source
    for (s, t), score in refined_pairs.items():
        per_source[s].append((t, score))

    top_k_matches = {}

    # select top-k per source
    for s, tgts in per_source.items():
        tgts.sort(key=lambda x: x[1], reverse=True)
        for t, score in tgts[:top_k]:
            top_k_matches[(s, t)] = score

    return top_k_matches
    
def final_match(pruned_pairs, alpha=0.2, top_k=1):
    """
    Final matching with score refinement and top-k selection.

    Args:
        pruned_pairs: {source: [(target, score)]}
        alpha: nesting bonus
        top_k: max targets per source
    Returns:
        {(source, target): adjusted_score}
    """
    refined = refine_scores(pruned_pairs, alpha=alpha)
    return select_top_k_matches(refined, top_k=top_k)

    

# ----------------------------
# Step 1: Load SOURCE and TARGET datasets
# ----------------------------
def path_dict_to_df(path_dict):
    """
    Convert a path_dict to a DataFrame.

    Args:
        path_dict (dict): Dictionary of paths and their corresponding values.
    Returns:
        pd.DataFrame: A DataFrame with paths as columns and values as rows.
    """
    # Convert sets to lists and take the first element if there are any values
    flat_dict = {path: list(values)[0] if values else None for path, values in path_dict.items()}
    return pd.DataFrame([flat_dict])

def load_dataset(path):
    return pd.read_csv(path, delimiter=";")

# -----------------------------
# Step 2: Select datasets to evaluate
# -----------------------------
def filter_dataset_by_size(source_groups, target_groups, min_paths, max_paths):
    """
    Get sources and targets whose number of paths is between min_paths and max_paths (inclusive).

    Args:
        source_groups (dict): {filename: DataFrame}
        target_groups (dict): {filename: DataFrame}
        min_paths (int): Minimum number of paths required.
        max_paths (int): Maximum number of paths allowed.

    Returns:
        list: Filenames that meet the criteria.
    """
    return [
        fn for fn in source_groups.keys()
        if (min_paths <= len(source_groups[fn]) <= max_paths) and
           (min_paths <= len(target_groups[fn]) <= max_paths)
    ]

def compute_max_depth(source_df, target_df):
    """
    Compute the maximum depth of paths in both source and target DataFrames.

    Args:
        source_df (pd.DataFrame): Source DataFrame.
        target_df (pd.DataFrame): Target DataFrame.

    Returns:
        int: Maximum depth of paths.
    """
    max_source = source_df["path"].apply(lambda x: len(str(x).split('.'))).max()
    max_target = target_df["path"].apply(lambda x: len(str(x).split('.'))).max()

    return max(max_source, max_target)

def group_datasets_by_depth(filenames, source_groups, target_groups, n_bins=3):
    """
    Assign each filename to a depth bin: shallow, medium, deep.
    
    Args:
        filenames (list): List of filenames to stratify.
        source_groups (dict): {filename: DataFrame}
        target_groups (dict): {filename: DataFrame}
        n_bins (int): Number of depth bins.
    Returns:
        dict: {bin_index: [filenames]}
    """
    dataset_depths = {fn: compute_max_depth(source_groups[fn], target_groups[fn]) for fn in filenames}
    min_d, max_d = min(dataset_depths.values()), max(dataset_depths.values())
    bin_size = (max_d - min_d) / n_bins
    bins = {i: [] for i in range(n_bins)}
    for fn, depth in dataset_depths.items():
        bin_idx = min(int((depth - min_d) / bin_size), n_bins - 1)
        bins[bin_idx].append(fn)
    return bins

def sample_datasets_from_bins(bins, total_sample):
    """
    Randomly sample total_sample filenames proportionally from bins.

    Args:
        bins (dict): {bin_index: [filenames]}
        total_sample (int): Total number of filenames to sample.
    Returns:
        list: Sampled filenames.
    """
    n_bins = len(bins)
    per_bin = math.ceil(total_sample / n_bins)
    selected = []
    for bin_files in bins.values():
        if bin_files:
            n_sample = min(per_bin, len(bin_files))
            selected.extend(random.sample(bin_files, n_sample))
    return selected




def embed_mean(x):
    """Compute mean embedding from a list of embeddings."""
    return np.mean(np.vstack([decode_embedding(v) for v in x]), axis=0)

def find_best_k(X, k_min=3, k_max=10):
    """
    Find the best k for KMeans clustering using silhouette score.

    Args:
        X (np.ndarray): Feature matrix.
        k_min (int): Minimum number of clusters to try.
        k_max (int): Maximum number of clusters to try.
    Returns:
        tuple: (best_k, scores) where best_k is the optimal number of clusters and scores is a dict of {k: silhouette_score}.
    """
    best_k = None
    best_score = -1
    scores = {}

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score

        if score > best_score:
            best_k = k
            best_score = score

    return best_k, scores

def sample_datasets(source_dir, sample_fraction):
    """
    Cluster datasets and sample a fraction of them while preserving 
    variation in dataset size and complexity.

    Args:
        source_dir (str): Directory containing source CSV files.
        sample_fraction (float): Fraction of datasets to sample.

    Returns:
        list: Sampled dataset filenames.
    """

    source_dir = Path(source_dir)
    csv_files = sorted(source_dir.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in: {source_dir}")

    # Load all CSVs
    dfs = [pd.read_csv(f, delimiter=";") for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    # Aggregate statistics per dataset
    agg = df.groupby("filename").agg({
        "nesting_depth": ["mean", "max"],
        "num_children": "mean",
        "num_siblings": "mean",
        "key_entropy": "mean",
        "freq": "sum",
    })
    agg.columns = ["_".join(col) for col in agg.columns]
    agg.reset_index(inplace=True)

    # Compute average embedding
    agg["avg_emb"] = df.groupby("filename")["path_emb"].apply(embed_mean).values

    # Build feature matrix
    numeric_cols = [
        "nesting_depth_mean", "nesting_depth_max",
        "num_children_mean", "num_siblings_mean",
        "key_entropy_mean", "freq_sum"
    ]
    numeric_features = agg[numeric_cols].values
    emb_features = np.vstack(agg["avg_emb"].values)
    X = np.hstack([numeric_features, emb_features])

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find the best K and cluster
    best_k, scores = find_best_k(X_scaled)
    print(f"Best number of clusters = {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    agg["cluster"] = kmeans.fit_predict(X_scaled)

    # Sample datasets from each cluster
    sample_size = int(len(agg) * sample_fraction)
    print(f"Sampling {sample_size} datasets (~{sample_fraction*100:.0f}%)")

    samples = []

    for cluster in range(best_k):
        cluster_rows = agg[agg["cluster"] == cluster]
        cluster_n = len(cluster_rows)

        # Sample proportional to cluster size
        take = max(1, int(cluster_n * sample_fraction))
        chosen = cluster_rows.sample(n=take, random_state=42)["filename"].tolist()
        samples.extend(chosen)

    return samples

    

# ----------------------------
# Step 3: Apply matching algorithm
# ----------------------------
def reformat_valentine_matches(valentine_matches):
    """
    Remove the first element of the paths in Valentine matches.
    It's necessary because Valentine returns paths with a labelled "source" or "target" at the beginning.

    Args:
        valentine_matches (dict): {(source_path,(target_path): score)}

    Returns:
        dict: {(source_path, target_path): score}
    """
    matches = defaultdict(list)

    for (tgt, src), score in valentine_matches.items():
        new_tgt = ".".join(tgt[1:])
        new_src = ".".join(src[1:])
        matches[(new_src, new_tgt)].append(score)

    return matches

# ----------------------------
# Step 4: Evaluate matches against ground truth
# ----------------------------
def get_ground_truth_pairs(ground_truth_path, filename):
    """
    Load all (source_path, target_path) pairs from ground truth JSON file.

    Args:
        ground_truth_path (str or Path): Path to ground truth JSON file.
        filename (str): Filename to filter ground truth mappings.

    Returns:
        set: Set of (source_path, target_path) tuples.
    """
    gt = set()
    with open(ground_truth_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mapping = json.loads(line)
            if mapping.get("filename") != filename:
                continue
            src = mapping.get("original_path")
            tgt = mapping.get("transformed_path")
            if src is not None and tgt is not None:
                gt.add((src, tgt))
    return gt

def evaluate_matches(matches, ground_truth_pairs):
    """
    Evaluate predicted matches against ground truth from a JSON file,
    restricted to the source paths found in `matches`.

    Args:
        matches (dict): Predicted matches in the form {(source_path, target_path): score}.
        ground_truth_pairs (set): Set of (source, target) pairs representing ground truth.

    Returns:
        dict: Evaluation metrics.
    """
    # Get all source-target pairs in predicted matches
    predicted_pairs = set(matches.keys())   

    # Compute intersections and differences
    true_positives = ground_truth_pairs & predicted_pairs    
    false_positives = predicted_pairs - ground_truth_pairs
    false_negatives = ground_truth_pairs - predicted_pairs

    # Print match results
    print("\n--- Match Results ---", flush=True)
    for s, t in true_positives:
        print(f"TRUE  : {s} → {t}", flush=True)
    for s, t in false_positives:
        print(f"FALSE : {s} → {t} (not in ground truth)", flush=True)
    for s, t in false_negatives:
        print(f"MISSED: {s} → {t} (in ground truth but not predicted)", flush=True)

    # Compute metrics
    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }

def evaluate_valentine(source_df, target_df, gt_pairs, matcher_name, matcher_instance):
    """Run Valentine matcher and evaluate."""
    new_source_df = pd.DataFrame(columns=source_df["path"].astype(str))
    new_target_df = pd.DataFrame(columns=target_df["path"].astype(str))

    matches = valentine_match(new_target_df, new_source_df, matcher_instance, "target", "source")
    formatted_matches = reformat_valentine_matches(matches)
    metrics = evaluate_matches(formatted_matches, gt_pairs)
    return metrics

def evaluate_jasper(source_df, target_df, gt_pairs, device):
    """Run Jasper matcher and evaluate."""
    candidate_matches = match_paths(source_df, target_df, ling_weight=0.5, struct_weight=0.5, min_score=0.7, device=device)
    pruned_matches = prune_top_k_candidates(candidate_matches, top_k=5)
    pruned_pairs = {s: [(t, score) for t, score in tgts] for s, tgts in pruned_matches.items()}
    final_matches = final_match(pruned_pairs, alpha=0.2, top_k=1)
    metrics = evaluate_matches(final_matches, gt_pairs)
    metrics.update({"matcher": "JASPER"})
    return metrics


# ----------------------------
# Step 5: CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Find Matches between source and target.")
    parser.add_argument("source_dir", help="Directory with source csv files")
    parser.add_argument("target_dir", help="Directory with target csv files")
    parser.add_argument("groundtruth_file", help="Path to ground truth JSON file.")
    parser.add_argument("mode", choices=["coma", "cupid", "jaccard", "distribution", "similarityflooding", "jasper"], help="Matching algorithm to use.")
    return parser.parse_args()

def main():
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args()

    # Sample datasets
    selected_datasets = sample_datasets(args.source_dir, sample_fraction=1.0)
    print(f"Evaluating {len(selected_datasets)} datasets. Mode: {args.mode}", flush=True)

    # Initialize matcher once
    if args.mode in VALENTINE_MATCHERS:
        matcher_instance = VALENTINE_MATCHERS[args.mode]()
        mode_type = "valentine"
    elif args.mode == "jasper":
        matcher_instance = None
        mode_type = "jasper"
    else:
        raise ValueError("Invalid mode. Choose from Valentine matcher keys or 'jasper'")

    min_paths = 100
    max_paths = 500
    precision_list = []
    recall_list = []
    f1_list = []

    for filename in tqdm(sorted(selected_datasets), desc="Processing datasets"):
        source_path = Path(args.source_dir) / filename.replace(".json", ".csv")
        target_path = Path(args.target_dir) / filename.replace(".json", ".csv")
        source_df = load_dataset(source_path)
        target_df = load_dataset(target_path)
        gt_pairs = get_ground_truth_pairs(args.groundtruth_file, filename)

        #if len(source_df) > min_paths and len(source_df) < max_paths:
        #if len(source_df) <= min_paths:
        if len(source_df) >= max_paths:
            print(f"\nProcessing {filename}: {len(source_df)} → {len(target_df)} paths, {len(gt_pairs)} GT pairs.", flush=True)

            if mode_type == "valentine":
                metrics = evaluate_valentine(source_df, target_df, gt_pairs, args.mode, matcher_instance)
            else:
                metrics = evaluate_jasper(source_df, target_df, gt_pairs, device=device)

            metrics.update({"filename": filename})

            print(f"Metrics for {filename}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1_score']:.2f}", flush=True)
            precision_list.append(metrics['precision'])
            recall_list.append(metrics['recall'])
            f1_list.append(metrics['f1_score'])

    # Print average precision, recall, f1 to nearest 2 decimals
    print(f"\n=== Overall Evaluation  of  {args.mode} ===", flush=True)
    avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0
    avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0
    print(f"Average Precision: {avg_precision:.2f}", flush=True)
    print(f"Average Recall:    {avg_recall:.2f}", flush=True)
    print(f"Average F1 Score:  {avg_f1:.2f}", flush=True)   

    print(f"Execution time: {round(time.time() - start_time, 2)} seconds", flush=True)

if __name__ == "__main__":
    main()



# As k increases, how is f1 score affected
# As k increases during pruning, check how many times we catch the correct answers