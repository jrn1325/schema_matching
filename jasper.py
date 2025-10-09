import argparse
import ast
import json
import math
import pandas as pd
import random
import sys
import time
import torch
import torch.nn.functional as F


from collections import defaultdict
from gurobipy import *
from scipy.optimize import linprog
from valentine import valentine_match
from valentine.algorithms import Coma, Cupid, DistributionBased, JaccardDistanceMatcher, SimilarityFlooding



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

def get_linguistic_similarity(source_combined_emb, target_combined_emb):
    """
    Compute cosine similarity between two precomputed embeddings.

    Args:
        source_combined_emb (torch.Tensor): Embedding for source path.
        target_combined_emb (torch.Tensor): Embedding for target path.

    Returns:
        float: Cosine similarity score between -1 and 1.
    """
    if source_combined_emb is None or target_combined_emb is None:
        return 0.0
    if not isinstance(source_combined_emb, torch.Tensor):
        source_combined_emb = torch.tensor(source_combined_emb, dtype=torch.float32)
    if not isinstance(target_combined_emb, torch.Tensor):
        target_combined_emb = torch.tensor(target_combined_emb, dtype=torch.float32)

    return F.cosine_similarity(source_combined_emb.unsqueeze(0), target_combined_emb.unsqueeze(0), dim=1).item()


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

def get_structural_similarity(s_row, t_row, w_depth=0.33, w_sibling=0.33, w_entropy=0.33):
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

    
# Jasper matching functions

def match_paths(source_df, target_df, ling_weight=0.5, struct_weight=0.5, min_score=0.7):
    """
    Match paths from two sets using precomputed embeddings and structural similarity.

    Args:
        source_df, target_df (DataFrame): Source and target DataFrames containing path statistics.
        ling_weight (float): Weight for linguistic similarity.
        struct_weight (float): Weight for structural similarity.
        min_score (float): Minimum score threshold to consider a match.

    Returns:
        dict: {source_path: [(target_path, score), ...]}
    """
    matches = defaultdict(list)

    for _, s_row in source_df.iterrows():
        for _, t_row in target_df.iterrows():
            if not are_types_compatible(s_row["types"], t_row["types"]):
                continue

            # Convert stringified embeddings to tensors
            source_path_emb = torch.tensor(ast.literal_eval(s_row["path_emb"]), dtype=torch.float32)
            source_value_emb = torch.tensor(ast.literal_eval(s_row["values_emb"]), dtype=torch.float32)
            target_path_emb = torch.tensor(ast.literal_eval(t_row["path_emb"]), dtype=torch.float32)
            target_value_emb = torch.tensor(ast.literal_eval(t_row["values_emb"]), dtype=torch.float32)

            # Combine embeddings
            source_combined_emb = combined_embedding(source_path_emb, source_value_emb)
            target_combined_emb = combined_embedding(target_path_emb, target_value_emb)

            # Compute similarity
            ling_score = get_linguistic_similarity(source_combined_emb, target_combined_emb)
            struct_score = get_structural_similarity(s_row, t_row)
            final_score = ling_weight * ling_score + struct_weight * struct_score

            if final_score >= min_score:
                matches[s_row["path"]].append((t_row["path"], final_score))

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

def predict_matches(source_vars):
    """
    Selects final match from optimized source_vars.

    Args:
        source_vars (dict): Maps source paths to list of (var, target_path, score).

    Returns:
        dict: {source_path: [(target_path, score)]} — ready for evaluation
    """
    final_matching = {}

    for s_path, var_list in source_vars.items():
        for var, t_path, score in var_list:
            try:
                if hasattr(var, "X") and round(var.X):
                    final_matching[s_path] = [(t_path, score)]
                    break
            except Exception as e:
                print(f"[Error] Reading var.X for {s_path}->{t_path}: {e}")

    # Sort by descending score
    final_matching = dict(sorted(final_matching.items(), key=lambda x: -x[1][0][1]))

    return final_matching

def get_key_prefix(key):
    """
    Get the prefix of a key.

    Args:
        key (str): Target or source key.

    Returns:
        str: The prefix of the key.
    """
    # If no dot in the key, the prefix is just the '$' symbol
    if '.' not in key:
        prefix = '$'
    else:
        prefix, _ = key.rsplit('.', maxsplit=1)  # Split at the last dot
        prefix = '$.' + prefix  # Add '$.' to indicate the start of the path
    return prefix

def quadratic_programming(match_dict):
    """
    Solves optimal source-target JSON path matching via quadratic programming and returns final matches.

    Args:
        match_dict (dict): {(target_path, source_path): score}

    Returns:
        dict: {source_path: (target_path, score)} — best match per source
    """
    quadratic_model = Model("json_matching")
    quadratic_model.setParam("OutputFlag", False)

    nested_t_vars = {}
    nested_s_vars = {}  
    prefix_vars = {}
    source_vars = defaultdict(list)
    score_terms = []


    for s_path, t_matches in match_dict.items():
        s_prefix = 's' + get_key_prefix(s_path)
        for t_path, score in t_matches:
            t_prefix = 't' + get_key_prefix(t_path)

            # Binary variable: 1 if source_path matches target_path
            match_var = quadratic_model.addVar(vtype=GRB.BINARY, name=f"{s_path}-----{t_path}")
            score_terms.append(score * match_var)
            source_vars[s_path].append((match_var, t_path, score))

            # Prefix_var: 1 if any matching under the prefix pair exists
            prefix_key = f'{s_prefix}-----{t_prefix}'
            if prefix_key not in prefix_vars:
                prefix_vars[prefix_key] = quadratic_model.addVar(vtype=GRB.BINARY, name=prefix_key)
            prefix_var = prefix_vars[prefix_key]
            
            # Nested_t_var: 1 if target_path matches any source_path with given prefix
            t_nest_key = f'{t_path}-----{s_prefix}'
            if t_nest_key not in nested_t_vars:
                nested_t_vars[t_nest_key] = quadratic_model.addVar(vtype=GRB.BINARY, name=t_nest_key)
            nested_t_var = nested_t_vars[t_nest_key]

            # Nested_s_var: 1 if source_path matches any target_path with given prefix
            s_nest_key = f'{s_path}-----{t_prefix}'
            if s_nest_key not in nested_s_vars:           
                nested_s_vars[s_nest_key] = quadratic_model.addVar(vtype=GRB.BINARY, name=s_nest_key)
            nested_s_var = nested_s_vars[s_nest_key]

            quadratic_model.addConstr(nested_t_var <= prefix_var, name=f"c_tprefix_{t_path}_{s_path}")
            quadratic_model.addConstr(nested_s_var <= prefix_var, name=f"c_sprefix_{t_path}_{s_path}")

    # Each source path matches at most one target path
    for s_path, var_list in source_vars.items():
        quadratic_model.addConstr(quicksum(var for var, _, _ in var_list) <= 1, name=f"c_unique_{s_path}")

    quadratic_model.update()
    quadratic_model.setObjective(quicksum(score_terms), GRB.MAXIMIZE)
    quadratic_model.optimize()

    if quadratic_model.status != GRB.OPTIMAL:
        print(f"Warning: Model was not solved optimally. Status: {quadratic_model.status}")
        if quadratic_model.status == GRB.INFEASIBLE:
            quadratic_model.computeIIS()
            quadratic_model.write("iis.ilp")
            print("Model is infeasible. See 'iis.ilp'.")
        return {}

    return predict_matches(source_vars)


# ----------------------------
# Step 1: Load SOURCE and TARGET datasets
# ----------------------------
def load_datasets(source_file, target_file):
    """
    Load source and target dataset from CSV files.

    Args:
        source_file (str): Path to the source CSV file.
        target_file (str): Path to the target CSV file.
    Returns:
        tuple: A tuple containing two DataFrames (source_df, target_df).
    """
    source_df = pd.read_csv(source_file, delimiter=';')
    target_df = pd.read_csv(target_file, delimiter=';')
    return source_df, target_df

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


# -----------------------------
# Step 2: Select datasets to evaluate
# -----------------------------
def filter_dataset_by_size(source_groups, target_groups, max_paths):
    """
    Get sources and targets with at most max_paths paths.

    Args:
        source_groups (dict): {filename: DataFrame}
        target_groups (dict): {filename: DataFrame}
        max_paths (int): Maximum number of paths allowed.
    Returns:
        list: Filenames that meet the criteria.
    """
    return [
        fn for fn in source_groups.keys()
        if len(source_groups[fn]) <= max_paths and len(target_groups[fn]) <= max_paths
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
    max_source = source_df["path"].apply(lambda x: len(x.split('.'))).max()
    max_target = target_df["path"].apply(lambda x: len(x.split('.'))).max()
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


# ----------------------------
# Step 3: Apply matching algorithm
# ----------------------------
def find_valentine(target_df, source_df):
    """
    Run multiple Valentine matching algorithms and return their results.

    Args:
        target_df (pd.DataFrame): Target dataset.
        source_df (pd.DataFrame): Source dataset.
    Returns:
        dict: Dictionary with algorithm names as keys and their match results as values.
    """
    models = {
        "SimilarityFlooding": SimilarityFlooding(),
        "Coma": Coma(),
        "DistributionBased": DistributionBased(),
        "JaccardDistanceMatcher": JaccardDistanceMatcher(),
        "Cupid": Cupid(),
    }

    results = {}
    for name, matcher in models.items():
        results[name] = valentine_match(target_df, source_df, matcher, "target", "source")

    return results

def reformat_valentine_matches(valentine_matches):
    """
    Reformat Valentine matches to {source_path: [(target_path, score), ...]}.

    Args:
        valentine_matches (dict): Output from find_valentine.
    Returns:
        dict: Reformatted matches.
    """
    matches = defaultdict(list)
    for (target_tuple, source_tuple), score in valentine_matches.items():
        target_path = target_tuple[1]
        source_path = source_tuple[1]
        matches[source_path].append((target_path, score))
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
        matches (dict): {source_path: [(target_path, score), ...]}
        ground_truth_pairs (set): Set of (source, target) pairs representing ground truth.

    Returns:
        dict: Evaluation metrics.
    """
    # Flatten matches to (source, target) pairs
    predicted_pairs = set()
    for s_path, targets in matches.items():
        for t_path, score in targets:
            predicted_pairs.add((s_path, t_path))

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


# ----------------------------
# Step 5: CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Find Matches between source and target.")
    parser.add_argument("source_file", help="Path to source CSV file.")
    parser.add_argument("target_file", help="Path to target CSV file.")
    parser.add_argument("groundtruth_file", help="Path to ground truth JSON file.")
    parser.add_argument("mode", choices=["valentine", "jasper"], help="Matching algorithm to use.")
    return parser.parse_args()

def main():
    all_metrics = []  # stores metrics for each matcher-dataset pair
    output_file = "results_summary.json"
    start_time = time.time()

    # 1. Load datasets
    args = parse_args()
    all_source_df, all_target_df = load_datasets(args.source_file, args.target_file)
    print(f"Loaded {len(all_source_df)} source rows and {len(all_target_df)} target rows.", flush=True)

    # 2. Group by filename
    source_groups = {fn: df for fn, df in all_source_df.groupby("filename")}
    target_groups = {fn: df for fn, df in all_target_df.groupby("filename")}

    # 3. Select datasets for evaluation
    eligible_files = filter_dataset_by_size(source_groups, target_groups, max_paths=200)
    print(f"Found {len(eligible_files)} datasets with <=200 paths.", flush=True)

    bins = group_datasets_by_depth(eligible_files, source_groups, target_groups, n_bins=3)
    sample_size = max(1, int(0.1 * len(eligible_files)))
    selected_datasets = sample_datasets_from_bins(bins, total_sample=sample_size)
    print(f"Evaluating {len(selected_datasets)} datasets (10%).", flush=True)

    # 4. Run matching and evaluation
    for fn in sorted(selected_datasets):
        source_df = source_groups[fn]
        target_df = target_groups[fn]
        pairs = get_ground_truth_pairs(args.groundtruth_file, fn)
        print(f"\nProcessing {fn}: {len(source_df)} → {len(target_df)} paths, {len(pairs)} GT pairs.", flush=True)

        mode = args.mode

        if mode == "valentine":
            new_source_df = pd.DataFrame(1, index=range(len(source_df)), columns=source_df["path"].astype(str))
            new_target_df = pd.DataFrame(1, index=range(len(target_df)), columns=target_df["path"].astype(str))

            valentine_matches = find_valentine(new_target_df, new_source_df)

            for matcher_name, matches in valentine_matches.items():
                formatted_matches = reformat_valentine_matches(matches)
                final_matches = {k: sorted(v, key=lambda x: -x[1])[0:1] for k, v in formatted_matches.items()}
                metrics = evaluate_matches(final_matches, pairs)
                metrics.update({"filename": fn, "matcher": matcher_name})
                all_metrics.append(metrics)
                print(f"{matcher_name}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1_score']:.2f}", flush=True)

        elif mode == "jasper":
            candidate_matches = match_paths(source_df, target_df, ling_weight=0.5, struct_weight=0.5, min_score=0.7)
            pruned_matches = prune_top_k_candidates(candidate_matches, top_k=5)
            final_matches = quadratic_programming(pruned_matches)
            metrics = evaluate_matches(final_matches, pairs)
            metrics.update({"filename": fn, "matcher": "JASPER"})
            all_metrics.append(metrics)
            print(f"JASPER: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1_score']:.2f}", flush=True)

        else:
            raise ValueError("Invalid mode. Choose from: valentine, jasper")

    # 5. Compute aggregated results
    avg_scores = defaultdict(lambda: {"precision": [], "recall": [], "f1_score": []})
    for m in all_metrics:
        matcher = m["matcher"]
        avg_scores[matcher]["precision"].append(m["precision"])
        avg_scores[matcher]["recall"].append(m["recall"])
        avg_scores[matcher]["f1_score"].append(m["f1_score"])

    # Compute mean per matcher
    summary = {}
    for matcher, scores in avg_scores.items():
        summary[matcher] = {
            "avg_precision": sum(scores["precision"]) / len(scores["precision"]),
            "avg_recall": sum(scores["recall"]) / len(scores["recall"]),
            "avg_f1": sum(scores["f1_score"]) / len(scores["f1_score"]),
            "n_datasets": len(scores["precision"])
        }

    # 6. Save everything
    results = {
        "per_dataset": all_metrics,
        "summary": summary,
        "total_datasets": len(selected_datasets),
        "execution_time_sec": round(time.time() - start_time, 2)
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_file}")


if __name__ == "__main__":
    main()