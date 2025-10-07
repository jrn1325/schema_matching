import argparse
import collections
import csm
import csv
import json
import math
import pandas as pd
import random
import re
import sys
import time
import torch.nn.functional as F


from collections import defaultdict
from gurobipy import *
from scipy.optimize import linprog
from valentine import valentine_match
from valentine.algorithms import Coma, Cupid, DistributionBased, JaccardDistanceMatcher, SimilarityFlooding



def score_path(entry, w_depth=0.4, w_entropy=0.3, w_siblings=0.3):
    """
    Compute a hybrid score for a path entry using weighted sum of:
    - nesting depth
    - key entropy (diversity of subkeys)
    - number of siblings (cardinality)

    Args:
        entry (dict): Stats for a single path.
        w_depth (float): Weight for nesting depth.
        w_entropy (float): Weight for key entropy.
        w_siblings (float): Weight for number of siblings.
    
    Returns:
        float: Composite score.
    """
    return (
        w_depth * entry["nesting_depth"] +
        w_entropy * entry["key_entropy"] +
        w_siblings * entry["num_siblings"]
    )

def sample_paths(path_stats, k):
    """
    Sample k paths using score-weighted random sampling.

    Args:
        path_stats (dict): Output of extract_paths.
        k (int): Number of paths to sample.

    Returns:
        dict: Subset of path_stats with k sampled paths.
    """
    # Compute scores for each path
    scored_items = [(path, stats, score_path(stats)) for path, stats in path_stats.items()]

    # Avoid division by zero if all scores are zero
    total_score = sum(score for _, _, score in scored_items)
    if total_score == 0:
        # Fall back to uniform sampling
        weights = [1] * len(scored_items)
    else:
        weights = [score / total_score for _, _, score in scored_items]

    # Sample k unique items without replacement
    sampled = random.choices(
        population=scored_items,
        weights=weights,
        k=min(k, len(scored_items))
    )

    # Convert to dict format
    return {path: stats for path, stats, _ in sampled}



# Linguistic similarity functions for Jasper
def get_linguistic_similarity(emb1, emb2):
    """
    Compute cosine similarity between two precomputed embeddings.
    
    Args:
        emb1 (torch.Tensor): Embedding for path 1.
        emb2 (torch.Tensor): Embedding for path 2.
    Returns:
        float: Cosine similarity score between -1 and 1.
    """
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1).item()


# Structural similarity functions for Jasper
def weighted_depth_similarity(keys1, keys2):
    """
    Compute depth similarity with a bigger penalty for larger differences.

    Args:
        keys1 (list): Path 1 keys.
        keys2 (list): Path 2 keys.

    Returns:
        float: Depth similarity score (0 to 1).
    """
    len1, len2 = len(keys1), len(keys2)
    if len1 == 0 and len2 == 0:
        return 1.0
    depth_diff = abs(len1 - len2)
    max_depth = max(len1, len2)
    return 1 - (depth_diff / max_depth) ** 2

def divergence_penalty(keys1, keys2):
    """
    Penalize paths that diverge earlier in the hierarchy, with greater penalties for divergence
    that happens at higher levels of the path.

    Args:
        keys1 (list): Path 1 keys.
        keys2 (list): Path 2 keys.

    Returns:
        float: Penalty factor (0 to 1). Lower value for early divergence.
    """
    for i, (a, b) in enumerate(zip(keys1, keys2)):
        if a != b:
            return 1 / (1 + i)
    return 1.0

def sibling_similarity(num_siblings1, num_siblings2):
    """
    Compare the number of sibling keys at the same level in two paths.

    Args:
        num_siblings1 (int): The number of sibling keys at the same level for path1.
        num_siblings2 (int): The number of sibling keys at the same level for path2.

    Returns:
        float: Similarity score (0 to 1).
    """
    diff = abs(num_siblings1 - num_siblings2)
    return 1 / (1 + diff)

def type_similarity(type1, type2):
    """
    Compare the dominant data types under each path.

    Args:
        type1 (str): The data type of the value at path1.
        type2 (str): The data type of the value at path2.

    Returns:
        float: 1.0 if same type, else 0.0.
    """
    return 1.0 if type1 == type2 else 0.0

def key_entropy_similarity(entropy1, entropy2):
    """
    Compare the key entropy of two paths to measure the variability in their nested structures.
    A lower difference in entropy indicates more similarity in the structure.

    Args:
        entropy1 (float): The key entropy value at path1, representing the variability of keys.
        entropy2 (float): The key entropy value at path2, representing the variability of keys.

    Returns:
        float: Similarity score between 0 and 1.
    """
    diff = abs(entropy1 - entropy2)
    return 1 / (1 + diff)

def get_structural_similarity(
    path1, path2, path1_stats, path2_stats,
    w_depth=0.167, w_prefix=0.167, w_divergence=0.167,
    w_sibling=0.167, w_type=0.167, w_entropy=0.167
):
    """
    Compute structural similarity between two paths based on their statistics.
    
    Args:
        path1 (str): Path 1 in dot notation.
        path2 (str): Path 2 in dot notation.
        path1_stats (dict): Statistics for path 1.
        path2_stats (dict): Statistics for path 2.
    Returns:
        float: Structural similarity score between 0 and 1.
    """
    keys1 = path1.split('.')
    keys2 = path2.split('.')

    depth_sim = weighted_depth_similarity(keys1, keys2)
    prefix_len = sum(a == b for a, b in zip(keys1, keys2))
    prefix_sim = prefix_len / max(len(keys1), len(keys2))
    divergence = divergence_penalty(keys1, keys2)
    sibling_sim = sibling_similarity(
        path1_stats.get("num_siblings", 0),
        path2_stats.get("num_siblings", 0)
    )
    type_sim = type_similarity(
        path1_stats.get("type"),
        path2_stats.get("type")
    )
    entropy_sim = key_entropy_similarity(
        path1_stats.get("key_entropy", 0.0),
        path2_stats.get("key_entropy", 0.0)
    )

    return (
        w_depth * depth_sim +
        w_prefix * prefix_sim +
        w_divergence * divergence +
        w_sibling * sibling_sim +
        w_type * type_sim +
        w_entropy * entropy_sim
    )


# Matching functions for Jasper
def are_types_compatible(type1, type2, stats1=None, stats2=None, strict=True):
    """
    Check if two JSON types are compatible.

    Args:
        type1 (str): First type (e.g., 'object', 'string', 'number').
        type2 (str): Second type.
        stats1 (dict, optional): Stats for path1.
        stats2 (dict, optional): Stats for path2.
        strict (bool): Whether to enforce exact matching of types.

    Returns:
        bool: True if compatible, False otherwise.
    """
    if type1 == type2:
        return True

    # Allow null matching anything
    if type1 == "null" or type2 == "null":
        return True

    if strict:
        return False

    # Incompatible scalar combinations
    scalar_incompatibles = {
        ("string", "number"), ("number", "string"),
        ("string", "boolean"), ("boolean", "string"),
        ("number", "boolean"), ("boolean", "number"),
    }
    if (type1, type2) in scalar_incompatibles or (type2, type1) in scalar_incompatibles:
        return False

    # Allow integer ≈ number
    if (type1 == "integer" and type2 == "number") or (type2 == "integer" and type1 == "number"):
        return True

    # Object vs array of objects (relaxed)
    if type1 == "object" and type2 == "array":
        return is_array_of_objects(stats2)
    if type2 == "object" and type1 == "array":
        return is_array_of_objects(stats1)

    return False

    
# Jasper matching functions
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

def match_paths(paths1, paths2, ling_weight=0.5, struct_weight=0.5, min_score=0.7):
    """
    Match paths from two sets using precomputed embeddings and structural similarity.

    Args:
        paths1, paths2 (dict): path -> stats
    """
    total_weight = ling_weight + struct_weight
    ling_weight /= total_weight
    struct_weight /= total_weight

    matches = {}

    # Precompute embeddings
    paths1_list = list(paths1.keys())
    paths2_list = list(paths2.keys())
    embeddings1 = compute_embeddings_batch(paths1_list)
    embeddings2 = compute_embeddings_batch(paths2_list)
    
    # Prune incompatible type pairs
    compatible_pairs = [
        (p1, p2)
        for p1 in paths1_list
        for p2 in paths2_list
        if are_types_compatible(
            paths1[p1].get("type"), paths2[p2].get("type"),
            paths1[p1], paths2[p2]
        )
    ]

    # Compute similarities only on valid pairs
    for p1, p2 in compatible_pairs:
        ling_score = get_linguistic_similarity(embeddings1[p1], embeddings2[p2])
        struct_score = get_structural_similarity(p1, p2, paths1[p1], paths2[p2])
        final_score = ling_weight * ling_score + struct_weight * struct_score

        if final_score >= min_score:
            matches[(p1, p2)] = final_score
            

    return matches

def predict_matches(source_vars):
    """
    Selects final match from optimized source_vars.

    Args:
        source_vars (dict): Maps source paths to list of (var, target_path, score).

    Returns:
        dict: {source_path: (target_path, score)}
    """
    final_matching = defaultdict(set)

    for s_path, var_list in source_vars.items():
        for var, t_path, score in var_list:
            try:
                if hasattr(var, 'X') and round(var.X):
                    final_matching[t_path].add((s_path, score))
                    break
            except Exception as e:
                print(f"[Error] Reading var.X for {t_path}->{s_path}: {e}")

    return dict(sorted(final_matching.items()))

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

    grouped_by_target = defaultdict(list)
    for (t_path, s_path), score in match_dict.items():
        grouped_by_target[t_path].append((s_path, score))

    for t_path, s_matches in grouped_by_target.items():
        t_prefix = 't' + get_key_prefix(t_path)
        for s_path, score in s_matches:
            s_prefix = 's' + get_key_prefix(s_path)

            # match_var: 1 if target_path matches source_path, else 0
            match_var = quadratic_model.addVar(vtype=GRB.BINARY, name=f"{t_path}-----{s_path}")
            score_terms.append(score * match_var)
            source_vars[s_path].append((match_var, t_path, score))

            # prefix_var: 1 if any matching under the prefix pair exists
            prefix_key = f'{t_prefix}-----{s_prefix}'
            if prefix_key not in prefix_vars:
                prefix_vars[prefix_key] = quadratic_model.addVar(vtype=GRB.BINARY, name=prefix_key)
            prefix_var = prefix_vars[prefix_key]
            
            # nested_t_var: 1 if target_path matches any source_path with given prefix
            t_nest_key = f'{t_path}-----{s_prefix}'
            if t_nest_key not in nested_t_vars:
                nested_t_vars[t_nest_key] = quadratic_model.addVar(vtype=GRB.BINARY, name=t_nest_key)
            nested_t_var = nested_t_vars[t_nest_key]

            # nested_s_var: 1 if source_path matches any target_path with given prefix
            s_nest_key = f'{s_path}-----{t_prefix}'
            if s_nest_key not in nested_s_vars:           
                nested_s_vars[s_nest_key] = quadratic_model.addVar(vtype=GRB.BINARY, name=s_nest_key)
            nested_s_var = nested_s_vars[s_nest_key]

            quadratic_model.addConstr(nested_t_var <= prefix_var, name=f"c_tprefix_{t_path}_{s_path}")
            quadratic_model.addConstr(nested_s_var <= prefix_var, name=f"c_sprefix_{t_path}_{s_path}")

    for s_path, match_vars in source_vars.items():
        quadratic_model.addConstr(quicksum(var for var, _, _ in match_vars) <= 1, name=f"c_unique_{s_path}")

    quadratic_model.update()

    penalty_terms = []
    for key, var in prefix_vars.items():
        depth = key.count('.') + 1
        penalty_terms.append(0.001 * depth * var)

    quadratic_model.setObjective(quicksum(score_terms) - quicksum(penalty_terms), GRB.MAXIMIZE)
    quadratic_model.optimize()

    if quadratic_model.status != GRB.OPTIMAL:
        print(f"[Warning] Model was not solved optimally. Status: {quadratic_model.status}")
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


# ----------------------------
# Step 2: Apply matching algorithm
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
    Reformat Valentine matches to {target_path: [(source_path, score), ...]}.

    Args:
        valentine_matches (dict): Output from find_valentine.
    Returns:
        dict: Reformatted matches.
    """
    matches = defaultdict(list)
    for (target_tuple, source_tuple), score in valentine_matches.items():
        target_path = target_tuple[1]
        source_path = source_tuple[1]
        matches[target_path].append((source_path, score))
    return matches


# ----------------------------
# Step 3: Evaluate matches against ground truth
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
    for t, sources in matches.items():
        for s, score in sources:
            predicted_pairs.add((s, t))

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
# Step 4: CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Find Matches between source and target.")
    parser.add_argument("source_file", help="Path to source CSV file.")
    parser.add_argument("target_file", help="Path to target CSV file.")
    parser.add_argument("groundtruth_file", help="Path to ground truth JSON file.")
    parser.add_argument("mode", choices=["valentine", "jasper"], help="Matching algorithm to use.")
    return parser.parse_args()

def main():
    #valentine_matchers = ["SimilarityFlooding", "Coma", "DistributionBased", "JaccardDistanceMatcher", "Cupid"]
    all_metrics = []
    start_time = time.time()

    # 1. Load datasets
    args = parse_args()
    all_source_df, all_target_df = load_datasets(args.source_file, args.target_file)
    print(f"Loaded {len(all_source_df)} source rows and {len(all_target_df)} target rows.", flush=True)

   # 2. Group by filename
    source_groups = {fn: df for fn, df in all_source_df.groupby("filename")}
    target_groups = {fn: df for fn, df in all_target_df.groupby("filename")}

    # Loop over each unique filename
    for fn in sorted(source_groups.keys()):
        source_df = source_groups[fn]
        target_df = target_groups[fn]

        # Skip datasets with more than 100 paths
        if len(source_df) > 100 or len(target_df) > 100:
            print(f"Skipping file {fn} (source: {len(source_df)}, target: {len(target_df)}) because it has more than 100 paths.", flush=True)
            continue

        # 3. Load ground truth pairs for this filename
        pairs = get_ground_truth_pairs(args.groundtruth_file, fn)
        print(f"\nProcessing file: {fn} with {len(source_df)} source rows and {len(target_df)} target rows.", flush=True)
        print(f"Ground truth has {len(pairs)} pairs.", flush=True)

        mode = args.mode
        if mode == "valentine":
            # 4. Convert to DataFrame format expected by Valentine
            new_source_df = pd.DataFrame(1, index=range(len(source_df)), columns=source_df["path"].astype(str))
            new_target_df = pd.DataFrame(1, index=range(len(target_df)), columns=target_df["path"].astype(str))

            # 5. Run one of Valentine matchers
            print(f"Running Valentine matcher: {mode}", flush=True)
            valentine_matches = find_valentine(new_target_df, new_source_df)

            for matcher_name, matches in valentine_matches.items():
                print(f"Matcher: {matcher_name}, Matches: {len(matches)}", flush=True)

                # 6. Reformat matches to {target_path: [(source_path, score), ...]}
                formatted_matches = reformat_valentine_matches(matches)

                # 7. Get the top match per target path
                final_matches = {k: sorted(v, key=lambda x: -x[1])[0:1] for k, v in formatted_matches.items()}

                # 8. Evaluate matches
                metrics = evaluate_matches(final_matches, pairs)
                all_metrics.append(metrics)

                print(f"Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1 Score: {metrics['f1_score']:.2f}", flush=True)

        elif mode == "jasper":
            matches = match_paths(sampled_target_paths, sampled_source_paths)
            matches = quadratic_programming(matches)
            final_matches = {k: v for k, v in sorted(matches.items(), key=lambda x: -x[1][1])[:100]}

        else:
            raise ValueError("Invalid mode. Choose from: " + ", ".join(valentine_matchers + ["jasper"]))

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds", flush=True)

    # Optionally: average metrics over all runs
    #if all_metrics:
    #    avg = lambda k: round(sum(m[k] for m in all_metrics) / len(all_metrics), 2)
    #    print(f"\nAveraged Metrics across {len(all_metrics)} datasets:")
    #    print(f"Precision: {avg('precision')}, Recall: {avg('recall')}, F1 Score: {avg('f1_score')}", flush=True)


if __name__ == "__main__":
    main()