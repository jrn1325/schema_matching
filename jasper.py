import argparse
import ast
import valentine.algorithms as algos
print(dir(algos))
import base64
import json
import math
import numpy as np
import pandas as pd
import random
import sys
import time
import torch

from collections import defaultdict
from gurobipy import *
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from valentine import valentine_match
from valentine.algorithms import Coma, Cupid, DistributionBased, JaccardDistanceMatcher, SimilarityFlooding



ARRAY_WILDCARD = "<ARRAY_ITEM>"
DELIM = "_DELIM_"

VALENTINE_MATCHERS = {
    "coma": lambda: Coma(use_instances=True),
    "cupid": lambda: Cupid(),
    "jaccard": lambda: JaccardDistanceMatcher(),
    "distribution": lambda: DistributionBased(),
    "similarityflooding": lambda: SimilarityFlooding(),
    
}

SIZE_FILTERS = {
    "small":  lambda n: n <= 100,
    "medium": lambda n: 100 < n < 500,
    "large":  lambda n: 500 <= n < 1000,
    "xlarge": lambda n: n >= 1000
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
    source_path_keys = s_row["path"]
    target_path_keys = t_row["path"]

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

def match_paths(source_df, target_df, ling_weight=1.0, struct_weight=0.0, min_score=0.7, device="cuda"):
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
                src_path = tuple(ast.literal_eval(s_row.path))
                tgt_path = tuple(ast.literal_eval(t_row.path))
                matches[src_path].append((tgt_path, final_score))

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
    """
    Get the parent/prefix of a JSON path.

    Args:
        path (list): A JSON path list
    Returns:
        list or None: The parent path list, or None if there is no parent.
    """
    if len(path) <= 1:
        return None
    return path[:-1]

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
    max_source = source_df["path"].apply(lambda x: len(x)).max()
    max_target = target_df["path"].apply(lambda x: len(x)).max()

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
    #print(f"Best number of clusters = {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    agg["cluster"] = kmeans.fit_predict(X_scaled)

    # Sample datasets from each cluster
    sample_size = int(len(agg) * sample_fraction)
    #print(f"Sampling {sample_size} datasets (~{sample_fraction*100:.0f}%)")

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
# Step 3: Apply matching algorithm & Evaluate matches against ground truth
# ----------------------------

def parse_path(s):
    return tuple(json.loads(s))

def encode_path(p):
    return DELIM.join(p)

def normalize_path(p):
    if isinstance(p, str):
        if DELIM in p:
            return tuple(p.split(DELIM))
        else:
            return (p,)
    return p

def safe_parse_values(x):
    # Case 1: already a list/dict
    if isinstance(x, (list, dict)):
        return x

    if not isinstance(x, str):
        return x

    s = x.strip()

    if not s:
        return []

    # Step 1: try JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Step 2: try Python literal
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass

    # Step 3: fallback (return raw string)
    return x


def run_valentine(source_df, target_df, matcher, matcher_instance):
    """
    Run Valentine matcher.
    
    Args:
        source_df (pd.DataFrame): Source dataset.
        target_df (pd.DataFrame): Target dataset.
        matcher (str): Name of the Valentine matcher to use.
        matcher_instance: Initialized Valentine matcher instance.
    Returns:    
        dict: key as source and value as target with score.
    """
    # Parse paths once
    source_paths = source_df["path"].apply(parse_path)
    target_paths = target_df["path"].apply(parse_path)

    if matcher in ["coma", "cupid", "similarityflooding"]:  # schema matching
        source_cols = [encode_path(p) for p in source_paths]
        target_cols = [encode_path(p) for p in target_paths]

        new_source_df = pd.DataFrame(columns=source_cols)
        new_target_df = pd.DataFrame(columns=target_cols)

    elif matcher in ["jaccard", "distribution"]:  # instance matching
        source_df = source_df.copy()
        target_df = target_df.copy()

        source_df["path"] = source_paths.apply(encode_path)
        target_df["path"] = target_paths.apply(encode_path)

        source_df["values"] = source_df["values"].apply(safe_parse_values)
        target_df["values"] = target_df["values"].apply(safe_parse_values)

        new_source_df = source_df.set_index("path")["values"].apply(pd.Series).T
        new_target_df = target_df.set_index("path")["values"].apply(pd.Series).T

    matches = valentine_match(new_source_df, new_target_df, matcher_instance, "source", "target")
    return matches

def reformat_valentine_matches(valentine_matches):
    """
    Reformat Valentine matches from {(source_path, target_path): score} with encoded paths to {(source_path, target_path): score} with decoded paths.   

    Args:
        valentine_matches (dict): {(source_path,(target_path): score)}

    Returns:
        dict: {(source_path, target_path): score}
    """
    matches = defaultdict(list)

    for (src, tgt), score in valentine_matches.items():
        # Extract path directly (Valentine format: (side, path))
        src_path = src[1]
        tgt_path = tgt[1]

        src_path = normalize_path(src_path)
        tgt_path = normalize_path(tgt_path)

        matches[(src_path, tgt_path)].append(score)

    return matches


# ----------------------------
# Step 4: Apply quadratic programming
# ----------------------------
def get_depth(path):
    """
    Get the depth of a JSON path, defined as the number of keys in the path.

    Args:
        path (list): A JSON path as a list of keys, e.g., ["user", "address", "street"] for "user.address.street".
    Returns:
        int: The depth of the path.
    
    """
    return len(path)

def structurally_incompatible(s1, s2, t1, t2):
    """
    Check if two matches (s1→t1 and s2→t2) are structurally incompatible.   
    Structural incompatibility occurs when:
    1. The source paths share a parent but the target paths do not (or vice versa).
    2. The source paths have significantly different depths but the target paths do not (or vice versa).

    Args:       
        s1, s2 (str): Source paths.
        t1, t2 (str): Target paths.
    Returns:    
        bool: True if the matches are structurally incompatible, False otherwise.
    """

    # Prevent crossing hierarchy
    if share_parent(s1, s2) and not share_parent(t1, t2):
        return True

    # Prevent depth mismatch explosion
    if abs(get_depth(s1) - get_depth(t1)) > 2:
        return True

    return False

def get_parent(path):
    """
    Get the parent path of a JSON path.

    Args:
        path (list): A JSON path as a list of keys, e.g., ["user", "address", "street"] for "user.address.street".
    Returns:
        list or None: The parent path, e.g., ["user", "address"] for ["user", "address", "street"]. Returns None if there is no parent.
    """

    if len(path) <= 1:
        return None

    return path[:-1]

def share_parent(p1, p2):
    """
    Check if two paths share the same parent.

    Args:
        p1, p2 (list): Two JSON path lists.

    Returns:
        bool: True if the paths share the same parent, False otherwise.
    """
    return get_parent(p1) == get_parent(p2)

def parent_similarity_bonus(s_path, t_path):

    s_parent = get_parent(s_path)
    t_parent = get_parent(t_path)

    if not s_parent or not t_parent:
        return False

    return s_parent == t_parent

def likely_related(s1, s2):
    """
    Heuristic to quickly check if two source paths are likely related based on their depth and parent.

    Args:    
        s1, s2 (str): Two source JSON paths.
    Returns:    
        bool: True if the paths are likely related, False otherwise.
    """
    if get_depth(s1) != get_depth(s2):
        return False

    if get_parent(s1) != get_parent(s2):
        return False

    return True

def extract_solution(x_vars, candidate_matches):
    """"
    Extract the solution from the quadratic programming variables.

    Args:
        x_vars (dict): A dictionary of (source_path, target_path) -> variable.

    Returns:
        dict: A dictionary of source_path -> target_path for selected matches.
    """
    
    matches = {}

    for (s_path, t_path), var in x_vars.items():

        if var.X > 0.5:
            # Recover original score
            score = dict(candidate_matches[s_path])[t_path]

            matches[(s_path, t_path)] = score

    return matches

def quadratic_programming(candidate_matches, lambda_parent=0.7, lambda_conflict=1.5):

    model = Model("json_matching_ilp")
    model.setParam("OutputFlag", False)

    x_vars = {}
    source_vars = defaultdict(list)
    target_vars = defaultdict(list)

    linear_terms = []

    # -----------------------------------
    # 1️ Decision Variables + Scores
    # -----------------------------------
    for s_path, targets in candidate_matches.items():

        for t_path, score in targets:

            # Optional Parent Bonus (Linearized Heuristic)
            adjusted_score = score

            if parent_similarity_bonus(s_path, t_path):
                adjusted_score += lambda_parent

            x = model.addVar(
                vtype=GRB.BINARY,
                name=f"{s_path}__{t_path}"
            )

            x_vars[(s_path, t_path)] = x
            source_vars[s_path].append(x)
            target_vars[t_path].append(x)

            linear_terms.append(adjusted_score * x)

    # -----------------------------------
    # 2️ Hard Constraint: Unique Source
    # -----------------------------------
    for s_path, vars_list in source_vars.items():
        model.addConstr(quicksum(vars_list) <= 1)

    # -----------------------------------
    # 3️ Hard Constraint: Unique Target
    # -----------------------------------
    for t_path, vars_list in target_vars.items():
        model.addConstr(quicksum(vars_list) <= 1)

    # -----------------------------------
    # 4️ Structural Conflict Constraints
    # -----------------------------------
    keys = list(x_vars.keys())

    for i in range(len(keys)):
        s1, t1 = keys[i]
        x1 = x_vars[(s1, t1)]

        for j in range(i + 1, len(keys)):
            s2, t2 = keys[j]
            x2 = x_vars[(s2, t2)]

            if structurally_incompatible(s1, s2, t1, t2):
                model.addConstr(x1 + x2 <= 1)

    # -----------------------------------
    # 5️ Linear Objective
    # -----------------------------------
    model.setObjective(quicksum(linear_terms), GRB.MAXIMIZE)

    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"Warning: Status {model.status}")
        return {}

    return extract_solution(x_vars, candidate_matches)

def run_jasper(source_df, target_df, device):
    """
    Run Jasper matcher and evaluate.
    
    Args:
        source_df (pd.DataFrame): Source dataset.
        target_df (pd.DataFrame): Target dataset.
        device (str): Device to use for tensor computations.
    Returns:    
        dict: key as (source, target) and value as score.
    """
    candidate_matches = match_paths(source_df, target_df, ling_weight=1.0, struct_weight=0.0, min_score=0.7, device=device)
    pruned_matches = prune_top_k_candidates(candidate_matches, top_k=3)
    pruned_pairs = {s: [(t, score) for t, score in tgts] for s, tgts in pruned_matches.items()}
    #final_matches = quadratic_programming(pruned_pairs)ribution
    final_matches = final_match(pruned_pairs, alpha=0.2, top_k=1)
    return final_matches



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

            src = tuple(mapping["original_path"])
            tgt = tuple(mapping["transformed_path"])
            gt.add((src, tgt))
    return gt

def evaluate_matches(matches, ground_truth_pairs):
    """
    Evaluate predicted matches against ground truth from a JSON file

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




# ----------------------------
# Step 5: CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Find Matches between source and target.")
    parser.add_argument("source_dir", help="Directory with source csv files")
    parser.add_argument("target_dir", help="Directory with target csv files")
    parser.add_argument("groundtruth_file", help="Path to ground truth JSON file.")
    parser.add_argument("mode", choices=["coma", "cupid", "jaccard", "distribution", "similarityflooding", "jasper"], help="Matching algorithm to use.")
    parser.add_argument("size", type=str, choices=["small", "medium", "large", "xlarge"], help="Size of datasets to process.")
    return parser.parse_args()

def main():
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    source_files = {f.name for f in source_dir.glob("*.csv")}

    size_filter = SIZE_FILTERS[args.size]

    # -----------------------
    # Initialize matcher
    # -----------------------
    if args.mode in VALENTINE_MATCHERS:
        matcher_instance = VALENTINE_MATCHERS[args.mode]()
        mode_type = "valentine"
    elif args.mode == "jasper":
        matcher_instance = None
        mode_type = "jasper"
    else:
        raise ValueError("Invalid mode")

    precision_list, recall_list, f1_list = [], [], []
    time_dict = {}

    print(f"\n=== Running {args.mode} ===")

    # -----------------------
    # Main loop
    # -----------------------
    for filename in tqdm(source_files, desc="Processing datasets"):

        source_path = source_dir / filename
        target_path = target_dir / filename

        source_df = load_dataset(source_path)
        target_df = load_dataset(target_path)

        # Size bucket filter (OPEN-ENDED)
        if not size_filter(len(source_df)):
            continue

        gt_pairs = get_ground_truth_pairs(
            args.groundtruth_file,
            filename.replace(".csv", ".json")
        )

        print(
            f"\nProcessing {filename}: "
            f"{len(source_df)} → {len(target_df)} paths, "
            f"{len(gt_pairs)} GT pairs.",
            flush=True
        )

        file_start_time = time.time()

        if mode_type == "valentine":
            matches = run_valentine(source_df, target_df, args.mode, matcher_instance)
            matches = reformat_valentine_matches(matches)
        else:
            matches = run_jasper(source_df, target_df, device=device)

        duration = round(time.time() - file_start_time, 2)

        metrics = evaluate_matches(matches, gt_pairs)

        print(
            f"Time: {duration}s | "
            f"P={metrics['precision']:.3f}, "
            f"R={metrics['recall']:.3f}, "
            f"F1={metrics['f1_score']:.3f}",
            flush=True
        )

        precision_list.append(metrics["precision"])
        recall_list.append(metrics["recall"])
        f1_list.append(metrics["f1_score"])

        time_dict[filename] = {
            "time": duration,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1_score"],
        }

    # -----------------------
    # Aggregate stats
    # -----------------------
    avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0
    avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0

    print(f"\n=== Overall Evaluation of {args.mode} ===")
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall:    {avg_recall:.3f}")
    print(f"Average F1 Score:  {avg_f1:.3f}")
    print(f"Execution time: {round(time.time() - start_time, 2)} seconds")

    # Sort by runtime descending
    time_dict = dict(
        sorted(time_dict.items(),
               key=lambda item: item[1]["time"],
               reverse=True)
    )

    out_file = f"{args.mode}_{args.size}_execution_times.json"
    with open(out_file, "w") as f:
        json.dump(time_dict, f, indent=4)

    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
