import collections
import csm
import json
import math
import pandas as pd
import random
import re
import sys
import torch
import torch.nn.functional as F


from collections import defaultdict, Counter
from gurobipy import *
from pathlib import Path
from scipy.optimize import linprog
from transformers import RobertaTokenizer, RobertaModel
from valentine import valentine_match
from valentine.algorithms import SimilarityFlooding, Coma, Cupid, DistributionBased, JaccardDistanceMatcher


MODEL = "microsoft/codebert-base"
tokenizer = RobertaTokenizer.from_pretrained(MODEL)
model = RobertaModel.from_pretrained(MODEL)


def parse_document(doc, path=''):
    """
    Recursively parse all paths from a JSON object using dot notation.

    Args:
        doc (dict or list): The JSON document.
        path (str): Current path.
        
    Yields:
        tuple: (path, value)
    """
    if isinstance(doc, dict):
        siblings = len(doc)
        for key, value in doc.items():
            full_path = f"{path}.{key}" if path else key
            yield from parse_document(value, full_path)
            yield (full_path, value, siblings)
    elif isinstance(doc, list):
        for item in doc:
            full_path = f"{path}.*"
            yield from parse_document(item, full_path)
    else:
        yield (path, doc, 0)


def normalize_value(val):
    """
    Normalize a value for consistent representation.
    Args:
        val: The value to normalize.
    Returns:
        str: The normalized value.
    """
    if isinstance(val, (str, int, float, bool, type(None))):
        return val
    return json.dumps(val, sort_keys=True)


def infer_type(value):
    """
    Infer the type of a value.

    Args:
        value: The value to infer.
    Returns:
        str: The inferred type.
    """
    if isinstance(value, dict):
        return "object"
    elif isinstance(value, list):
        return "array"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, (int, float)):
        return "number"
    elif isinstance(value, str):
        return "string"
    elif value is None:
        return "null"
    else:
        return "unknown"
    

def key_entropy(keys):
    """
    Calculate the entropy of keys in a list.

    Args:
        keys (list): List of keys.

    Returns:
        float: The entropy value.
    """
    if not keys:
        return 0.0

    counter = Counter(keys)
    total = len(keys)
    return -sum((count / total) * math.log2(count / total) for count in counter.values() if count > 0)


def extract_paths(docs):
    path_stats = defaultdict(lambda: {
        "values": set(),
        "types": defaultdict(int),
        "num_siblings_sum": 0,
        "num_siblings_count": 0,
        "key_entropy_sum": 0,
        "key_entropy_count": 0,
        "num_children_sum": 0,
        "num_children_count": 0,
        "nesting_depth": 0,
    })

    for doc in docs:
        for path, value, siblings in parse_document(doc):
            parts = path.split('.')
            depth = len(parts)
            entry = path_stats[path]

            entry["values"].add(normalize_value(value))
            entry["types"][infer_type(value)] += 1
            entry["num_siblings_sum"] += siblings
            entry["num_siblings_count"] += 1
            entry["nesting_depth"] = max(entry["nesting_depth"], depth)

            if isinstance(value, dict):
                keys = list(value.keys())
                entry["key_entropy_sum"] += key_entropy(keys)
                entry["key_entropy_count"] += 1
                entry["num_children_sum"] += len(keys)
                entry["num_children_count"] += 1

    # Finalize averages and dominant type
    final = {}
    for path, entry in path_stats.items():
        final[path] = {
            "values": entry["values"],
            "type": max(entry["types"].items(), key=lambda x: x[1])[0],
            "num_siblings": entry["num_siblings_sum"] / entry["num_siblings_count"] if entry["num_siblings_count"] else 0,
            "key_entropy": entry["key_entropy_sum"] / entry["key_entropy_count"] if entry["key_entropy_count"] else 0,
            "num_children": entry["num_children_sum"] / entry["num_children_count"] if entry["num_children_count"] else 0,
            "nesting_depth": entry["nesting_depth"]
        }

    return final



def read_json(file_path):
    """
    Read a JSONL file and return a list of JSON objects.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        list: A list of JSON objects.
    """
    with open(file_path) as f:
        docs = [json.loads(line) for line in f]
    return docs


def compute_embeddings(path):
    """
    Compute embeddings for a JSON path using a pre-trained model.

    Args:
        path (str): A JSON path string.

    Returns:
        torch.Tensor: A tensor of shape (1, embedding_dim).
    """
    inputs = tokenizer(path, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def normalize_key(key):
    """
    Normalize a key by replacing underscores with spaces and splitting camel case.

    Args:
        key (str): The key to normalize.
    Returns:
        str: The normalized key.
    """
    key = re.sub(r'([a-z])([A-Z])', r'\1 \2', key)  # camelCase → camel Case
    key = key.replace('_', ' ')                     # snake_case → snake case
    return key.lower()


def linguistic_similarity(path1, path2):
    """
    Compute similarity between two JSON paths using cosine similarity on their key tokens.
    
    Args:
        path1 (list[str] or str): First JSON path.
        path2 (list[str] or str): Second JSON path.
    
    Returns:
        float: Similarity score between 0 and 1.
    """
    # Normalize to list of keys
    if isinstance(path1, str):
        path1 = path1.split('.')
    if isinstance(path2, str):
        path2 = path2.split('.')

    # Normalize keys
    norm1 = ' '.join(normalize_key(k) for k in path1)
    norm2 = ' '.join(normalize_key(k) for k in path2)

    # Compute embeddings
    emb1 = compute_embeddings(norm1)
    emb2 = compute_embeddings(norm2)
    return F.cosine_similarity(emb1, emb2, dim=1).item()


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


def structural_similarity(path1, path2, path1_stats, path2_stats):
    """
    Compute structural similarity between two JSON paths using multiple factors.

    Args:
        path1 (str): The first JSON path.
        path2 (str): The second JSON path.
        path1_stats (dict): Statistics for the first path.
        path2_stats (dict): Statistics for the second path.

    Returns:
        float: A similarity score between 0 and 1.
    """
    # Split paths into keys
    keys1 = path1.split('.')
    keys2 = path2.split('.')
    
    # Component similarities
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
        0.2 * depth_sim +
        0.2 * prefix_sim +
        #0.2 * divergence +
        0.2 * sibling_sim +
        0.2 * type_sim +
        0.2 * entropy_sim
    )


def are_types_compatible(type1, type2, stats1=None, stats2=None, strict=True):
    """
    Check if two types are compatible based on JSON types.

    Args:
        type1 (str): First type.
        type2 (str): Second type.
        stats1 (dict, optional): Stats for path1.
        stats2 (dict, optional): Stats for path2.
        strict (bool): Whether to use strict compatibility.

    Returns:
        bool: True if compatible, else False.
    """
    if type1 == type2:
        return True

    # Strict mode: scalar types must match exactly
    if strict:
        return False

    # Relaxed mode: allow some structure-based compatibility
    scalar_incompatibles = {
        ("string", "number"), ("number", "string"),
        ("string", "boolean"), ("boolean", "string"),
        ("number", "boolean"), ("boolean", "number"),
    }
    if (type1, type2) in scalar_incompatibles or (type2, type1) in scalar_incompatibles:
        return False

    # Allow object <-> array if array is homogeneous
    if type1 == "object" and type2 == "array":
        return "object" in (stats2 or {}).get("type", [])
    if type2 == "object" and type1 == "array":
        return "object" in (stats1 or {}).get("type", [])

    return True


def match_paths(paths1, paths2, ling_weight=0.5, struct_weight=0.5, min_score=0.7):
    """
    Match paths from two sets of JSON documents based on linguistic and structural similarity.

    Args:
        paths1 (dict): Paths and their statistics from the first set of documents.
        paths2 (dict): Paths and their statistics from the second set of documents.
        ling_weight (float): Weight for linguistic similarity.
        struct_weight (float): Weight for structural similarity.
        min_score (float): Minimum similarity score to consider a match.

    Returns:
        dict: Dictionary where keys are (path1, path2) and values are similarity scores.
    """
    total_weight = ling_weight + struct_weight
    ling_weight /= total_weight
    struct_weight /= total_weight

    matches = {}
    for path1, path1_stats in paths1.items():
        for path2, path2_stats in paths2.items():
            type1 = path1_stats.get("type")
            type2 = path2_stats.get("type")

            if not are_types_compatible(type1, type2, path1_stats, path2_stats):
                continue

            ling_score = linguistic_similarity(path1, path2)
            struct_score = structural_similarity(path1, path2, path1_stats, path2_stats)
            final_score = ling_weight * ling_score + struct_weight * struct_score

            if final_score >= min_score:
                matches[(path1, path2)] = final_score

    return matches





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


def find_valentine(df1, df2, ground_truth):
    """Match source keys to target keys using the Valentine library's Similarity Flooding algorithm.

    Args:
        df1 (pd.DataFrame): The target dataframe containing the keys to be matched.
        df2 (pd.DataFrame): The source dataframe containing the keys to be matched.
        ground_truth (list): A list of tuples containing the ground truth matches, where each tuple is in the form (target_key, source_key).


    Returns:
        match_dict (dict): A dictionary where keys are target keys from df1 and values are lists of matching source keys from df2 based on the ground truth.
    """

    # Instantiate matcher and run the matching algorithm
    matcher = Cupid()
    matches = valentine_match(df1, df2, matcher, "target", "source")
    return matches

    
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


def predict_matches(source_vars):
    """
    Selects final match from optimized source_vars.

    Args:
        source_vars (dict): Maps source paths to list of (var, target_path, score).

    Returns:
        dict: {source_path: (target_path, score)}
    """
    final_matching = {}

    for s_path, var_list in source_vars.items():
        for var, t_path, score in var_list:
            try:
                if hasattr(var, 'X') and round(var.X):
                    final_matching[s_path] = (t_path, score)
                    break
            except Exception as e:
                print(f"[Error] Reading var.X for {s_path}->{t_path}: {e}")

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

            match_var = quadratic_model.addVar(vtype=GRB.BINARY, name=f"{t_path}-----{s_path}")
            score_terms.append(score * match_var)
            source_vars[s_path].append((match_var, t_path, score))

            prefix_key = f'{t_prefix}-----{s_prefix}'
            if prefix_key not in prefix_vars:
                prefix_vars[prefix_key] = quadratic_model.addVar(vtype=GRB.BINARY, name=prefix_key)
            prefix_var = prefix_vars[prefix_key]

            t_nest_key = f'{t_path}-----{s_prefix}'
            if t_nest_key not in nested_t_vars:
                nested_t_vars[t_nest_key] = quadratic_model.addVar(vtype=GRB.BINARY, name=t_nest_key)
            nested_t_var = nested_t_vars[t_nest_key]

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



def pick_random_source_target_pair(original_dir, transformed_dir, seed=42):
    """
    Picks a random file from transformed_dir and finds the matching file in original_dir.
    
    Returns:
        (Path, Path): (source_file_path, target_file_path)
    """
    random.seed(seed)
    transformed_dir = Path(transformed_dir)
    original_dir = Path(original_dir)

    transformed_files = list(transformed_dir.glob("*.json"))
    if not transformed_files:
        raise FileNotFoundError("No files found in transformed directory.")

    target_file = random.choice(transformed_files)
    source_file = original_dir / target_file.name

    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")

    return source_file, target_file


def main():
    args = sys.argv[1:]

    if len(args) != 3:
        print("Usage (auto): python match.py <original_dir> <transformed_dir> <mode>")
        sys.exit(1)

    original_dir = args[0]
    transformed_dir = args[1]
    mode = args[2].lower()

    source_file, target_file = pick_random_source_target_pair(original_dir, transformed_dir, 55)
    print(f"Randomly selected files:\n  Source: {source_file}\n  Target: {target_file}", flush=True)

    # Read and process JSON files
    source_docs = read_json(source_file)
    target_docs = read_json(target_file)
    print(f"Source documents: {len(source_docs)}, Target documents: {len(target_docs)}", flush=True)

    source_paths = extract_paths(source_docs)
    target_paths = extract_paths(target_docs)
    print(f"Source paths: {len(source_paths)}, Target paths: {len(target_paths)}", flush=True)

    # Run matching
    if mode == "cupid":
        matches = find_valentine(
            path_dict_to_df(target_paths),
            path_dict_to_df(source_paths),
            ground_truth=None
        )
    
    elif mode == "jasper":
        matches = match_paths(target_paths, source_paths)
    else:
        raise ValueError("Invalid mode. Use 'cupid' or 'jasper'.")

    # Keep only top 500 matches
    matches = {k: v for k, v in sorted(matches.items(), key=lambda x: -x[1])[:500]}

    # Post-process matches and display
    final_matches = quadratic_programming(matches)
    for s_path, t_path in final_matches.items():
        print(f"source: {s_path}, target: {t_path}", flush=True)


    #metrics = final_matches.get_metrics(ground_truth_path)
    #print(metrics)


if __name__ == "__main__":
    main()