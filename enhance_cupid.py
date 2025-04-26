import collections
import csm
import json
import math
import pandas as pd
import random
import sys
import torch
import torch.nn.functional as F


from collections import defaultdict, Counter
from transformers import RobertaTokenizer, RobertaModel

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
    Calculate the entropy of keys in a dictionary.

    Args:
        keys (list): List of keys.
    Returns:
        float: The entropy value.
    """
    if not keys:
        return 0.0
    counter = Counter(keys)
    total = sum(counter.values())
    probs = [count / total for count in counter.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def extract_paths(docs):
    """
    Extract all paths from JSON documents.

    Args:
        docs (list): A list of JSON documents.

    Returns:
        dict: A dictionary where the keys are paths and the values are sets of corresponding values.
    """
    path_stats = defaultdict(lambda: {
        "values": set(),
        "types": Counter(),
        "num_siblings": [],
        "key_entropy": [],
        "num_children": [],
        "nesting_depth": 0,
    })

    for doc in docs:
        for path, value, siblings in parse_document(doc):
            parts = path.split('.')
            depth = len(parts)
            entry = path_stats[path]
            entry["values"].add(json.dumps(value, sort_keys=True))
            entry["types"][infer_type(value)] += 1
            entry["num_siblings"].append(siblings)
            entry["nesting_depth"] = depth

            if isinstance(value, dict):
                keys = list(value.keys())
                entropy = key_entropy(keys)
                entry["key_entropy"].append(entropy)
                entry["num_children"].append(len(keys))

    # Calculate averages
    for entry in path_stats.values():
        entry["num_siblings"] = sum(entry["num_siblings"]) / len(entry["num_siblings"]) if entry["num_siblings"] else 0
        entry["key_entropy"] = sum(entry["key_entropy"]) / len(entry["key_entropy"]) if entry["key_entropy"] else 0
        entry["num_children"] = sum(entry["num_children"]) / len(entry["num_children"]) if entry["num_children"] else 0
        entry["type"] = entry["types"].most_common(1)[0][0]
        del entry["types"]

    return path_stats


def read_json(file_path, seed=42):
    """
    Read a JSONL file and return a list of JSON objects.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        list: A list of JSON objects.
    """
    with open(file_path) as f:
        docs = [json.loads(line) for line in f]

    random.seed(seed)
    sample_docs = random.sample(docs, 3)
    return sample_docs


def compute_embeddings(path):
    """
    Compute embeddings for a JSON path using a pre-trained model.

    Args:
        path (str): A JSON path string.

    Returns:
        torch.Tensor: A tensor of shape (1, embedding_dim).
    """
    inputs = tokenizer(path, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


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
    
    # Join keys into space-separated strings
    path1_text = ' '.join(path1)
    path2_text = ' '.join(path2)
    
    emb1 = compute_embeddings(path1_text)
    emb2 = compute_embeddings(path2_text)
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
    depth_diff = abs(len(keys1) - len(keys2))
    return 1 / (1 + depth_diff**2)


def divergence_penalty(keys1, keys2):
    """
    Penalize paths that diverge earlier in the hierarchy.

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
    Compare the number of sibling keys at the same level.

    Args:
        num_siblings1 (int): Sibling count for path1.
        num_siblings2 (int): Sibling count for path2.

    Returns:
        float: Similarity score (0 to 1).
    """
    diff = abs(num_siblings1 - num_siblings2)
    return 1 / (1 + diff)


def type_similarity(type1, type2):
    """
    Compare the dominant data types under each path.

    Args:
        type1 (str): Data type of path1.
        type2 (str): Data type of path2.

    Returns:
        float: 1.0 if same type, else 0.0.
    """
    return 1.0 if type1 == type2 else 0.0


def key_entropy_similarity(entropy1, entropy2):
    """
    Compare key entropy to understand variability of nested structures.

    Args:
        entropy1 (float): Key entropy at path1.
        entropy2 (float): Key entropy at path2.

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
        paths1_stats (dict): Statistics for the first path.
        paths2_stats (dict): Statistics for the second path.

    Returns:
        float: A similarity score between 0 and 1.
    """
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


def are_types_compatible(type1, type2, stats1=None, stats2=None):
    """
    Check if two types are compatible based on their JSON schema types.
    Args:
        type1 (str): The first type.
        type2 (str): The second type.
        stats1 (dict, optional): Statistics for the first path.
        stats2 (dict, optional): Statistics for the second path.
    Returns:
        bool: True if types are compatible, False otherwise.
    """
    # Normalize types
    t1, t2 = type1.lower(), type2.lower()

    incompatible = {("string", "number"), ("number", "string"),
                    ("string", "boolean"), ("boolean", "string"),
                    ("number", "boolean"), ("boolean", "number")}

    # Disallow incompatible scalar types
    if (t1, t2) in incompatible or (t2, t1) in incompatible:
        return False

    # Allow string <-> array if array contains strings
    if t1 == "string" and t2 == "array":
        return "string" in stats2.get("type", [])
    if t2 == "string" and t1 == "array":
        return "string" in stats1.get("type", [])

    return True


def match_paths(paths1, paths2):
    """
    Match paths from two sets of JSON documents based on linguistic and structural similarity.
    Args:
        paths1 (dict): Paths and their statistics from the first set of documents.
        paths2 (dict): Paths and their statistics from the second set of documents.
    Returns:
        dict: A dictionary where keys are tuples of matched paths and values are their similarity scores.
    """
    matches = {}
    for path1, path1_stats in paths1.items():
        for path2, path2_stats in paths2.items():
            type1 = path1_stats.get("type")
            type2 = path2_stats.get("type")

            if not are_types_compatible(type1, type2, path1_stats, path2_stats):
                continue

            ling_score = linguistic_similarity(path1, path2)
            struct_score = structural_similarity(path1, path2, path1_stats, path2_stats)
            final_score = 0.5 * ling_score + 0.5 * struct_score
            matches[(path1, path2)] = final_score

    return matches


def main():
    sample_source_docs = read_json("./files/source_file.json")
    sample_target_docs = read_json("./files/target_file.json")
    paths1 = extract_paths(sample_source_docs)
    paths2 = extract_paths(sample_target_docs)
    matches = match_paths(paths1, paths2)

    #for (path1, path2), score in matches.items():
    #    print(f"Path 1: {path1}, Path 2: {path2}, Score: {score:.4f}")
    #sys.exit(0)
    final_matches = csm.quadratic_programming(matches)

    for s_path, t_path in final_matches.items():
            print(f"Final source path: {s_path}, Final target path: {t_path}")
    #for path1, path2, score in matches:
    #    print(f"Path 1: {path1}, Path 2: {path2}, Score: {score:.4f}")


if __name__ == "__main__":
    main()