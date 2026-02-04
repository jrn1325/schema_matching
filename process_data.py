import argparse
import base64
import csv
import json
import math
import numpy as np
import time
import torch
import transformers

from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm


ARRAY_WILDCARD = "<ARRAY_ITEM>"
MODEL_NAME = "microsoft/codebert-base"
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = transformers.AutoModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ----------------------------
# JSON path extraction and stats
# ----------------------------
def clean_object(obj):
    """
    Recursively copies the object and replaces newlines in all strings.
    """
    if isinstance(obj, dict):
        return {k: clean_object(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_object(v) for v in obj]
    elif isinstance(obj, str):
        return obj.replace("\n", "_NEW_LINE_CHARACTER_")
    else:
        return obj
    
def parse_document(doc, path=''):
    """
    Parse JSON document and yield (path, value, siblings).

    Args:
        doc: JSON document (dict, list, or primitive)
    Yields:
        (path, value, siblings): tuple of path string, value, and number of siblings
    """
    if isinstance(doc, dict):
        siblings = len(doc)
        for key, value in doc.items():
            full_path = f"{path}.{key}" if path else key
            cleaned_value = clean_object(value)
            
            yield (full_path, cleaned_value, siblings)
            yield from parse_document(cleaned_value, full_path)
    elif isinstance(doc, list):
        siblings = len(doc)
        for item in doc:
            full_path = f"{path}.{ARRAY_WILDCARD}"
            yield from parse_document(item, full_path)
    else:
        yield (path, doc, 0)

def normalize_value(value):
    """
    Normalize value for consistent representation.
    
    Args:
        value: JSON value
    Returns:
        normalized string representation
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return str(value)
    return json.dumps(value, sort_keys=True)

def infer_type(value):
    """
    Infer JSON type of a value.
    
    Args:
        value: JSON value
    Returns:
        type as string: "object", "array", "string", "number", "integer", "boolean", "null", or "unknown"
    """
    if isinstance(value, dict):
        return "object"
    elif isinstance(value, list):
        return "array"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "number"
    elif isinstance(value, str):
        return "string"
    elif value is None:
        return "null"
    return "unknown"

def key_entropy(keys):
    """
    Compute entropy of keys in a dict.

    Args:
        keys: list of keys
    Returns:
        entropy value
    """
    if not keys:
        return 0.0
    counter = Counter(keys)
    total = len(keys)
    return -sum((count / total) * math.log2(count / total) for count in counter.values())

def calc_embeddings(values, model, tokenizer, batch_size=64):
    """
    Calculate embeddings for a list of values using a transformer model.

    Args:
        values: list of string values.
        model: transformer model.
        tokenizer: transformer tokenizer.
        batch_size: batch size for processing.
    Returns:    
        average embedding vector
    """

    all_embeddings = []
    for i in range(0, len(values), batch_size):
        batch = values[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state.mean(dim=1)
        all_embeddings.append(emb)
    all_embeddings = torch.cat(all_embeddings, dim=0)  
    avg_embedding = all_embeddings.mean(dim=0)         
    return avg_embedding

def extract_paths(docs):
    """
    Extract paths and their stats from a list of JSON documents.

    Args:
        docs: list of JSON documents.
    Returns:
        dict of path stats
    """

    path_stats = defaultdict(lambda: {
        "types": set(),
        "items_type": set(),
        "child_keys": set(),
        "freq": 0,
        "doc_count": 0,
        "nesting_depth": 0,
        "num_siblings_sum": 0,
        "num_siblings_count": 0,
        "key_entropy_sum": 0,
        "key_entropy_count": 0,
        "num_children_sum": 0,
        "num_children_count": 0,
        "values": set(),
    })

    num_docs = len(docs)
    for doc in docs:
        seen_paths = set()
        for path, value, siblings in parse_document(doc):
            parts = path.split('.')
            depth = len(parts)
            entry = path_stats[path]

            entry["types"].add(infer_type(value))
            entry["freq"] += 1
            entry["num_siblings_sum"] += siblings
            entry["num_siblings_count"] += 1
            entry["nesting_depth"] = max(entry["nesting_depth"], depth)
            entry["values"].add(normalize_value(value))

            if isinstance(value, dict):
                keys = list(value.keys())
                entry["child_keys"].update(keys)
                entry["key_entropy_sum"] += key_entropy(keys)
                entry["key_entropy_count"] += 1
                entry["num_children_sum"] += len(keys)
                entry["num_children_count"] += 1
            elif isinstance(value, list) and value:
                for item in value:
                    entry["items_type"].add(infer_type(item))

            seen_paths.add(path)

        for path in seen_paths:
            path_stats[path]["doc_count"] += 1

    final = {}
    for path, entry in path_stats.items():
        final[path] = {
            "types": list(entry["types"]),
            "items_type": list(entry.get("items_type", [])),
            "child_keys": list(entry.get("child_keys", [])),
            "num_siblings": entry["num_siblings_sum"] / entry["num_siblings_count"] if entry["num_siblings_count"] else 0,
            "key_entropy": entry["key_entropy_sum"] / entry["key_entropy_count"] if entry["key_entropy_count"] else 0,
            "num_children": entry["num_children_sum"] / entry["num_children_count"] if entry["num_children_count"] else 0,
            "nesting_depth": entry["nesting_depth"],
            "freq": entry["freq"],
            "norm_freq": entry["doc_count"] / num_docs if num_docs else 0,
            "values": entry["values"],
        }
    return final


# ----------------------------
# File processing
# ----------------------------
def process_file(file_path):
    """
    Process a single JSON file and extract path stats.

    Args:
        file_path: path to JSON file    
    Returns:
        dict of path stats
    """
    file_path = Path(file_path)
    try:
        with open(file_path, encoding="utf-8") as f:
            docs = [json.loads(line) for line in f if line.strip()]
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        return None

    return file_path.name, extract_paths(docs)

def encode_embedding(vec):
    """
    Convert numpy array or list of floats to base64 string.

    Args:
        vec: numpy array or list of floats.
    Returns:
        base64 encoded string
    """
    if isinstance(vec, list):
        vec = np.array(vec, dtype=np.float32)
    if isinstance(vec, np.ndarray):
        return base64.b64encode(vec.astype(np.float32).tobytes()).decode('utf-8')
    return ""

def process_files(source_dir, target_dir, source_output_dir, target_output_dir):
    """
    Process files in the source and target directories, extracting path stats and saving to CSV.
    Creates one CSV per JSON file in the output directory.
    
    Args:
        source_dir: directory with source JSON files.
        target_dir: directory with target JSON files.
        source_output_dir: directory to store CSVs for source files.
        target_output_dir: directory to store CSVs for target files.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    source_output_dir = Path(source_output_dir)
    target_output_dir = Path(target_output_dir)

    source_output_dir.mkdir(parents=True, exist_ok=True)
    target_output_dir.mkdir(parents=True, exist_ok=True)

    # Get all JSON files
    source_files = sorted(source_dir.glob("*.json"))
    target_files = sorted(target_dir.glob("*.json"))

    # Build a set of target filenames for fast lookup
    target_filenames = {f.name for f in target_files}

    # Keep only source files that exist in target
    source_files = [f for f in source_files if f.name in target_filenames]

    if len(source_files) != len(target_files):
        print("Warning: Number of matched source and target files does not match.")

    for split_name, files, output_dir in [
        ("source", source_files, source_output_dir),
        ("target", target_files, target_output_dir)
    ]:
        print(f"\n Parsing {split_name} files in parallel ({len(files)} files)...")
        results = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_file, f) for f in files]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Parsing {split_name}"):
                res = fut.result()
                if res:
                    results.append(res)

        # For each file, create a separate CSV
        for filename, path_stats in tqdm(results, desc=f"Writing {split_name} CSVs", total=len(results)):
            all_rows = []
            for path, stats in path_stats.items():
                path_emb = calc_embeddings([path], model, tokenizer).cpu().numpy()
                values_emb = calc_embeddings(list(stats["values"]), model, tokenizer).cpu().numpy() if stats["values"] else np.zeros(768)

                all_rows.append({
                    "filename": filename,
                    "path": path,
                    "types": list(stats.get("types", [])),
                    "child_keys": list(stats.get("child_keys", [])),
                    "num_siblings": stats.get("num_siblings", 0),
                    "key_entropy": stats.get("key_entropy", 0),
                    "num_children": stats.get("num_children", 0),
                    "nesting_depth": stats.get("nesting_depth", 0),
                    "freq": stats.get("freq", 0),
                    "norm_freq": stats.get("norm_freq", 0),
                    "values": list(stats.get("values", [])),
                    "path_emb": encode_embedding(path_emb),
                    "values_emb": encode_embedding(values_emb),
                })

            output_csv = output_dir / f"{Path(filename).stem}.csv"
            with open(output_csv, "w", newline="", encoding="utf-8") as fout:
                writer = csv.DictWriter(fout, fieldnames=[
                    "filename", "path", "types", "child_keys",
                    "num_siblings", "key_entropy", "num_children", "nesting_depth",
                    "freq", "norm_freq", "values", "path_emb", "values_emb"
                ], delimiter=";")
                writer.writeheader()
                writer.writerows(all_rows)



# ----------------------------
# CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Extract JSON path statistics to CSV (one CSV per JSON file)")
    parser.add_argument("source_dir", help="Directory with source JSON files")
    parser.add_argument("target_dir", help="Directory with target JSON files")
    parser.add_argument("source_output_dir", help="Directory to store CSVs for source files")
    parser.add_argument("target_output_dir", help="Directory to store CSVs for target files")
    return parser.parse_args()

def main():
    args = parse_args()
    start = time.time()
    process_files(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        source_output_dir=args.source_output_dir,
        target_output_dir=args.target_output_dir,
    )
    print("Completed in %.2f seconds" % (time.time() - start))

if __name__ == "__main__":
    main()
