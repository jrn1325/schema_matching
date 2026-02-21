#!/usr/bin/env python3
import argparse
import google.genai as genai
import json
import os
import random
import time
import wordninja
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# Constants
# ----------------------------
CACHE_PATH = "synonym_cache.json"
CACHE_LOCK_PATH = "synonym_cache.json.lock"
IN_PROGRESS_MARKER = "__IN_PROGRESS__"
ARRAY_WILDCARD = "<ARRAY_ITEM>"

# ----------------------------
# Step 1: Path extraction
# ----------------------------
def extract_paths(doc, prefix=""):
    paths = set()
    if isinstance(doc, dict):
        for k, v in doc.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            paths.add(new_prefix)
            paths.update(extract_paths(v, new_prefix))
    elif isinstance(doc, list):
        for item in doc:
            new_prefix = f"{prefix}.{ARRAY_WILDCARD}" if prefix else ARRAY_WILDCARD
            paths.update(extract_paths(item, new_prefix))
    else:
        if prefix:
            paths.add(prefix)
    return paths

def collect_all_paths(docs):
    all_paths = set()
    for doc in docs:
        if isinstance(doc, (dict, list)):
            all_paths.update(extract_paths(doc))
    return sorted(all_paths)

def extract_leaf_key(paths):
    unique_keys = set()
    for path in paths:
        parts = path.split(".")
        if parts[-1] != ARRAY_WILDCARD:
            unique_keys.add(parts[-1])
    return unique_keys

# ----------------------------
# Step 2: Gemini client
# ----------------------------
_client = None
def get_gemini_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _client

# ----------------------------
# Step 3: Synonym generation (parallel-safe)
# ----------------------------
def build_synonym_mapping(unique_keys, cache_path="synonym_cache.json", batch_size=8, model="gemini-2.5-flash"):
    """
    Build or extend a mapping of JSON keys to their synonyms using Gemini.

    Args:
        unique_keys (list): List of unique JSON keys to process.
        cache_path (str): Path to the synonym cache file.
        batch_size (int): Number of keys to process in each Gemini request.
        model (str): The Gemini model to use for synonym generation.
    
    Returns:
        tuple(dict, bool): (synonym_map, all_keys_have_synonyms)
    """
    client = get_gemini_client()

    # Load existing cache
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            synonym_map = json.load(f)
    else:
        synonym_map = {}

    # Only process keys that are missing or empty
    pending_keys = [k for k in unique_keys if k not in synonym_map or not synonym_map[k]]
    print(f"{len(pending_keys)} new keys to process (out of {len(unique_keys)} total).")

    for i in tqdm(range(0, len(pending_keys), batch_size), desc="Building synonyms"):
        batch_keys = pending_keys[i:i + batch_size]
        prompt = (
            "Provide a synonym for each JSON key below. "
            "Return only ONE WORD per line (no punctuation, no explanations). "
            "Do NOT repeat the same key as its synonym. "
            "Order must match the given keys.\n\n"
            + "\n".join(batch_keys)
        )

        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                break
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "quota" in msg or "rate" in msg:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    print(f"Rate limited — retrying in {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    print(f"Unexpected error: {e}")
                    raise

        if not response or not hasattr(response, "text") or not response.text.strip():
            print("No valid response — likely quota exhausted. Stopping.")
            break

        lines = [l.strip() for l in response.text.strip().split("\n") if l.strip()]
        if len(lines) != len(batch_keys):
            print(f"Synonym count mismatch ({len(lines)} vs {len(batch_keys)}). Stopping.")
            break

        # Validate synonyms
        for key, synonym in zip(batch_keys, lines):
            if not synonym or synonym.lower() == key.lower():
                print(f"Invalid synonym for '{key}' — leaving empty in cache.")
                synonym_map[key] = ""
            else:
                synonym_map[key] = synonym

        # Save progress after each batch
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(synonym_map, f, ensure_ascii=False, indent=2)

        time.sleep(random.uniform(0.5, 1.5))

    # Check if all keys now have synonyms
    all_keys_have_synonyms = all(synonym_map.get(k) for k in unique_keys)
    return synonym_map, all_keys_have_synonyms

# ----------------------------
# Step 4: Structural transformations
# ----------------------------
def increase_nesting(doc):
    """
    Increase nesting of JSON keys based on word splits.

    Args:
        doc (dict): The JSON document.
    Returns:
        dict: The transformed JSON document with increased nesting.
    """
    if isinstance(doc, dict):
        new_doc = {}
        for k, v in doc.items():
            words = wordninja.split(k)
            if len(words) > 1:
                nested = v
                for word in reversed(words[1:]):
                    nested = {word: nested}
                new_doc[words[0]] = increase_nesting(nested)
            else:
                new_doc[k] = increase_nesting(v)
        return new_doc
    elif isinstance(doc, list):
        return [increase_nesting(item) for item in doc]
    else:
        return doc

def decrease_nesting(doc, parent_key=""):
    """
    Decrease nesting of JSON keys by flattening single-key dictionaries.

    Args:
        doc (dict): The JSON document.
        parent_key (str): The parent key to prefix to child keys.   
    Returns:
        dict: The transformed JSON document with decreased nesting.
    """
    if not parent_key and isinstance(doc, dict):
        return {k: decrease_nesting(v, k) if isinstance(v, dict) else v for k, v in doc.items()}

    result = {}
    if isinstance(doc, dict):
        for k, v in doc.items():
            new_key = f"{parent_key}_{k}" if parent_key else k

            if isinstance(v, dict):
                nested = decrease_nesting(v, new_key)
                for nk, nv in nested.items():
                    if nk not in result:
                        result[nk] = nv
                    else:
                        result[k] = v
            else:
                if new_key not in result:
                    result[new_key] = v
                else:
                    result[k] = v
    elif isinstance(doc, list):
        result = [decrease_nesting(item, parent_key) if isinstance(item, dict) else item for item in doc]
    else:
        result = doc
    return result

# ----------------------------
# Step 5: Assign transformations
# ----------------------------
def assign_transformations(paths, ratio_l=0.5, ratio_s1=0.25, ratio_s2=0.25, seed=None):
    """
    Assign transformations to JSON paths based on specified ratios.

    Args:
        paths (list): List of JSON paths.
        ratio_l (float): Ratio of paths to apply linguistic transformations.
        ratio_s1 (float): Ratio of paths to apply structural increase nesting.
        ratio_s2 (float): Ratio of paths to apply structural decrease nesting.
        seed (int): Random seed for reproducibility.    
    Returns:
        dict: Mapping of paths to their assigned transformations.
    """

    if seed is not None:
        random.seed(seed)
    n = len(paths)
    shuffled = paths.copy()
    random.shuffle(shuffled)
    s1_count = int(n * ratio_s1)
    s2_count = int(n * ratio_s2)
    l_count = int(n * ratio_l)
    s1_paths = set(shuffled[:s1_count])
    s2_paths = set(shuffled[s1_count:s1_count+s2_count])
    l_paths = set(random.sample(shuffled, l_count))
    paths_sorted = sorted(paths, key=lambda p: len(p.split(".")))
    transform_map = {}
    for path in paths_sorted:
        transform_map[path] = {
            "linguistic": path in l_paths,
            "structural": (
                "increase_nesting" if path in s1_paths else
                "reduce_nesting" if path in s2_paths else None
            )
        }
    return transform_map

# ----------------------------
# Step 6: Apply transformations
# ----------------------------
def transform_json(doc, orig_prefix, trans_prefix, transform_map, synonym_map, ground_truth, filename, seen_paths):
    """
    Recursively transform a JSON document based on the transformation map and synonym map.

    Args:
        doc (dict or list): The JSON document to transform.
        orig_prefix (str): The original path prefix
        trans_prefix (str): The transformed path prefix
        transform_map (dict): Mapping of paths to their transformations.
        synonym_map (dict): Mapping of original keys to their synonyms.
        ground_truth (list): List to record the transformations applied.
        filename (str): The name of the file being processed.
        seen_paths (set): Set to track already processed paths for ground truth.
    Returns:
        dict or list: The transformed JSON document.
    """
    if isinstance(doc, dict):
        new_obj = {}
        for key, value in doc.items():
            orig_path = f"{orig_prefix}.{key}" if orig_prefix else key
            rule = transform_map.get(orig_path, {"linguistic": False, "structural": None})
            new_key = synonym_map.get(key, key) if rule["linguistic"] else key
            trans_path = f"{trans_prefix}.{new_key}" if trans_prefix else new_key

            if rule["structural"] == "increase_nesting":
                words = wordninja.split(new_key)
                if len(words) > 1:
                    nested_value = transform_json(value, orig_path, trans_path, transform_map, synonym_map, ground_truth, filename, seen_paths)
                    for w in reversed(words[1:]):
                        nested_value = {w: nested_value}
                    first_word = words[0]
                    if first_word in new_obj and isinstance(new_obj[first_word], dict):
                        new_obj[first_word] = {**new_obj[first_word], **nested_value}
                    else:
                        new_obj[first_word] = nested_value
                    new_key = words[0]
                    trans_path = f"{trans_prefix}.{new_key}" if trans_prefix else new_key
                else:
                    new_obj[new_key] = transform_json(value, orig_path, trans_path, transform_map, synonym_map, ground_truth, filename, seen_paths)
            elif rule["structural"] == "reduce_nesting" and isinstance(value, dict):
                new_obj[new_key] = decrease_nesting(value, parent_key=new_key)
            else:
                new_obj[new_key] = transform_json(value, orig_path, trans_path, transform_map, synonym_map, ground_truth, filename, seen_paths)

            key_id = (filename, orig_path)
            if key_id not in seen_paths:
                seen_paths.add(key_id)
                ground_truth.append({
                    "filename": filename,
                    "original_path": orig_path,
                    "transformed_path": trans_path,
                    "linguistic": rule["linguistic"],
                    "structural": rule["structural"]
                })
        return new_obj
    elif isinstance(doc, list):
        array_orig = f"{orig_prefix}.{ARRAY_WILDCARD}" if orig_prefix else ARRAY_WILDCARD
        array_trans = f"{trans_prefix}.{ARRAY_WILDCARD}" if trans_prefix else ARRAY_WILDCARD

        return [
            transform_json(
                item,
                array_orig,
                array_trans,
                transform_map,
                synonym_map,
                ground_truth,
                filename,
                seen_paths
            )
            for item in doc
        ]
    else:
        return doc

def apply_transformations(docs, transform_map, synonym_map, filename):
    """
    Apply transformations to a list of JSON documents.

    Args:
        docs (list): List of JSON documents.
        transform_map (dict): Mapping of paths to their transformations.
        synonym_map (dict): Mapping of original keys to their synonyms.
        filename (str): The name of the file being processed.
    Returns:
        tuple: (transformed_docs, ground_truth)
    """
    transformed_docs = []
    ground_truth = []
    seen_paths = set()
    for doc in docs:
        transformed_docs.append(
            transform_json(doc, "", "", transform_map, synonym_map, ground_truth, filename, seen_paths)
        )
    return transformed_docs, ground_truth

# ----------------------------
# Step 7: JSON I/O
# ----------------------------
def load_json_lines(filename):
    """
    Load JSON documents from a file with one JSON object per line.

    Args:
        filename (str): The path to the JSON file.
    Yields:
        dict: The JSON document.
    """
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def save_json_lines(filename, docs):
    """
    Save JSON documents to a file with one JSON object per line.
    
    Args:
        filename (str): The path to the output JSON file.
        docs (list): List of JSON documents to save.
    """
    with open(filename, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

def save_ground_truth(ground_truth, output_file):
    """
    Save ground truth data to a file with one JSON object per line.

    Args:
        ground_truth (list): List of ground truth entries.
        output_file (str or Path): The path to the output ground truth file.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a") as f:
        for entry in ground_truth:
            f.write(json.dumps(entry) + "\n")

# ----------------------------
# Step 8: Process datasets
# ----------------------------
def process_one_file(file_path, output_dir, ling_ratio, struct1_ratio, struct2_ratio, batch_size=8, seed=42):
    """
    Process a single JSON file: transform its keys and structure.

    Args:
        file_path (Path): Path to the input JSON file.
        output_dir (Path): Directory to save the transformed JSON file.
        ling_ratio (float): Ratio of keys to apply linguistic transformations.
        struct1_ratio (float): Ratio of keys to apply structural increase nesting.
        struct2_ratio (float): Ratio of keys to apply structural decrease nesting.
        batch_size (int): Batch size for synonym generation.
        seed (int): Random seed for reproducibility.        
    Returns:
        list: Ground truth entries for the transformations applied.
    """
    output_file = output_dir / file_path.name
    if output_file.exists():
        print(f"Skipping {file_path.name} (already exists).")
        return []

    docs = list(load_json_lines(file_path))
    if not docs:
        print(f"No documents in {file_path.name}, skipping.")
        return []

    # 1. Collect all unique paths across all docs
    paths = collect_all_paths(docs)
    unique_keys = extract_leaf_key(paths)

    # 2. Build synonym map
    synonym_map, all_keys_synonyms_ready = build_synonym_mapping(
        unique_keys, cache_path="synonym_cache.json", batch_size=batch_size
    )

    if not all_keys_synonyms_ready:
        print(f"Skipping {file_path.name} — not all keys have synonyms yet.")
        return []

    # 3. Assign transformations
    transform_map = assign_transformations(
        paths, ratio_l=ling_ratio, ratio_s1=struct1_ratio, ratio_s2=struct2_ratio, seed=seed
    )

    # 4. Apply transformations recursively
    transformed_docs, ground_truth = apply_transformations(
        docs, transform_map, synonym_map, file_path.name
    )

    # 5. Save results
    save_json_lines(output_file, transformed_docs)
    print(f"Processed {file_path.name}: {len(docs)} documents transformed.", flush=True)
    return ground_truth

def process_datasets(input_dir, output_dir, groundtruth_file, sample_size=2,
                     ling_ratio=0.5, struct1_ratio=0.25, struct2_ratio=0.25,
                     batch_size=8, seed=42, n_jobs=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groundtruth_file = Path(groundtruth_file)
    groundtruth_file.parent.mkdir(parents=True, exist_ok=True)

    all_files = list(input_dir.glob("*.json"))
    if not all_files:
        print("No JSON files found in input directory.")
        return

    unprocessed_files = [f for f in all_files if not (output_dir / f.name).exists()]
    if not unprocessed_files:
        print("All files have already been transformed.")
        return

    random.seed(seed)
    selected_files = random.sample(unprocessed_files, min(sample_size, len(unprocessed_files)))
    print(f"Selected {len(selected_files)} files for processing.")

    for file_path in tqdm(selected_files, desc="Processing files"):
        try:
            file_ground_truth = process_one_file(file_path, output_dir, ling_ratio, struct1_ratio, struct2_ratio, batch_size, seed)
            save_ground_truth(file_ground_truth, groundtruth_file)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            time.sleep(1)

    print(f"Processed {len(selected_files)} files and saved ground truth.")

# ----------------------------
# Step 9: CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Transform JSON keys and structure.")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("groundtruth_file")
    parser.add_argument("--sample_size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ling_ratio", type=float, default=0.5)
    parser.add_argument("--struct1_ratio", type=float, default=0.25)
    parser.add_argument("--struct2_ratio", type=float, default=0.25)
    return parser.parse_args()

def main():
    args = parse_args()
    start = time.time()
    process_datasets(
        input_dir=args.input_dir, output_dir=args.output_dir, groundtruth_file=args.groundtruth_file,
        sample_size=args.sample_size, ling_ratio=args.ling_ratio, struct1_ratio=args.struct1_ratio,
        struct2_ratio=args.struct2_ratio, batch_size=args.batch_size, seed=args.seed
    )
    print(f"Finished in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    main()
