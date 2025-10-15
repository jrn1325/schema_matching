#!/usr/bin/env python3
import argparse
import google.genai as genai
import json
import os
import random
import time
import wordninja
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# Step 1: Path extraction
# ----------------------------
def extract_paths(doc, prefix=""):
    """
    Extract all unique paths from a JSON document.

    Args:
        doc (dict): The JSON document.
        prefix (str): Defaults to "".

    Returns:
        list: List of unique paths.
    """
    paths = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            paths.append(new_prefix)
            paths.extend(extract_paths(v, new_prefix))
    elif isinstance(doc, list):
        if prefix:
            paths.append(prefix)
        for item in doc:
            new_prefix = f"{prefix}.*" if prefix else "*"
            if isinstance(item, (dict, list)):
                paths.extend(extract_paths(item, new_prefix))
    else:
        if prefix:
            paths.append(prefix)
    return paths

def collect_all_paths(docs):
    """
    Collect all unique paths from a list of JSON documents.

    Args:
        docs (list): List of JSON documents.

    Returns:
        list: List of unique paths.
    """
    all_paths = set()
    for doc in docs:
        if isinstance(doc, (dict, list)):
            all_paths.update(extract_paths(doc))
    return sorted(all_paths)

def extract_leaf_key(paths):
    """
    Extract unique leaf keys from a list of paths.

    Args:
        paths (list): List of paths as strings.
    Returns:
        set: Set of unique leaf keys (last key of each path).
    """
    unique_keys = set()
    for path in paths:
        parts = path.split(".")
        if parts[-1] != "*":  # Ignore array placeholders
            unique_keys.add(parts[-1])
    return unique_keys


# ----------------------------
# Step 2: Synonym generation
# ----------------------------
def build_synonym_mapping(unique_keys, batch_size=8, model="gemini-2.5-flash"):
    """
    Build a mapping of JSON keys to their synonyms using gemini model.

    Args:
        unique_keys (set): Set of unique JSON keys.
        batch_size (int): Number of keys to process in one batch. Defaults to 8.
        model (str): Model name. Defaults to "gemini-2.5-flash".

    Returns:
        dict: Mapping of original keys to their synonyms.
    """
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    synonym_map = {}
    keys = list(unique_keys)

    for i in tqdm(range(0, len(keys), batch_size), desc="Building synonyms"):
        batch_keys = keys[i:i+batch_size]
        prompt = (
            "Provide a synonym for each JSON key below. "
            "Return only one word per line, no extra text, in the same order:\n" +
            "\n".join(f"Key: '{key}' → Synonym:" for key in batch_keys)
        )

        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                break
            except Exception as e:
                if "429" in str(e):
                    #print(f"Rate limit exceeded, retrying in {2 ** attempt} seconds...", flush=True)
                    time.sleep(2 ** attempt)
                else:
                    raise

        for key, line in zip(batch_keys, (response.text.split("\n") if response and hasattr(response, "text") else batch_keys)):
            synonym = line.strip() or key
            synonym_map[key] = synonym

    return synonym_map


# ----------------------------
# Step 3: Structural transformations
# ----------------------------
def increase_nesting(doc):
    """
    Increase nesting of dict keys if WordNinja can split them.
    Example: {'participantName': 'Alice'} -> {'participant': {'Name': 'Alice'}}

    Args:
        doc (dict or list): The JSON document.
    Returns:
        dict or list: The transformed document.
    """
    if isinstance(doc, dict):
        new_doc = {}
        for k, v in doc.items():
            # Use WordNinja to check if the key is splittable
            words = wordninja.split(k)
            if len(words) > 1:
                # Build nested dict
                nested = v
                for word in reversed(words[1:]):
                    nested = {word: nested}
                new_doc[words[0]] = increase_nesting(nested)
            else:
                # Keep as is
                new_doc[k] = increase_nesting(v)
        return new_doc
    elif isinstance(doc, list):
        return [increase_nesting(item) for item in doc]
    else:
        return doc

def decrease_nesting(doc, parent_key=""):
    """
    Flatten nested dictionaries by joining keys, except for top-level keys.
    Example: {'participant': {'Name': 'Alice'}} -> {'participantName': 'Alice'}

    Args:
        doc (dict or list): The JSON document.
        parent_key (str, optional): The prefix for keys. Defaults to "".
    Returns:
        dict or list: The transformed document.
    """
    if not parent_key and isinstance(doc, dict):
        # Top-level dict: cannot reduce nesting
        return {k: decrease_nesting(v, k) if isinstance(v, dict) else v for k, v in doc.items()}

    result = {}
    if isinstance(doc, dict):
        for k, v in doc.items():
            new_key = f"{parent_key}{k[0].upper()}{k[1:]}" if parent_key else k
            if isinstance(v, dict):
                nested = decrease_nesting(v, new_key)
                for nk, nv in nested.items():
                    if nk not in result: # Check for key collision
                        result[nk] = nv
                    else:
                        print(f"Key collision detected: {nk}, keeping nested dict under {k}", flush=True)
                        result[k] = v
            else:
                if new_key not in result: # Check for key collision
                    result[new_key] = v
                else:
                    print(f"Key collision detected: {new_key}, keeping original key {k}", flush=True)
                    result[k] = v
    elif isinstance(doc, list):
        result = [decrease_nesting(item, parent_key) if isinstance(item, dict) else item for item in doc]
    else:
        result = doc
    return result


# ----------------------------
# Step 4: Assign transformations
# ----------------------------
def assign_transformations(paths, ratio_l=0.5, ratio_s1=0.25, ratio_s2=0.25, seed=None):
    """
    Assign transformations to paths.

    Args:
        paths (list): List of unique paths.
        ratio_l (float): Ratio for linguistic changes.
        ratio_s1 (float): Ratio for structural increase nesting.
        ratio_s2 (float): Ratio for structural decrease nesting.
        seed (int): Random seed.

    Returns:
        dict: A mapping of path to its transformation details.
    """
    if seed is not None:
        random.seed(seed)

    # Shuffle paths
    n = len(paths)
    shuffled = paths.copy()
    random.shuffle(shuffled)

    # Determine the number of paths for each transformation
    s1_count = int(n * ratio_s1)
    s2_count = int(n * ratio_s2)
    l_count = int(n * ratio_l)

    # Select paths for each transformation and ensure no overlap between s1 and s2
    s1_paths = set(shuffled[:s1_count])
    s2_paths = set(shuffled[s1_count:s1_count+s2_count])
    l_paths = set(random.sample(shuffled, l_count))

    # Sort paths by depth (top-level first)
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
# Step 5: Apply transformations
# ----------------------------
def transform_json(doc, orig_prefix, trans_prefix, transform_map, synonym_map, ground_truth, filename, seen_paths):
    """
    Recursively transform JSON document, applying linguistic and structural changes.
    Propagates renamed prefixes automatically.

    Args:
        doc (dict or list)
        orig_prefix (str): Original path prefix
        trans_prefix (str): Transformed path prefix
        transform_map (dict): Path -> {"linguistic": bool, "structural": str|None}
        synonym_map (dict): Key -> synonym
        ground_truth (list)
        filename (str)
        seen_paths (set)
    Returns:
        Transformed doc
    """
    if isinstance(doc, dict):
        new_obj = {}
        for key, value in doc.items():
            # Get original full path
            orig_path = f"{orig_prefix}.{key}" if orig_prefix else key

            # Get transformation rule for this path
            rule = transform_map.get(orig_path, {"linguistic": False, "structural": None})

            # Apply linguistic transformation
            new_key = synonym_map.get(key, key) if rule["linguistic"] else key

            # Compute transformed prefix
            trans_path = f"{trans_prefix}.{new_key}" if trans_prefix else new_key

            # Apply structural transformations
            if rule["structural"] == "increase_nesting":
                words = wordninja.split(new_key)
                if len(words) > 1:
                    nested_value = transform_json(value, orig_path, trans_path, transform_map, synonym_map, ground_truth, filename, seen_paths)
                    for w in reversed(words[1:]):
                        nested_value = {w: nested_value}
                    new_obj[words[0]] = nested_value
                    new_key = words[0]
                    trans_path = f"{trans_prefix}.{new_key}" if trans_prefix else new_key
                else:
                    new_obj[new_key] = transform_json(value, orig_path, trans_path, transform_map, synonym_map, ground_truth, filename, seen_paths)
                    rule["structural"] = "Can't increase"

            elif rule["structural"] == "reduce_nesting" and isinstance(value, dict):
                new_obj[new_key] = decrease_nesting(value, parent_key=new_key)
            else:
                new_obj[new_key] = transform_json(value, orig_path, trans_path, transform_map, synonym_map, ground_truth, filename, seen_paths)

            # Record ground truth once per unique path
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
        return [transform_json(x, orig_prefix, trans_prefix, transform_map, synonym_map, ground_truth, filename, seen_paths) for x in doc]
    else:
        return doc

    
def apply_transformations(docs, transform_map, synonym_map, filename):
    """
    Apply transformations to all documents, maintaining parent->child path propagation.
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
# Step 6: Load/save JSON
# ----------------------------
def load_json_lines(filename):
    """
    Load JSON lines from a file.

    Args:
        filename (str): Path to the file.
    Yields:
        dict: Each JSON object.
    """
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def save_json_lines(filename, docs):
    """
    Save a list of JSON documents to a file in JSON lines format.

    Args:
        filename (str): Path to the file.
        docs (list): List of JSON documents.
    """ 
    with open(filename, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

def save_ground_truth(ground_truth, output_file):
    """
    Save ground truth mappings to a file in JSON lines format.

    Args:
        ground_truth (list): List of ground truth mappings.
        output_file (str): Path to the output file.
    """ 
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "a") as f:
        for entry in ground_truth:
            f.write(json.dumps(entry) + "\n")


# ----------------------------
# Step 7: Process one file
# ----------------------------
def process_one_file(file_path, output_dir, ling_ratio, struct1_ratio, struct2_ratio, batch_size=8, seed=42):
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
    synonym_map = build_synonym_mapping(unique_keys, batch_size=batch_size)

    # 3. Assign transformations
    transform_map = assign_transformations(paths, ratio_l=ling_ratio, ratio_s1=struct1_ratio, ratio_s2=struct2_ratio, seed=seed)

    # 4. Apply transformations recursively
    transformed_docs, ground_truth = apply_transformations(docs, transform_map, synonym_map, file_path.name)

    # 5. Save results
    save_json_lines(output_file, transformed_docs)
    print(f"Processed {file_path.name}: {len(docs)} documents transformed.", flush=True)
    return ground_truth


# ----------------------------
# Step 8: Process all datasets
# ----------------------------
def process_datasets(input_dir, output_dir, groundtruth_file, sample_size=2,
                     ling_ratio=0.5, struct1_ratio=0.25, struct2_ratio=0.25,
                     batch_size=8, seed=42, n_jobs=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groundtruth_file = Path(groundtruth_file)
    groundtruth_file.parent.mkdir(parents=True, exist_ok=True)

    # List all JSON files in the directory
    all_files = list(input_dir.glob("*.json"))
    if not all_files:
        print("No JSON files found in input directory.")
        return

    # Filter out files that already have been transformed 
    unprocessed_files = [
        f for f in all_files if not (output_dir / f.name).exists()
    ]

    if not unprocessed_files:
        print("All files have already been transformed.")
        return

    # Sample files to process only from the unprocessed set
    random.seed(seed)
    selected_files = random.sample(unprocessed_files, min(sample_size, len(unprocessed_files)))
    print(f"Selected {len(selected_files)} files for processing.")

    all_ground_truth = []

    # Run in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(
                process_one_file, file_path, output_dir,
                ling_ratio, struct1_ratio, struct2_ratio,
                batch_size, seed
            ): file_path
            for file_path in selected_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            file_path = futures[future]
            try:
                file_ground_truth = future.result()
                all_ground_truth.extend(file_ground_truth)
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

    # Save single consolidated ground truth file
    save_ground_truth(all_ground_truth, groundtruth_file)
    print(f"Processed {len(selected_files)} files and saved ground truth.")
# ----------------------------
# Step 9: CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Transform JSON keys and structure.")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("groundtruth_file")
    parser.add_argument("--sample_size", type=int, default=50)
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
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        groundtruth_file=args.groundtruth_file,
        sample_size=args.sample_size,
        ling_ratio=args.ling_ratio,
        struct1_ratio=args.struct1_ratio,
        struct2_ratio=args.struct2_ratio,
        batch_size=args.batch_size,
        seed=args.seed
    )
    print(f"Finished in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    main()
