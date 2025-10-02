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
    Extract all unique paths from a JSON-like document.

    Args:
        doc (dict): The JSON document.
        prefix (str, optional): Defaults to "".

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
        new_prefix = f"{prefix}.*" if prefix else "*"
        paths.append(new_prefix)
        for item in doc:
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
        all_paths.update(extract_paths(doc))
    return sorted(all_paths)

def extract_leaf_key(paths):
    """
    Extract unique leaf keys from a list of paths.

    Args:
        paths (list): List of paths as strings.
    Returns:
        set: Set of unique leaf keys (last segment of each path).
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
        batch_size (int, optional): Number of keys to process in one batch. Defaults to 8.
        model (str, optional): Model name. Defaults to "gemini-2.5-flash".

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
# Step 3: Linguistic transformations
# ----------------------------
def replace_last_key(doc, path, synonym_map):
    """
    Replace only the last key in a given path in the JSON document.

    Args:
        doc (dict or list): The JSON document.
        path (str): Dot-separated path string ('Logging.Sinks.*.Level').
        synonym_map (dict): Mapping of original keys to synonyms.
    """
    parts = path.split(".")
    if not parts:
        return

    cur = doc
    for i, part in enumerate(parts):
        if part == "*":
            if isinstance(cur, list):
                for item in cur:
                    replace_last_key(item, ".".join(parts[i+1:]), synonym_map)
            return
        if i == len(parts) - 1:
            if isinstance(cur, dict) and part in cur:
                new_key = synonym_map.get(part, part)
                if new_key != part:
                    cur[new_key] = cur.pop(part)
        else:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return


# ----------------------------
# Step 4: Structural transformations
# ----------------------------
def get_by_path(doc, path):
    """
    Get all values in a JSON document matching a given path.

    Args:
        doc (dict or list): The JSON document.
        path (str): The path to match.

    Returns:
            list of tuples: Each tuple contains (parent, key, value).
    """
    parts = path.split(".")
    results = []
    stack = [(doc, parts, None, None)]
    while stack:
        cur, rem, parent, key = stack.pop()
        if not rem:
            results.append((parent, key, cur))
            continue
        p = rem[0]
        rest = rem[1:]
        if isinstance(cur, dict) and p in cur:
            stack.append((cur[p], rest, cur, p))
        elif isinstance(cur, list) and p == "*":
            for i, item in enumerate(cur):
                stack.append((item, rest, cur, i))
    return results

def set_by_path(doc, path, func):
    """
    Apply a transformation function to all values in a JSON document matching a given path.

    Args:
        doc (dict or list): The JSON document.
        path (str): The path to match.
        func (callable): The transformation function to apply.
    """
    for parent, key, value in get_by_path(doc, path):
        parent[key] = func(value)

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
# Step 6: Assign transformations
# ----------------------------
def get_transformed_path(original_path, synonym_map):
    """
    Return transformed path using synonym_map for all keys.

    Args:
        original_path (str)
        synonym_map (dict)
    Returns:
        str: Transformed path
    """
    parts = original_path.split(".")
    transformed_parts = [p if p == "*" else synonym_map.get(p, p) for p in parts]
    return ".".join(transformed_parts)

def assign_transformations(paths, ratio_l=0.5, ratio_s1=0.25, ratio_s2=0.25, seed=None):
    """
    Assign transformations to paths (list version).

    Args:
        paths (list): List of unique paths.
        ratio_l (float): Ratio for linguistic changes.
        ratio_s1 (float): Ratio for structural increase nesting.
        ratio_s2 (float): Ratio for structural decrease nesting.
        seed (int): Random seed.

    Returns:
        list: Each entry is a dict {"path": str, "linguistic": bool, "structural": str|Null}
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

    assignments = []
    for path in paths:
        assignments.append({
            "path": path,
            "linguistic": path in l_paths,
            "structural": (
                "increase_nesting" if path in s1_paths else
                "reduce_nesting" if path in s2_paths else None
            )
        })
    
    # Sort paths by depth (top-level first)
    assignments = sorted(assignments, key=lambda r: len(r["path"].split(".")))

    return assignments


# ----------------------------
# Step 7: Apply transformations
# ----------------------------
def apply_transformations(docs, assignments, synonym_map, filename):
    """
    Apply transformations and produce ground truth, propagating parent changes to children.

    Args:
        docs (list): List of JSON documents.
        assignments (list): List of transformations per path, sorted by shortest path first.
        synonym_map (dict): Key -> synonym mapping.
        filename (str): File name.

    Returns:
        tuple: (transformed_docs, ground_truth_list)
    """
    transformed_docs = []
    ground_truth_set = set()
    ground_truth = []
    path_transform_map = {}

    for doc in docs:
        new_doc = deepcopy(doc)
        for rule in assignments:
            path = rule["path"]
            parts = path.split(".")

            # Build transformed path from parent transformations
            transformed_parts = []
            for i, p in enumerate(parts):
                parent_path = ".".join(parts[:i])
                if parent_path in path_transform_map:
                    transformed_parts = path_transform_map[parent_path].split(".")
                # Apply linguistic change only to last key
                if i == len(parts) - 1 and rule["linguistic"]:
                    transformed_parts.append(synonym_map.get(p, p))
                else:
                    transformed_parts.append(p)

            # Apply structural transformations
            if rule["structural"] == "increase_nesting":
                set_by_path(new_doc, path, increase_nesting)
                # Split last key if possible
                last_key_words = wordninja.split(transformed_parts[-1])
                if len(last_key_words) > 1:
                    transformed_parts[-1:] = last_key_words
                else:
                    rule["structural"] = "Can't increase"

            elif rule["structural"] == "reduce_nesting" and "." in path:
                set_by_path(new_doc, path, decrease_nesting)
                # Flatten all parts into one key
                flattened_key = "".join(
                    part[0].upper() + part[1:] if i > 0 else part
                    for i, part in enumerate(transformed_parts)
                )
                transformed_parts = [flattened_key]

            # Propagate to children
            transformed_path = ".".join(transformed_parts)
            path_transform_map[path] = transformed_path  

            # Record ground truth
            gt_key = (filename, path)
            if gt_key not in ground_truth_set:
                ground_truth_set.add(gt_key)
                original_key = path.split(".")[-1]
                transformed_key = transformed_path.split(".")[-1]
                ground_truth.append({
                    "filename": filename,
                    "original_path": path,
                    "transformed_path": transformed_path,
                    "linguistic": original_key != transformed_key,
                    "structural": rule["structural"]
                })

        transformed_docs.append(new_doc)

    return transformed_docs, ground_truth




# ----------------------------
# Step 8: Load/save JSON
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
# Step 9: Process one file
# ----------------------------
def process_one_file(file_path, output_dir, ling_ratio, struct1_ratio, struct2_ratio, batch_size=8, seed=42):
    """
    Process a single JSON file: load, transform, and save.

    Args:
        file_path (str): Path to the input JSON file.
        output_dir (str): Directory to save the transformed file.
        ling_ratio (float): Ratio for linguistic transformations.
        struct1_ratio (float): Ratio for structural increase nesting.
        struct2_ratio (float): Ratio for structural decrease nesting.
        batch_size (int, optional): Batch size for synonym generation. Defaults to 8.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    Returns:
        List: List of ground truth mappings)
    """
    output_file = output_dir / file_path.name
    if output_file.exists():
        print(f"Skipping {file_path.name} (already exists).")
        return []

    docs = list(load_json_lines(file_path))
    if not docs:
        print(f"No documents in {file_path.name}, skipping.")
        return []

    # 1. Collect paths and keys
    paths = collect_all_paths(docs)
    unique_keys = extract_leaf_key(paths)

    # 2. Build synonym map
    synonym_map = build_synonym_mapping(unique_keys, batch_size=batch_size)

    # 3. Assign transformations
    assignments = assign_transformations(paths, ratio_l=ling_ratio, ratio_s1=struct1_ratio, ratio_s2=struct2_ratio, seed=seed)

    # 4. Apply transformations
    transformed_docs, ground_truth = apply_transformations(docs, assignments, synonym_map, file_path.name)

    # 5. Save
    save_json_lines(output_file, transformed_docs)
    print(f"Processed {file_path.name}: {len(docs)} documents transformed.", flush=True)
    return ground_truth

# ----------------------------
# Step 10: Process all datasets
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
# Step 11: CLI
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
