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
        dict: Mapping of path identifiers to paths.
    """
    all_paths = set()
    for doc in docs:
        all_paths.update(extract_paths(doc))
    return {f"path_{i}": p for i, p in enumerate(sorted(all_paths), 1)}

def extract_unique_keys_from_paths(paths):
    """
    Extract unique keys from a dictionary of paths.
    
    Args:
        paths (dict): Mapping of path identifiers to paths.
    Returns:
        set: Set of unique keys.
    """ 
    unique_keys = set()
    for path in paths.values():
        parts = path.split(".")
        unique_keys.update(parts)
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
            "\n".join(f"{j + 1}. Key: '{key}' → Synonym:" for j, key in enumerate(batch_keys))
        )

        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                break
            except Exception as e:
                if "429" in str(e):
                    time.sleep(2 ** attempt)
                else:
                    raise

        for key, line in zip(batch_keys, (response.text.split("\n") if response and hasattr(response, "text") else batch_keys)):
            synonym = "".join(c for c in line.strip() if c.isalnum() or c == "_") or key
            synonym_map[key] = synonym

    return synonym_map


# ----------------------------
# Step 3: Linguistic transformations
# ----------------------------
def replace_keys(doc, synonym_map, parent_key_map=None):
    """
    Replace keys in a JSON document based on a synonym mapping.

    Args:
        doc (dict or list): The JSON document.
        synonym_map (dict): Mapping of original keys to synonyms.
        parent_key_map (dict, optional): Mapping of parent keys to their synonyms. Defaults to None.
    """
    if parent_key_map is None:
        parent_key_map = {}

    if isinstance(doc, dict):
        keys_to_process = list(doc.keys())
        for k in keys_to_process:
            v = doc[k]

            # Determine the new key, propagating parent mapping if needed
            mapped_key = synonym_map.get(k, k)
            if k in parent_key_map:
                mapped_key = parent_key_map[k]

            # Update dict if key changed
            if mapped_key != k:
                doc[mapped_key] = doc.pop(k)

            # Update parent_key_map for children
            child_parent_map = parent_key_map.copy()
            child_parent_map[k] = mapped_key

            # Recurse if value is dict or list
            if isinstance(v, (dict, list)):
                replace_keys(doc[mapped_key], synonym_map, child_parent_map)

    elif isinstance(doc, list):
        for item in doc:
            if isinstance(item, (dict, list)):
                replace_keys(item, synonym_map, parent_key_map)


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
                result.update(decrease_nesting(v, new_key))
            else:
                result[new_key] = v
    elif isinstance(doc, list):
        result = [decrease_nesting(item, parent_key) if isinstance(item, dict) else item for item in doc]
    else:
        result = doc
    return result


# ----------------------------
# Step 6: Assign transformations
# ----------------------------
def build_parent_child_map(paths):
    """
    Build a mapping of parent path -> set of child paths.

    Args:
        paths (dict): {name: path_string}
    Returns:
        dict: parent_path -> set of child paths
    """
    parent_map = {}
    for p in paths.values():
        parts = p.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            child = ".".join(parts[:i+1])
            parent_map.setdefault(parent, set()).add(child)
    return parent_map

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

def assign_transformations(paths, synonym_map, ratio_l=0.5, ratio_s1=0.25, ratio_s2=0.25, seed=None):
    """
    Assign transformations to paths and propagate to descendants.

    Args:
        paths (dict): {name: path_string}
        synonym_map (dict): {key: synonym}
        ratio_l, ratio_s1, ratio_s2 (float): Ratios
        seed (int, optional): Random seed
    Returns:
        dict: {path_id: {"path": str, "linguistic": bool, "structural": str|None}}
    """
    if seed is not None:
        random.seed(seed)

    n = len(paths)
    all_keys = list(paths.keys())
    random.shuffle(all_keys)

    s1_count = int(n * ratio_s1)
    s2_count = int(n * ratio_s2)
    l_count = int(n * ratio_l)

    s1_keys = set(all_keys[:s1_count])
    s2_keys = set(all_keys[s1_count:s1_count+s2_count])
    l_keys = set(random.sample(all_keys, l_count))

    # Build parent-child map
    parent_map = build_parent_child_map(paths)

    # Propagate linguistic changes
    all_linguistic_paths = set()
    for p in l_keys:
        # Add p and all descendants
        stack = [paths[p]]
        while stack:
            cur = stack.pop()
            all_linguistic_paths.add(cur)
            for child in parent_map.get(cur, []):
                if child not in all_linguistic_paths:
                    stack.append(child)

    assignments = {}
    for name, path in paths.items():
        assignments[name] = {
            "path": path,
            "linguistic": path in all_linguistic_paths and any(synonym_map.get(k, k) != k for k in path.split(".") if k != "*"),
            "structural": (
                "increase_nesting" if name in s1_keys else
                "reduce_nesting" if name in s2_keys else None
            )
        }
    return assignments


# ----------------------------
# Step 7: Apply transformations
# ----------------------------
def apply_transformations(docs, assignments, synonym_map, filename):
    """
    Apply assigned transformations to a list of JSON documents and produce ground truth.

    Args:
        docs (list): List of JSON documents.
        assignments (dict): {path_id: {"path": str, "linguistic": bool, "structural": str|None}}
        synonym_map (dict): {key: synonym}
        filename (str): Name of the file being processed.
    Returns:
        tuple: (transformed_docs, list of ground truth entries)
    """
    transformed_docs = []
    ground_truth_dict = {}  # Track unique paths per file

    for doc in docs:
        new_doc = deepcopy(doc)
        for rule in assignments.values():
            path = rule["path"]
            structural = rule["structural"]

            # Apply linguistic key renaming if assigned
            if rule["linguistic"]:
                replace_keys(new_doc, synonym_map)

            # Apply structural transformations
            if structural == "increase_nesting":
                set_by_path(new_doc, path, increase_nesting)
            elif structural == "reduce_nesting":
                if "." in path:
                    set_by_path(new_doc, path, decrease_nesting)
                else:
                    # Skip top-level reduce_nesting
                    structural = "Can't reduce top-level"

            # Compute transformed path
            parts = path.split(".")
            transformed_parts = [
                synonym_map.get(p, p) if p != "*" else p
                for p in parts
            ]

            if rule["structural"] == "increase_nesting":
                # Apply WordNinja to the last key to reflect increased nesting
                last_key = transformed_parts[-1]
                words = wordninja.split(synonym_map.get(last_key, last_key))
                if len(words) > 1:
                    set_by_path(new_doc, path, increase_nesting)
                    transformed_path = ".".join(words)
                else:
                    # Nesting not possible
                    structural = "Can't increase"
                    transformed_path = get_transformed_path(path, synonym_map)

            elif rule["structural"] == "reduce_nesting" and "." in path:
                # Flatten nested keys
                flattened_key = "".join(
                    part[0].upper() + part[1:] if i > 0 else part
                    for i, part in enumerate(transformed_parts)
                )
                transformed_parts = [flattened_key]

            transformed_path = ".".join(transformed_parts)

            # Record ground truth once per original path
            if path not in ground_truth_dict:
                ground_truth_dict[path] = {
                    "filename": filename,
                    "original_path": path,
                    "transformed_path": transformed_path,
                    "linguistic": transformed_path != path, #rule["linguistic"],
                    "structural": structural
                }

        transformed_docs.append(new_doc)

    return transformed_docs, list(ground_truth_dict.values())


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
    
    with open(output_file, "w") as f:
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
    unique_keys = extract_unique_keys_from_paths(paths)

    # 2. Build synonym map
    synonym_map = build_synonym_mapping(unique_keys, batch_size=batch_size)

    # 3. Assign transformations
    assignments = assign_transformations(paths, synonym_map, ratio_l=ling_ratio, ratio_s1=struct1_ratio, ratio_s2=struct2_ratio, seed=seed)

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
    parser.add_argument("--sample_size", type=int, default=25)
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
