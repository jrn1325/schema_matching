import argparse
import google.genai as genai
import hashlib
import json
import random
import re
import time

from pathlib import Path
from tqdm import tqdm

# ----------------------------
# Synonym generation
# ----------------------------
def build_synonym_mapping(unique_keys, batch_size=64, model="gemini-2.5-flash"):
    """
    Build a mapping from keys to single-word JSON-friendly synonyms using Google Gemini.
    Processes multiple keys per API call for efficiency.

    Args:
        unique_keys (set): Set of unique JSON keys to generate synonyms for.
        batch_size (int): Number of keys to process in each API call.
        model (str): The Gemini model to use.
    Returns:
        dict: Mapping from original keys to their synonyms.
    """
    # Initialize the Gemini client once
    client = genai.Client() 

    synonym_map = {}
    keys = list(unique_keys)

    for i in tqdm(range(0, len(keys), batch_size), desc="Building synonyms"):
        batch_keys = keys[i:i+batch_size]

        prompt_lines = [
            f"{i + j + 1}. Key: '{key}' → Synonym:" for j, key in enumerate(batch_keys)
        ]
        prompt = (
            "Provide a synonym for each JSON key below."
            "Return **only one word per line**, no extra text, in the same order:\n" +
            "\n".join(prompt_lines)
        )

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            # Process response
            for key, line in zip(batch_keys, response.text.split("\n")):
                clean_line = line.strip()
                if not clean_line:
                    synonym = key 
                else:
                    synonym = "".join(c for c in clean_line if c.isalnum() or c == "_")
                synonym_map[key] = synonym

        except Exception as e:
            print(f"Error generating batch synonyms (keys {batch_keys}): {e}")
            # Fallback: use original keys
            for key in batch_keys:
                synonym_map[key] = key

    return synonym_map


# ----------------------------
# JSON key splitting and renaming
# ----------------------------
def split_compound_key(key):
    """
    Split a compound key into its constituent parts.

    Args:
        key (str): The compound key to split.

    Returns:
        list: List of key parts.
    """
    match = re.search(r'[a-z](?=[A-Z])', key)
    if match:
        idx = match.start() + 1
        return [key[:idx], key[idx:]]
    else:
        return [key]

def split_top_level_keys(obj, parent_path=None):
    """
    Split top-level compound keys into nested dicts/lists.
    List indices replaced with '*'.
    
    Returns:
        transformed_obj, mapping (dict of full old_path → full new_path)
    """
    if parent_path is None:
        parent_path = []

    mapping = {}

    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            old_path = ".".join(parent_path + [k])
            parts = split_compound_key(k)
            if len(parts) > 1:
                d = new_obj
                conflict = False
                for p in parts[:-1]:
                    if p in d and not isinstance(d[p], dict):
                        conflict = True
                        break
                    d = d.setdefault(p, {})
                if not conflict:
                    d[parts[-1]] = v
                    new_path = ".".join(parent_path + parts)
                    mapping[old_path] = new_path
                    continue
            new_obj[k] = v
            mapping[old_path] = old_path
        return new_obj, mapping

    elif isinstance(obj, list):
        new_list = []
        for item in obj:
            transformed, submap = split_top_level_keys(item, parent_path + ["*"])
            new_list.append(transformed)
            mapping.update(submap)
        return new_list, mapping

    else:
        return obj, {}

def rename_keys_recursive(obj, synonym_map, parent_path=None):
    """
    Recursively rename keys in a JSON object using synonym_map.
    Keeps track of full-path mappings. List indices replaced with '*'.
    
    Returns:
        transformed_obj, mapping (dict of full old_path → full new_path)
    """
    if parent_path is None:
        parent_path = []

    mapping = {}

    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_k = synonym_map.get(k, k)
            old_path = ".".join(parent_path + [k])
            new_path = ".".join(parent_path + [new_k])

            transformed, submap = rename_keys_recursive(v, synonym_map, parent_path + [new_k])
            new_obj[new_k] = transformed
            mapping[old_path] = new_path
            mapping.update(submap)
        return new_obj, mapping

    elif isinstance(obj, list):
        new_list = []
        for item in obj:
            transformed, submap = rename_keys_recursive(item, synonym_map, parent_path + ["*"])
            new_list.append(transformed)
            mapping.update(submap)
        return new_list, mapping

    else:
        return obj, {}

def transform_document(doc, synonym_map, mode):
    """
    Apply either linguistic OR structural transformation to a JSON doc.

    Args:
        doc (dict): Input JSON document
        synonym_map (dict): key → synonym mapping
        mode (str): "linguistic" or "structural"
    Returns:
        transformed_doc, final_map
    """
    if mode == "linguistic":
        transformed_doc, final_map = rename_keys_recursive(doc, synonym_map)
    elif mode == "structural":
        transformed_doc, final_map = split_top_level_keys(doc)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return transformed_doc, final_map

# ----------------------------
# Groundtruth mapping
# ----------------------------
def save_final_map(final_map, filename, output_path, mode):
    """
    Save final mapping to a JSON file (one per line) with filename and mode.

    Args:
        final_map (dict): Mapping old_path → new_path
        filename (str): Name of input dataset/file
        output_path (str or Path): JSONL groundtruth file
        mode (str): Transformation type ("linguistic" or "structural")
    """
    data = {
        "filename": filename,
        "mode": mode,
        "mappings": final_map
    }
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


# ----------------------------
# Utilities
# ----------------------------
def stream_json_file(file_path):
    """
    Stream JSON objects from a file, one per line.

    Args:
        file_path (str or Path): Path to the JSON file.
    Yields:
        dict: JSON object.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def collect_keys_from_file(file_path):
    """
    Collect all unique keys from a JSON file.

    Args:
        file_path (str or Path): Path to the JSON file.
    Returns:
        set: Set of unique keys.
    """
    unique_keys = set()
    for doc in stream_json_file(file_path):
        stack = [doc]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                for k, v in current.items():
                    unique_keys.add(k)
                    stack.append(v)
            elif isinstance(current, list):
                stack.extend(current)
    return unique_keys


# ----------------------------
# Dataset processing
# ----------------------------
def balanced_assign_modes(files):
    """
    Assign modes ("linguistic" / "structural") to files with a ~50/50 split.

    For odd number of files, the extra file goes to "linguistic".

    Args:
        files (list): List of Path objects.
    Returns:
        dict: Mapping from Path → mode
    """
    # Sort files by stable hash for deterministic behavior
    sorted_files = sorted(files, key=lambda f: int(hashlib.md5(f.name.encode("utf-8")).hexdigest(), 16))
    
    half = len(sorted_files) // 2
    assignments = {}
    for i, f in enumerate(sorted_files):
        if i < half + len(sorted_files) % 2:  # extra goes to linguistic
            mode = "linguistic"
        else:
            mode = "structural"
        assignments[f] = mode
    return assignments

def process_one_file(file_path, output_dir, groundtruth_file, synonym_map, mode):
    """
    Transform a single JSON file and save output + groundtruth mapping.
    """
    output_path = output_dir / file_path.name

    all_final_map = {}
    with open(output_path, "w", encoding="utf-8") as f_out:
        for doc in stream_json_file(file_path):
            transformed_doc, final_map = transform_document(doc, synonym_map, mode=mode)
            f_out.write(json.dumps(transformed_doc) + "\n")
            all_final_map.update(final_map)

    # Save groundtruth with filename and mode
    save_final_map(all_final_map, filename=file_path.name, output_path=groundtruth_file, mode=mode)

def process_datasets(input_root, output_root, groundtruth_file, sample_size=1, seed=42, batch_size=8):
    input_root, output_root, groundtruth_file = Path(input_root), Path(output_root), Path(groundtruth_file)
    output_root.mkdir(parents=True, exist_ok=True)
    groundtruth_file.parent.mkdir(parents=True, exist_ok=True)

    all_files = list(input_root.glob("*.json"))
    if not all_files:
        print("No input files found.")
        return

    random.seed(seed)
    selected_files = random.sample(all_files, min(sample_size, len(all_files)))
    print(f"Processing {len(selected_files)} files...", flush=True)

    # Collect all unique keys for synonym generation
    all_keys = set()
    for file_path in selected_files:
        all_keys.update(collect_keys_from_file(file_path))

    synonym_map = build_synonym_mapping(all_keys, batch_size=batch_size)

    # Assign balanced modes
    assignments = balanced_assign_modes(selected_files)

    for file_path, mode in assignments.items():
        process_one_file(file_path, output_root, groundtruth_file, synonym_map, mode=mode)


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Transform JSON with synonym-based key renaming and structural split.")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("groundtruth_file", help="File to store groundtruth mappings")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of files to process")
    parser.add_argument("--seed", type=int, default=101, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=64, help="GPU batch size for synonym generation")
    return parser.parse_args()

def main():
    args = parse_args()
    start = time.time()
    process_datasets(
        input_root=args.input_dir,
        output_root=args.output_dir,
        groundtruth_file=args.groundtruth_file,
        sample_size=args.sample_size,
        seed=args.seed,
        batch_size=args.batch_size
    )
    print(f"Finished in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    main()
