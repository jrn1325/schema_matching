import argparse
import google.genai as genai
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

        # Global numbering for clarity in the prompt
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

            # Process response safely
            for key, line in zip(batch_keys, response.text.split("\n")):
                clean_line = line.strip()
                if not clean_line:
                    synonym = key  # fallback if line is empty
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
    """Split a compound key into its constituent parts.

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

def split_top_level_keys(obj):
    """Split top-level keys into separate parts.

    Args:
        obj (dict): The JSON object to process.

    Returns:
        dict: The transformed JSON object with split keys.
    """
    new_obj = {}
    for k, v in obj.items():
        if not isinstance(v, (dict, list)):
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
                    continue
        new_obj[k] = v
    return new_obj

def rename_keys_recursive(obj, synonym_map):
    """Recursively rename keys in a JSON object using the synonym map.

    Args:
        obj (dict or list): The JSON object or list to process.
        synonym_map (dict): Mapping from original keys to synonyms.
    Returns:
        dict or list: The transformed JSON object or list with renamed keys.
    """
    if isinstance(obj, dict):
        return {synonym_map.get(k, k): rename_keys_recursive(v, synonym_map) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [rename_keys_recursive(i, synonym_map) for i in obj]
    else:
        return obj

def transform_document(doc, synonym_map):
    """Transform a JSON document by renaming keys and splitting top-level keys.
    Args:
        doc (dict): The JSON document to transform.
        synonym_map (dict): Mapping from original keys to synonyms.
    Returns:
        dict: The transformed JSON document.
    """
    renamed = rename_keys_recursive(doc, synonym_map)
    return split_top_level_keys(renamed)

# ----------------------------
# Groundtruth mapping
# ----------------------------
def build_mapping(original_doc, transformed_doc, output_file_name):
    """Build a mapping from original keys to new keys in the transformed document.

    Args:
        original_doc (dict): The original JSON document.
        transformed_doc (dict): The transformed JSON document.
        output_file_name (str): The name of the output file.
    Returns:
        list: List of mapping entries.
    """
    mappings = []
    seen = set()

    for orig_key, orig_val in original_doc.items():
        if not isinstance(orig_val, (dict, list)):
            final_val = orig_val
            path_stack = [(transformed_doc, [])]
            found_path = None

            while path_stack:
                current_obj, current_path = path_stack.pop()
                if isinstance(current_obj, dict):
                    for k, v in current_obj.items():
                        if v is final_val:
                            found_path = current_path + [k]
                            break
                        elif isinstance(v, dict):
                            path_stack.append((v, current_path + [k]))
                    if found_path:
                        break

            new_key_path = ".".join(found_path) if found_path else orig_key
            mapping = {
                "filename": output_file_name,
                "original_key_path": orig_key,
                "new_key_path": new_key_path,
                "original_key": orig_key
            }
            if tuple(mapping.items()) not in seen:
                seen.add(tuple(mapping.items()))
                mappings.append(mapping)

    return mappings

# ----------------------------
# Utilities
# ----------------------------
def stream_json_file(file_path):
    """Stream JSON objects from a file, one per line.
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
# File processing
# ----------------------------
def process_one_file(file_path, output_dir, groundtruth_file, synonym_map):
    output_path = output_dir / file_path.name
    seen_mappings = set()
    all_mappings = []

    with open(output_path, "w", encoding="utf-8") as f_out, \
         open(groundtruth_file, "a", encoding="utf-8") as f_gt:

        for doc in tqdm(stream_json_file(file_path), desc=f"Processing {file_path.name}"):
            transformed_doc = transform_document(doc, synonym_map)
            mapping = build_mapping(doc, transformed_doc, file_path.stem)

            for m in mapping:
                m_key = tuple(sorted(m.items()))
                if m_key not in seen_mappings:
                    seen_mappings.add(m_key)
                    all_mappings.append(m)

            f_out.write(json.dumps(transformed_doc) + "\n")

        for map_entry in all_mappings:
            f_gt.write(json.dumps(map_entry) + "\n")

# ----------------------------
# Dataset processing
# ----------------------------
def process_datasets(input_root, output_root, groundtruth_file, sample_size=1, seed=42, batch_size=8):
    input_root, output_root, groundtruth_file = Path(input_root), Path(output_root), Path(groundtruth_file)
    output_root.mkdir(parents=True, exist_ok=True)
    groundtruth_file.parent.mkdir(parents=True, exist_ok=True)

    all_files = list(Path(input_root).glob("*.json"))
    if not all_files:
        print("No input files found.")
        return

    random.seed(seed)
    selected_files = random.sample(all_files, min(sample_size, len(all_files)))
    print(f"Processing {len(selected_files)} files...", flush=True)

    # Step 1: Collect all unique keys from selected files
    all_keys = set()
    for file_path in tqdm(selected_files, desc="Collecting unique keys"):
        all_keys.update(collect_keys_from_file(file_path))

    # Step 2: Build synonym map safely on GPU
    synonym_map = build_synonym_mapping(all_keys, batch_size=batch_size)

    # Step 3: Process each file line by line
    for file_path in tqdm(selected_files, desc="Transforming files"):
        process_one_file(file_path, output_root, groundtruth_file, synonym_map)

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Transform JSON with synonym-based key renaming and structural split.")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("groundtruth_file", help="File to store groundtruth mappings")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of files to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
