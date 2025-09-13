import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

MODEL_NAME = "bert-base-uncased"

# ----------------------------
# Model loading
# ----------------------------
def load_model(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return pipeline("fill-mask", model=model, tokenizer=tokenizer)

fill_mask = None

# ----------------------------
# Modification functions
# ----------------------------

def flatten_object(obj, parent_key="", sep="."):
    """
    Flatten nested JSON objects into a single JSON with dot notation keys.
    Arrays stay intact, but objects inside arrays are also flattened.
    """
    out = {}
    stack = [(obj, parent_key)]

    while stack:
        current, current_key = stack.pop()
        if isinstance(current, dict):
            for k, v in current.items():
                new_key = f"{current_key}{sep}{k}" if current_key else k
                if isinstance(v, dict):
                    stack.append((v, new_key))
                elif isinstance(v, list):
                    new_list = []
                    for item in v:
                        if isinstance(item, dict):
                            new_list.append(flatten_object(item, parent_key=new_key, sep=sep))
                        else:
                            new_list.append(item)
                    out[new_key] = new_list
                else:
                    out[new_key] = v
        else:
            out[current_key] = current

    return out

@lru_cache(maxsize=None)
def get_synonym(word):
    """
    Get a synonym for the given word using CodeBERT masked LM.
    """
    global fill_mask
    fill_mask = load_model() if fill_mask is None else fill_mask
    text = f"The {word} is called [MASK]."
    preds = fill_mask(text, top_k=5)
    # Pick first prediction that is different from original
    for p in preds:
        candidate = p["token_str"].strip().replace("_", " ")
        if candidate.lower() != word.lower():
            return candidate
    return word

def rename_keys(doc, output_file):
    """
    Rename keys with synonyms and return both new doc and mapping.
    """
    renamed = {}
    mapping = []

    for key_path, v in doc.items():
        parts = key_path.split(".")
        key = parts[-1]
        synonym = get_synonym(key)
        new_key_path = ".".join(parts[:-1] + [synonym])
        renamed[new_key_path] = v

        mapping.append({
            "filename": str(output_file),
            "original_key_path": key_path,
            "new_key_path": new_key_path,
            "original_key": key,
            "synonym": synonym
        })
    return renamed, mapping

def flatten_documents(docs, output_file, groundtruth_file):
    """
    Flatten a list of JSON documents and write them line by line to JSONL.
    
    Args:
        docs (list): List of JSON documents.
        output_file (str): Path to output JSONL file.
    """
    all_mappings = []
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in docs:
            flat_doc = flatten_object(doc)
            renamed_doc, mapping = rename_keys(flat_doc, output_file)
            f.write(json.dumps(renamed_doc) + "\n")
            all_mappings.extend(mapping)

    # Save groundtruth mappings
    with open(groundtruth_file, "a", encoding="utf-8") as f:
        for map_entry in all_mappings:
            f.write(json.dumps(map_entry) + "\n")


# ----------------------------
# File processing functions
# ----------------------------

def split_docs(file_path):
    """Split JSON file into two halves of documents."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            docs = [json.loads(line) for line in f if line.strip()]
    except json.JSONDecodeError:
        print(f"[Error] Invalid JSON in {file_path}", flush=True)
        return [], []
    mid = len(docs) // 2
    return docs[:mid], docs[mid:]

def process_one_file(file_path, output_dir, groundtruth_file):
    output_path = output_dir / file_path.name

    # Split data into source and target documents
    source_docs, target_docs = split_docs(file_path)
    if not source_docs or not target_docs:
        return f"[Skipped] {file_path.name} — not enough documents to split"

    # Flatten the objects in the target documents
    flatten_documents(target_docs, output_path, groundtruth_file)

def process_datasets_parallel(input_root, output_root, groundtruth_file, sample_size=2, seed=101):
    input_root, output_root, groundtruth_file = Path(input_root), Path(output_root), Path(groundtruth_file)
    output_root.mkdir(parents=True, exist_ok=True)
    groundtruth_file.parent.mkdir(parents=True, exist_ok=True)

    all_files = list(Path(input_root).glob("*.json"))

    if not all_files:
        print("No input files found.")
        return

    # Pick a random sample of files to process
    random.seed(seed)
    selected = random.sample(all_files, min(sample_size, len(all_files)))
    print(f"Processing {len(selected)} files...")

    args_list = [
        (file, output_root, groundtruth_file)
        for file in selected
    ]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_one_file, *args) for args in args_list]
        for future in as_completed(futures):
            result = future.result()
            if result:
                print(result)

# ----------------------------
# CLI entrypoint
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Transform JSON with synonym-based key renaming.")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("groundtruth_file", help="File to store groundtruth mappings")
    parser.add_argument("--sample_size", type=int, default=2, help="Number of files to process")
    parser.add_argument("--seed", type=int, default=101, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    process_datasets_parallel(
        input_root=args.input_dir,
        output_root=args.output_dir,
        groundtruth_file=args.groundtruth_file,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    print(f"Finished in {time.time() - start:.2f} seconds.")


if __name__ == "__main__":
    main()
