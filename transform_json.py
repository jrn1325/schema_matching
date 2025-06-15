
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

from gensim.models import KeyedVectors
import gensim.downloader as api
import json
import inflect
import numpy as np
import os
import random
import sys


inflector = inflect.engine()

# lingustic change functions
def load_fasttext_model(local_path="fasttext.model"):
    """
    Load FastText word embedding model.
    - If the model file exists locally, load it from disk.
    - Otherwise, download it from Gensim and save for future use.

    Args:
        local_path (str): Path to saved model file.

    Returns:
        Gensim KeyedVectors model.
    """
    if os.path.exists(local_path):
        print(f"Loading FastText model from: {local_path}")
        return KeyedVectors.load(local_path)
    
    print("Downloading FastText model from Gensim API...")
    model = api.load("fasttext-wiki-news-subwords-300")
    model.save(local_path)
    print(f"Model saved to: {local_path}")
    return model


def build_vocab_index(model, max_words=100000):
    """
    Build a vocabulary list and KNN index from the embedding model.

    Args:
        model: The pretrained FastText model.
        max_words: Maximum number of words to index for performance.

    Returns:
        vocab: List of vocabulary words.
        vectors: Corresponding word vectors.
        index: NearestNeighbors index for similarity search.
    """
    vocab = model.index_to_key[:max_words]
    vectors = np.array([model[word] for word in vocab])
    index = NearestNeighbors(n_neighbors=6, metric="cosine").fit(vectors)
    return vocab, vectors, index


def is_plural_form(base, candidate):
    """
    Check if candidate is a plural or singular form of base.
    Args:
        base: The base word.
        candidate: The candidate word to check.
    Returns:
        True if candidate is a plural/singular form of base, else False.
    """
    return inflector.singular_noun(candidate) == base or inflector.plural_noun(base) == candidate


def is_too_similar(a, b, threshold=0.85):
    """
    Check if two words are too morphologically similar (e.g., 'create' vs 'created').
    Args:
        a: First word.
        b: Second word.
        threshold: Similarity ratio above which words are considered too similar.
    Returns:
        True if words are too similar, else False.
    """
    return SequenceMatcher(None, a, b).ratio() > threshold


def find_synonym(word, model, vocab, index, vectors):
    """
    Find a near-synonym based on vector similarity,
    excluding trivial variants (e.g., plural forms or near-identical matches).

    Args:
        word: The original word to find a synonym for.
        model: The pretrained FastText model.
        vocab: List of vocabulary words used in the KNN index.
        index: NearestNeighbors index built from FastText vectors.
        vectors: Corresponding FastText vectors.

    Returns:
        A synonym word, or the original word if no suitable synonym is found.
    """
    word = word.lower()
    if word not in model:
        return word

    word_vec = model[word].reshape(1, -1)

    # Find nearest neighbors
    _, indices = index.kneighbors(word_vec)

    for idx in indices[0]:
        candidate = vocab[idx]
        if (candidate.lower() != word and
            not is_plural_form(word, candidate) and
            not is_too_similar(word, candidate)):
            return candidate

    return word


def rename_key(key, model, vocab, index, vectors, change_rate):
    """
    Rename a key using synonym-based renaming based on change rate.

    Args:
        key: The original key.
        model, vocab, index, vectors: FastText-related objects.
        change_rate: Probability of applying transformation (0 to 1).

    Returns:
        A renamed (possibly unchanged) camelCase version of the key.
    """
    if random.random() > change_rate:
        return key
    parts = key.split('_')
    new_parts = [find_synonym(part, model, vocab, index, vectors) for part in parts]
    return ''.join([new_parts[0]] + [p.capitalize() for p in new_parts[1:]])



# structural change functions
def group_keys_by_prefix(flat_dict, change_rate):
    """
    Group flat keys into nested structures based on shared prefix.

    Args:
        flat_dict: A dictionary of flat keys.
        change_rate: Probability of grouping keys into sub-objects.

    Returns:
        grouped: Dict of nested keys by prefix.
        others: Dict of keys not grouped.
    """
    grouped = defaultdict(dict)
    others = {}
    for k, v in flat_dict.items():
        parts = k.split('_', 1)
        if len(parts) == 2 and random.random() < change_rate:
            prefix, subkey = parts
            grouped[prefix][subkey] = v
        else:
            others[k] = v
    return grouped, others


def nest_flat_structure(data, change_rate):
    """
    Recursively nest flat key structures using prefix grouping.

    Args:
        data: A (partially) flat dictionary.
        change_rate: How aggressively to group keys into nested dicts.

    Returns:
        A new dictionary with optionally nested substructures.
    """
    if not isinstance(data, dict):
        return data
    grouped, others = group_keys_by_prefix(data, change_rate)
    result = {}
    for prefix, subdict in grouped.items():
        nested = nest_flat_structure(subdict, change_rate)
        result[prefix] = nested
    for k, v in others.items():
        result[k] = nest_flat_structure(v, change_rate) if isinstance(v, dict) else v
    return result


def flatten_nested_structure(nested_dict, parent_key="", sep="_", change_rate=1.0):
    """
    Recursively flatten nested dictionaries into a single-level dict with concatenated keys.
    Args:
        nested_dict: The nested dictionary to flatten.
        parent_key: The base key string for recursion.
        sep: Separator between concatenated keys.
        change_rate: Probability of flattening a nested dict.
    Returns:
        A flat dictionary with concatenated keys.
    """
    if not isinstance(nested_dict, dict):
        return nested_dict  
    
    items = {}
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) and random.random() < change_rate:
            subitems = flatten_nested_structure(v, new_key, sep=sep, change_rate=change_rate)
            for sub_k, sub_v in subitems.items():
                if sub_k in items:
                    raise ValueError(f"Key collision when flattening: {sub_k}")
                items[sub_k] = sub_v
        else:
            if new_key in items:
                raise ValueError(f"Key collision when flattening: {new_key}")
            items[new_key] = v
    return items


def is_flat_dict(d):
    """
    Returns True if all values are scalar or non-dict types (not just at top level).

    Args:
        d: The dictionary to check.
    Returns:
        True if all values are non-dict types, else False.
    """
    if not isinstance(d, dict):
        return False
    return all(not isinstance(v, dict) for v in d.values())


def nest_or_flatten(data, change_rate=1.0):
    """
    If the data is flat, nest it. If the data is nested, flatten it.
    Args:
        data: Input JSON-like object (dict or list).
        change_rate: Float [0, 1] for how aggressively to transform.
    Returns:    
        Transformed JSON object.
    """
    if isinstance(data, dict) and all(isinstance(k, str) and "_" in k for k in data.keys()):
        return nest_flat_structure(data, change_rate)
    elif isinstance(data, dict) and any(isinstance(v, dict) for v in data.values()):
        return flatten_nested_structure(data, change_rate=change_rate)
    else:
        return data
    

def restructure_doc(data, model, vocab, index, vectors, change_rate):
    """
    Apply dynamic key renaming and optional structural nesting.

    Args:
        data: Input JSON-like object (dict or list).
        model, vocab, index, vectors: Embedding components.
        change_rate: Float [0, 1] for how aggressively to transform.

    Returns:
        Transformed JSON object.
    """
    if isinstance(data, dict):
        renamed = {}
        for k, v in data.items():
            new_k = rename_key(k, model, vocab, index, vectors, change_rate)
            new_v = restructure_doc(v, model, vocab, index, vectors, change_rate)
            renamed[new_k] = new_v
        return nest_or_flatten(renamed, change_rate)

    elif isinstance(data, list):
        return [restructure_doc(item, model, vocab, index, vectors, change_rate) for item in data]
    else:
        return data


# file processing functions
def read_lines(file):
    """
    Read lines from a file, handling exceptions.
    Args:
        file: Path to the file.
    Returns:
        List of lines, or empty list on error.
    """ 
    try:
        with open(file, 'r') as f:
            return f.readlines()
    except Exception as e:
        print(f"[Error] Failed to read {file}: {e}")
        return []


def process_one_file(file_path, output_dir, model_path, change_rate, seed):
    """
    Process a single json file, transforming each line.
    Args:
        file_path (Path): Path to the input NDJSON file.
        output_dir (Path): Directory to write output.
        model_path (str): Path to FastText model file.
        change_rate (float): [0, 1] transformation intensity.
        seed (int): Random seed for reproducibility.
    Returns:
        Status message string.
    """
   

    random.seed(seed + hash(file_path) % 10000)

    model = load_fasttext_model(model_path)
    vocab, vectors, index = build_vocab_index(model)

    output_path = output_dir / file_path.name
    count = 0

    try:
        with open(file_path, 'r') as fin, open(output_path, 'w') as fout:
            for i, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    transformed = restructure_doc(doc, model, vocab, index, vectors, change_rate)
                    fout.write(json.dumps(transformed) + '\n')
                    count += 1
                except Exception as e:
                    continue
    except Exception as e:
        return f"[Error] Failed {file_path.name}: {e}"

    return f"{file_path.name} — {count} lines transformed"


def process_datasets_parallel(input_root, output_root, change_rate, sample_size=10, seed=42, model_path=None):
    """
    Process multiple JSON files in parallel, transforming each.
    Args:
        input_root (str or Path): Directory with input JSON files.
        output_root (str or Path): Directory to write transformed files.
        change_rate (float): [0, 1] transformation intensity.
        sample_size (int): Number of files to randomly sample and process.
        seed (int): Random seed for reproducibility.
        model_path (str): Path to FastText model file.
    """

    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_files = list(input_root.glob("*.json"))
    if not all_files:
        print("No JSON files found in input directory.")
        return

    random.seed(seed)
    selected = random.sample(all_files, min(sample_size, len(all_files)))

    print(f"[INFO] Selected {len(selected)} of {len(all_files)} files using seed {seed}.")

    args_list = [
        (file, output_root, model_path, change_rate, seed)
        for file in selected
    ]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_one_file, *args) for args in args_list]
        for future in as_completed(futures):
            print(future.result())



def main():
    
    if len(sys.argv) < 3:
        print("Usage: python transform_json.py <input_dir> <output_dir> [change_rate]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    change_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    model_path = "fasttext.model"

    process_datasets_parallel(
        input_root=input_dir,
        output_root=output_dir,
        change_rate=change_rate,
        sample_size=10,
        seed=42,
        model_path=model_path,
    )



if __name__ == "__main__":
    main()


