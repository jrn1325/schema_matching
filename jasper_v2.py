import argparse
import ast
import base64
import json
import math
import networkx as nx
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
import wandb

from pathlib import Path
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
NUM_EPOCHS = 25
LEARNING_RATE = 2e-5
HIDDEN_DIM = 256
OUT_DIM = 128
CODEBERT_DIM = 768
device = "cuda" if torch.cuda.is_available() else "cpu"

# Type mapping for multi-hot type vectors
TYPE_MAP = {
    "string": 0,
    "integer": 1,
    "object": 2,
    "array": 3,
    "boolean": 4,
    "null": 5,
}
NUM_TYPES = len(TYPE_MAP)
STRUCT_DIM = 5 + NUM_TYPES  # 5 numeric features + type multi-hot

# -----------------------------
# DATA UTILITIES
# -----------------------------
def load_dataset(path):
    """Load a CSV dataset containing paths, embeddings, and structural info."""
    return pd.read_csv(path, delimiter=';')

def build_graph(paths):
    """
    Build a graph from tuple paths.

    Args:
        paths: List of tuples, where each tuple is a path like ("rules", "braces")
    Returns:
        A NetworkX graph where nodes are paths and edges connect parent-child paths.
    """
    G = nx.Graph()

    for path in paths:
        G.add_node(path)

        if len(path) > 1:
            parent = path[:-1]
            G.add_edge(parent, path)

    return G

def decode_embedding(b64_string, dim=CODEBERT_DIM):
    """Decode base64 string into a numpy array of shape (dim,)"""
    if not b64_string:
        return np.zeros(dim, dtype=np.float32)
    byte_data = base64.b64decode(b64_string)
    arr = np.frombuffer(byte_data, dtype=np.float32)
    return arr.reshape(dim)

def encode_types(type_list):
    """Convert list of types into a multi-hot torch vector"""
    vec = torch.zeros(NUM_TYPES)
    if type_list is None:
        return vec
    for t in type_list:
        if t in TYPE_MAP:
            vec[TYPE_MAP[t]] = 1.0
    return vec

def combine_embeddings(df, graph, dim=CODEBERT_DIM):
    """
    Combine path and value embeddings with structural features for each node.

    Returns:
        emb: (num_nodes, CODEBERT_DIM*2) tensor
        struct_feat: (num_nodes, STRUCT_DIM) tensor
    """
    df_lookup = {row.path: row for row in df.itertuples(index=False)}
    embeddings = []
    struct_features = []

    for node in graph.nodes():
        row = df_lookup.get(node)

        if row is None:
            path_emb = torch.zeros(dim)
            value_emb = torch.zeros(dim)
            struct_feat = torch.zeros(STRUCT_DIM)
        else:
            path_emb = torch.tensor(decode_embedding(row.path_emb), dtype=torch.float32)
            value_emb = torch.tensor(decode_embedding(row.values_emb), dtype=torch.float32)

            # ---- STRUCTURAL FEATURES ----
            num_children = math.log1p(row.num_children)
            num_siblings = math.log1p(row.num_siblings)
            depth = math.log1p(row.nesting_depth)
            freq = row.norm_freq
            entropy = row.key_entropy

            type_vec = encode_types(row.types)

            struct_feat = torch.tensor([num_children, num_siblings, depth, freq, entropy], dtype=torch.float32)
            struct_feat = torch.cat([struct_feat, type_vec], dim=0)

        embeddings.append(torch.cat([path_emb, value_emb], dim=0))
        struct_features.append(struct_feat)

    emb = torch.stack(embeddings).to(device)
    struct_feat = torch.stack(struct_features).to(device)
    return emb, struct_feat

def get_ground_truth_pairs(ground_truth_path, filename):
    """
    Load all ground truth node pairs for a specific filename.

    Args:
        ground_truth_path: Path to the JSONL file containing ground truth mappings.
        filename: The specific filename to filter mappings for.
    Returns:
        A set of (src_tuple, tgt_tuple) pairs.
    """
    gt = set()

    with open(ground_truth_path, "r") as f:
        for line in f:
            mapping = json.loads(line)

            if mapping.get("filename") != filename:
                continue

            src_path = mapping.get("original_path")
            tgt_path = mapping.get("transformed_path")

            if not isinstance(src_path, (list, tuple)):
                continue
            if not isinstance(tgt_path, (list, tuple)):
                continue

            src = tuple(src_path)
            tgt = tuple(tgt_path)

            if len(src) > 0 and len(tgt) > 0:
                gt.add((src, tgt))

    return gt

def convert_gt_to_indices(gt_pairs, source_nodes, target_nodes):
    """
    Convert ground truth node names into indices for BCE loss.

    Args:
        gt_pairs: Set of (src_tuple, tgt_tuple) ground truth pairs.
        source_nodes: List of source node tuples.
        target_nodes: List of target node tuples.
    Returns:
        List of (i, j) index pairs where source_nodes[i] matches target_nodes[j] according to gt_pairs.
    """
    
    src_map = {node: i for i, node in enumerate(source_nodes)}
    tgt_map = {node: j for j, node in enumerate(target_nodes)}

    indices = []
    
    for src, tgt in gt_pairs:
        i = src_map.get(src)
        j = tgt_map.get(tgt)

        if i is not None and j is not None:
            indices.append((i, j))

    return indices


# -----------------------------
# GRAPH PAIR OBJECT
# -----------------------------
class JsonGraphPair:
    """Represents a source-target graph pair and its ground truth matches."""

    def __init__(self, source_df, target_df, gt_pairs):
        self.filename = source_df.attrs["filename"]

        # ---- SOURCE GRAPH ----
        graph_src = build_graph(source_df["path"])
        self.source_nodes = list(graph_src.nodes())
        self.source_edge_index = from_networkx(graph_src).edge_index.long().to(device)
        self.source_emb, self.source_struct = combine_embeddings(source_df, graph_src)

        # ---- TARGET GRAPH ----
        graph_tgt = build_graph(target_df["path"])
        self.target_nodes = list(graph_tgt.nodes())
        self.target_edge_index = from_networkx(graph_tgt).edge_index.long().to(device)
        self.target_emb, self.target_struct = combine_embeddings(target_df, graph_tgt)

        # ---- GROUND TRUTH ----
        self.gt_pairs = gt_pairs
        self.gt_indices = convert_gt_to_indices(gt_pairs, self.source_nodes, self.target_nodes)

def split_pairs(pairs, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split a list of JsonGraphPair objects into train/val/test.

    Args:
        pairs: List of JsonGraphPair objects.
        train_ratio: Proportion of pairs to use for training.
        val_ratio: Proportion of pairs to use for validation.
        seed: Random seed for reproducibility.
    Returns:
        train_pairs, val_pairs, test_pairs: Three lists of JsonGraphPair objects.
    """
    pairs = list(pairs)
    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return pairs[:train_end], pairs[train_end:val_end], pairs[val_end:]


# -----------------------------
# LOSS FUNCTION
# -----------------------------
def matching_loss(source_tuple, target_tuple, gt_indices, loss_fn, alpha_type=1.0, alpha_struct=1.0):
    """
    Compute a combined loss that incorporates semantic similarity, type similarity, and structural similarity.

    Args:
        source_tuple: (source_emb, source_struct) where source_emb is (Ns, D) and source_struct is (Ns, STRUCT_DIM)
        target_tuple: (target_emb, target_struct) where target_emb is (Nt, D) and target_struct is (Nt, STRUCT_DIM)
        gt_indices: List of (i, j) index pairs indicating ground truth matches between source and target nodes.
        loss_fn: A binary classification loss function (e.g., BCEWithLogitsLoss) that takes (pred_matrix, gt_matrix) as input.
        alpha_type: Weight for the type similarity component in the combined similarity score.
        alpha_struct: Weight for the structural similarity component in the combined similarity score.
    Returns:
        A scalar loss value that can be backpropagated.
    """
    source_emb, source_struct = source_tuple
    target_emb, target_struct = target_tuple

    Ns, Nt = source_emb.shape[0], target_emb.shape[0]

    # ---- Semantic similarity ----
    sem_sim = F.normalize(source_emb, dim=1) @ F.normalize(target_emb, dim=1).T  # (Ns, Nt)

    # ---- Type similarity ----
    NUM_TYPES = source_struct.shape[1] - 5
    source_type = source_struct[:, -NUM_TYPES:]
    target_type = target_struct[:, -NUM_TYPES:]
    type_sim = source_type @ target_type.T
    type_sim = type_sim / NUM_TYPES

    # ---- Structural similarity ----
    source_struct_num = source_struct[:, :3]  # children, siblings, depth
    target_struct_num = target_struct[:, :3]
    diff = torch.cdist(source_struct_num, target_struct_num, p=1)
    struct_sim = 1.0 / (1.0 + diff)

    # ---- Combined similarity ----
    sim_matrix = sem_sim + alpha_type * type_sim + alpha_struct * struct_sim

    # ---- Ground truth matrix ----
    gt_matrix = torch.zeros((Ns, Nt), device=source_emb.device)
    for i, j in gt_indices:
        gt_matrix[i, j] = 1.0

    return loss_fn(sim_matrix, gt_matrix)

def compute_global_pos_weight(train_pairs):
    """
    Compute a global positive weight for BCE loss based on the ratio of positive to negative pairs across the entire training set.

    Args:
        train_pairs: List of JsonGraphPair objects in the training set.
    Returns:        
        A scalar value representing the positive weight to be used in BCEWithLogitsLoss.
    """

    total_pos = 0
    total_entries = 0
    for pair in train_pairs:
        Ns = pair.source_emb.shape[0]
        Nt = pair.target_emb.shape[0]
        total_entries += Ns * Nt
        total_pos += len(pair.gt_indices)
    return (total_entries - total_pos) / (total_pos + 1e-8)


# -----------------------------
# GCN MODEL
# -----------------------------
class GCN(torch.nn.Module):
    def __init__(self, struct_dim=STRUCT_DIM, hidden_dim=HIDDEN_DIM, codebert_dim=CODEBERT_DIM*2, out_dim=OUT_DIM):
        super().__init__()
        # Project structural features to embedding space
        self.struct_proj = torch.nn.Sequential(
            torch.nn.Linear(struct_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )
        # GCN layers: semantic + projected structural features
        self.conv1 = GCNConv(codebert_dim + 128, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x_tuple, edge_index):
        emb, struct_feat = x_tuple
        struct_proj = self.struct_proj(struct_feat)
        x = torch.cat([emb, struct_proj], dim=1)
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return F.normalize(h, dim=1)


# -----------------------------
# TRAINING LOOP
# -----------------------------
def train_model(train_pairs, val_pairs, alpha_type=1.0, alpha_struct=1.0):
    """
    Train the GCN model on the training set and evaluate on the validation set after each epoch.

    Args:
        train_pairs: List of JsonGraphPair objects for training.
        val_pairs: List of JsonGraphPair objects for validation.
        alpha_type: Weight for the type similarity component in the loss function.
        alpha_struct: Weight for the structural similarity component in the loss function.
    Returns:
        The trained GCN model.
    """
    wandb.init(project="json-graph-matching")
    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pos_weight = torch.tensor(compute_global_pos_weight(train_pairs), dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
        model.train()
        total_loss = 0.0

        for pair in train_pairs:
            optimizer.zero_grad()
            z_src_out = model((pair.source_emb, pair.source_struct), pair.source_edge_index)
            z_tgt_out = model((pair.target_emb, pair.target_struct), pair.target_edge_index)
            loss = matching_loss(
                (z_src_out, pair.source_struct),
                (z_tgt_out, pair.target_struct),
                pair.gt_indices,
                loss_fn,
                alpha_type=alpha_type,
                alpha_struct=alpha_struct
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_precision, val_recall, val_f1 = evaluate_model(model, val_pairs, silent=True)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_pairs),
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1
        })

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {total_loss/len(train_pairs):.4f} | "
              f"Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

    return model


# -----------------------------
# EVALUATION
# -----------------------------
def match_graphs(source_embs, target_embs, source_nodes, target_nodes):
    """
   Match based on cosine similarity of node embeddings.

    Args:
        source_embs: (Ns, D) tensor of source node embeddings.
        target_embs: (Nt, D) tensor of target node embeddings.
        source_nodes: List of source node tuples.
        target_nodes: List of target node tuples.
    Returns:
        A dictionary mapping each source node to a list of matched target nodes.
    """
    matches = {}
    logits = torch.matmul(source_embs, target_embs.T)
    for i, s_node in enumerate(source_nodes):
        best_idx = torch.argmax(logits[i])
        matches[s_node] = [target_nodes[best_idx]]
    return matches

def ensure_tuple(path):
    """
    Ensure the path is a tuple.
    
    Args:
        path: The path to convert to a tuple.
    Returns:
        A tuple representation of the path, or None if conversion fails.
    """
    if isinstance(path, tuple):
        return path
    if isinstance(path, list):
        return tuple(path)
    if isinstance(path, str):
        try:
            # Handles: "["rules", "braces"]"
            parsed = ast.literal_eval(path)
            if isinstance(parsed, (list, tuple)):
                return tuple(parsed)
        except:
            pass
    return None

def compute_metrics(matches, gt_pairs):
    """
    Compute precision, recall, F1 for predicted matches.

    Args:
        matches: Dict mapping source nodes to list of matched target nodes.
        gt_pairs: Set of (src_tuple, tgt_tuple) ground truth pairs.
    Returns:
        precision, recall, f1: Evaluation metrics.
    """

    predicted_pairs = set()

    for src, tlist in matches.items():
        src_t = ensure_tuple(src)
        if src_t is None:
            continue

        for tgt in tlist:
            tgt_t = ensure_tuple(tgt)
            if tgt_t is None:
                continue

            predicted_pairs.add((src_t, tgt_t))


    true_positives = predicted_pairs & gt_pairs
    precision = len(true_positives) / max(1, len(predicted_pairs))
    recall = len(true_positives) / max(1, len(gt_pairs))

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1

def evaluate_model(model, pairs, silent=False):
    """
    Evaluate the model on a list of JsonGraphPair objects.
    Args:
        model: The trained GCN model.
        pairs: List of JsonGraphPair objects to evaluate on.
        silent: If True, suppress per-file output.  
    Returns:
        Average precision, recall, and F1 across all pairs.
    """
    model.eval()
    all_precision, all_recall, all_f1 = [], [], []

    with torch.no_grad():
        for pair in pairs:
            z_src = model((pair.source_emb, pair.source_struct), pair.source_edge_index)
            z_tgt = model((pair.target_emb, pair.target_struct), pair.target_edge_index)
            matches = match_graphs(z_src, z_tgt, pair.source_nodes, pair.target_nodes)
            precision, recall, f1 = compute_metrics(matches, pair.gt_pairs)
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            if not silent:
                print(f"{pair.filename} | precision: {precision:.4f}, recall: {recall:.4f}, F1: {f1:.4f}")

    return np.mean(all_precision), np.mean(all_recall), np.mean(all_f1)


# -----------------------------
# SAVE / LOAD
# -----------------------------
def save_model(model, path="gcn_model.pt"):
    torch.save(model.state_dict(), path)

def load_model(path="gcn_model.pt"):
    model = GCN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# -----------------------------
# MAIN
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", help="Directory containing source CSV files")
    parser.add_argument("target_dir", help="Directory containing target CSV files")
    parser.add_argument("groundtruth_file", help="Path to ground truth JSONL file")
    parser.add_argument("mode", choices=["train", "eval"])
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir, target_dir = Path(args.source_dir), Path(args.target_dir)
    pairs = []

    # Load datasets and ground truth
    for file in tqdm(sorted(source_dir.glob("*.csv"))):
        filename = file.name
        src_df = load_dataset(source_dir / filename)
        tgt_df = load_dataset(target_dir / filename)
        src_df.attrs["filename"] = filename
        # Convert to .json for ground truth
        filename_json = filename.rsplit('.', 1)[0] + ".json"
        gt_pairs = get_ground_truth_pairs(args.groundtruth_file, filename_json)
        pairs.append(JsonGraphPair(src_df, tgt_df, gt_pairs))

    train_pairs, val_pairs, test_pairs = split_pairs(pairs)
    print(f"Datasets split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

    if args.mode == "train":
        model = train_model(train_pairs, val_pairs)
        save_model(model)
        test_precision, test_recall, test_f1 = evaluate_model(model, test_pairs, silent=True)
        print(f"Average Test Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    else:
        model = load_model()
        test_precision, test_recall, test_f1 = evaluate_model(model, test_pairs, silent=True)
        print(f"Average Test Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()