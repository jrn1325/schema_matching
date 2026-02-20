import argparse
import base64
import json
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

# -----------------------------
# DATA UTILITIES
# -----------------------------
def load_dataset(path):
    return pd.read_csv(path, delimiter=';')

def build_graph(paths):
    """
    Build a graph from a list of paths. Each path is a node, and edges connect parent-child paths.

    Args:
        paths: list of string paths (e.g. "root.child1.child2")
    Returns:
        G: networkx Graph with nodes as paths and edges connecting parent-child relationships
    """
    G = nx.Graph()
    for path in paths:
        path = str(path)
        G.add_node(path)
        if '.' in path:
            parent = path.rsplit('.', 1)[0]
            G.add_edge(parent, path)
    return G

def decode_embedding(b64_string, dim=CODEBERT_DIM):
    """
    Decode a base64 string into a numpy array of the specified dimension.

    Args:
        b64_string: base64-encoded string of the embedding
        dim: expected dimension of the embedding vector
    Returns:
        numpy array of shape (dim,) containing the decoded embedding
    """
    if not b64_string:
        return np.zeros(dim, dtype=np.float32)
    byte_data = base64.b64decode(b64_string)
    arr = np.frombuffer(byte_data, dtype=np.float32)
    return arr.reshape(dim)

def combine_embeddings(df, graph, dim=CODEBERT_DIM):
    """
    Combine path and value embeddings for each node in the graph. 

    Args:
        df: DataFrame containing 'path', 'path_emb', and 'values_emb' columns
        graph: networkx Graph with nodes corresponding to paths in df
        dim: dimension of each individual embedding (path and value)
    Returns:
        Tensor of shape (num_nodes, dim*2) containing combined embeddings for each node
    """
    df_lookup = {row.path: row for row in df.itertuples(index=False)}
    embeddings = []

    for node in graph.nodes():
        row = df_lookup.get(node)

        if row is None:
            path_emb = torch.zeros(dim)
            value_emb = torch.zeros(dim)

        else:
            path_emb = torch.tensor(
                decode_embedding(row.path_emb),
                dtype=torch.float32
            )

            value_emb = torch.tensor(
                decode_embedding(row.values_emb),
                dtype=torch.float32
            )

        combined = torch.cat([path_emb, value_emb], dim=0)
        embeddings.append(combined)

    emb = torch.stack(embeddings).to(device)

    return F.normalize(emb, dim=1)

def get_ground_truth_pairs(ground_truth_path, filename):
    """
    Load ground truth pairs for a specific filename from the ground truth file.

    Args:
        ground_truth_path: path to the ground truth JSONL file
        filename: name of the file to extract ground truth pairs for (e.g. "example.json")
    Returns:
        set of (source_node, target_node) pairs that are ground truth matches for the given filename
    """

    gt = set()
    with open(ground_truth_path, "r") as f:
        for line in f:
            mapping = json.loads(line)

            gt_filename = mapping.get("filename")
            if gt_filename != filename:
                continue

            src = mapping.get("original_path")
            tgt = mapping.get("transformed_path")
            if src and tgt:
                gt.add((src, tgt))

    return gt

def convert_gt_to_indices(gt_pairs, source_nodes, target_nodes):
    """
    Convert ground truth pairs of node names into index pairs based on their positions in the source and target node lists.

    Args:
        gt_pairs: set of (source_node, target_node) pairs that are ground truth matches
        source_nodes: list of node names in the source graph
        target_nodes: list of node names in the target graph
    Returns:    
        list of (source_index, target_index) pairs corresponding to the ground truth matches
    """
    src_map = {node: i for i, node in enumerate(source_nodes)}
    tgt_map = {node: j for j, node in enumerate(target_nodes)}

    indices = []

    for src, tgt in gt_pairs:
        if src in src_map and tgt in tgt_map:
            indices.append((src_map[src], tgt_map[tgt]))

    return indices

class JsonGraphPair:

    def __init__(self, source_df, target_df, gt_pairs):

        self.filename = source_df.attrs["filename"]

        # ---- SOURCE GRAPH ----
        graph_src = build_graph(source_df["path"])
        self.source_nodes = list(graph_src.nodes())
        self.source_edge_index = (
            from_networkx(graph_src)
            .edge_index
            .long()
            .to(device)
        )
        self.source_features = combine_embeddings(source_df, graph_src)

        # ---- TARGET GRAPH ----
        graph_tgt = build_graph(target_df["path"])
        self.target_nodes = list(graph_tgt.nodes())
        self.target_edge_index = (
            from_networkx(graph_tgt)
            .edge_index
            .long()
            .to(device)
        )
        self.target_features = combine_embeddings(target_df, graph_tgt)

        # ---- GROUND TRUTH ----
        self.gt_pairs = gt_pairs
        self.gt_indices = convert_gt_to_indices(
            gt_pairs,
            self.source_nodes,
            self.target_nodes
        )

def split_pairs(pairs, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split a list of JsonGraphPair objects into train, validation, and test sets based on specified ratios.

    Args:
        pairs: list of JsonGraphPair objects to split
        train_ratio: proportion of pairs to use for training
        val_ratio: proportion of pairs to use for validation
    Returns:
        train_pairs: list of JsonGraphPair objects for training
        val_pairs: list of JsonGraphPair objects for validation
        test_pairs: list of JsonGraphPair objects for testing
    """ 
    pairs = list(pairs)  
    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return pairs[:train_end], pairs[train_end:val_end], pairs[val_end:]


# -----------------------------
# MODEL
# -----------------------------
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(CODEBERT_DIM*2, HIDDEN_DIM)
        self.conv2 = GCNConv(HIDDEN_DIM, OUT_DIM)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return F.normalize(h, dim=1)


# -----------------------------
# LOSS
# -----------------------------
def compute_similarity_matrix(source_embs, target_embs):
    """
    Compute cosine similarity matrix between source and target embeddings.

    Args:
        source_embs: (Ns, d) source embeddings
        target_embs: (Nt, d) target embeddings
    Returns:
        (Ns, Nt) similarity matrix where S[i, j] is the cosine similarity between source_embs[i] and target_embs[j]
    """

    source_embs = F.normalize(source_embs, dim=1)
    target_embs = F.normalize(target_embs, dim=1)

    return torch.matmul(source_embs, target_embs.T)

def build_gt_matrix(gt_indices, Ns, Nt, device):
    """
    Build a binary ground truth matrix of shape (Ns, Nt) where gt_matrix[i, j] = 1 if (i, j) is a ground truth match and 0 otherwise.

    Args:
        gt_indices: list of (source_index, target_index) pairs that are ground truth matches
        Ns: number of source nodes
        Nt: number of target nodes
        device: torch device to create the matrix on
    Returns:
        gt_matrix: (Ns, Nt) binary matrix with 1s at ground truth match positions
    """
    gt_matrix = torch.zeros((Ns, Nt), device=device)

    for src_i, tgt_i in gt_indices:
        gt_matrix[src_i, tgt_i] = 1.0

    return gt_matrix

def matching_loss(source_embs, target_embs, gt_indices, loss_fn):
    """
    Compute the loss for a pair of graphs based on the similarity matrix and ground truth matches.

    Args:
        source_embs: (Ns, d) source embeddings
        target_embs: (Nt, d) target embeddings
        gt_indices: list of (source_index, target_index) pairs that are ground truth matches
        loss_fn: binary classification loss function (e.g. BCEWithLogitsLoss)
    Returns:
        loss: scalar loss value for the given pair of graphs
    """
    sim_matrix = compute_similarity_matrix(source_embs, target_embs)
    gt_matrix = build_gt_matrix(
        gt_indices,
        source_embs.shape[0],
        target_embs.shape[0],
        source_embs.device
    )

    return loss_fn(sim_matrix, gt_matrix)


# -----------------------------
# TRAINING
# -----------------------------
def compute_global_pos_weight(train_pairs):
    """
    Compute a global positive class weight for BCE loss based on the ratio of positive to negative pairs across the entire training set.    

    Args:
        train_pairs: list of JsonGraphPair objects in the training set
    Returns:
        pos_weight: float value to use as the pos_weight in BCEWithLogitsLoss
    """

    total_pos = 0
    total_entries = 0

    for pair in train_pairs:
        Ns = pair.source_features.shape[0]
        Nt = pair.target_features.shape[0]
        total_entries += Ns * Nt
        total_pos += len(pair.gt_indices)

    return (total_entries - total_pos) / (total_pos + 1e-8)

def train_model(train_pairs, val_pairs):
    """
    Train GCN with BCE loss and evaluate on validation set each epoch.

    Args:
        train_pairs: list of JsonGraphPair objects for training
        val_pairs: list of JsonGraphPair objects for validation
    Returns:
        model: trained GCN model
    """
    wandb.init(project="json-graph-matching")

    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):

        model.train()
        total_loss = 0.0

        for pair in train_pairs:
            optimizer.zero_grad()

            # Forward pass
            z_src = model(pair.source_features, pair.source_edge_index)
            z_tgt = model(pair.target_features, pair.target_edge_index)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(compute_global_pos_weight(train_pairs)).to(device))

            # Compute BCE loss
            loss = matching_loss(z_src, z_tgt, pair.gt_indices, loss_fn)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        val_precision, val_recall, val_f1 = evaluate_model(model, val_pairs, silent=True)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_pairs),
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1
        })

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {total_loss/len(train_pairs):.4f} | Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

    return model


# -----------------------------
# EVALUATION
# -----------------------------
def match_graphs(source_embs, target_embs, source_nodes, target_nodes):
    """
    Compute matches from BCE logits using sigmoid + threshold.

    Args:
        source_embs: (Ns, d) source embeddings
        target_embs: (Nt, d) target embeddings
        source_nodes: list of source node names
        target_nodes: list of target node names
    Returns:
        dict: {source_node: [matched_target_nodes]}
    """
    matches = {}

    logits = torch.matmul(source_embs, target_embs.T)

    for i, s_node in enumerate(source_nodes):
        best_idx = torch.argmax(logits[i])
        matches[s_node] = [target_nodes[best_idx]]

    return matches

def compute_metrics(matches, gt_pairs):
    """
    Compute precision, recall, F1 for predicted matches.
    
    Args:
        matches: dict {source_node: [target_nodes]}
        gt_pairs: set of (source_node, target_node)

    Returns:
        precision: float
        recall: float
        f1: float
    """
    predicted_pairs = set()
    for src, tgt_list in matches.items():
        for tgt in tgt_list:
            predicted_pairs.add((src, tgt))

    true_positives = predicted_pairs & gt_pairs
    precision = len(true_positives) / max(1, len(predicted_pairs))
    recall = len(true_positives) / max(1, len(gt_pairs))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def evaluate_model(model, pairs, silent=False):
    """
    Evaluate a trained model on a list of JsonGraphPair objects.

    Args:
        model: trained GCN model
        pairs: list of JsonGraphPair objects to evaluate on
        silent: if True, suppress printing of individual pair results
    Returns:
        avg_precision: average precision across all pairs
        avg_recall: average recall across all pairs
        avg_f1: average F1 score across all pairs
    """
    model.eval()
    all_precision = []
    all_recall = []
    all_f1 = []

    with torch.no_grad():
        for pair in pairs:
            z_src = model(pair.source_features, pair.source_edge_index)
            z_tgt = model(pair.target_features, pair.target_edge_index)

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
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)

    pairs = []

    # Load all datasets and ground truth
    for file in tqdm(sorted(source_dir.glob("*.csv"))):
        filename = file.name

        src_df = load_dataset(source_dir / filename)
        tgt_df = load_dataset(target_dir / filename)

        src_df.attrs["filename"] = filename

        # Change extension to .json for ground truth lookup
        filename = filename.rsplit('.', 1)[0] + ".json"
        gt_pairs = get_ground_truth_pairs(args.groundtruth_file, filename)

        pairs.append(JsonGraphPair(src_df, tgt_df, gt_pairs))

    # Split into train / validation / test
    train_pairs, val_pairs, test_pairs = split_pairs(pairs)
    print(f"Datasets split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

    if args.mode == "train":
        model = train_model(train_pairs, val_pairs)
        save_model(model)
        test_precision, test_recall, test_f1 = evaluate_model(model, test_pairs, silent=True)
        print(f"Average Test Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    else:  # eval mode
        model = load_model()
        test_precision, test_recall, test_f1 = evaluate_model(model, test_pairs, silent=True)
        print(f"Average Test Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()