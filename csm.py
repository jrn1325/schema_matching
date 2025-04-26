import collections
import json
import pandas as pd
import random
import sys

from collections import defaultdict
from scipy.optimize import linprog
from gurobipy import *
from valentine import valentine_match
from valentine.algorithms import SimilarityFlooding, Coma, Cupid, DistributionBased, JaccardDistanceMatcher



def parse_document(doc, path=("$",)):
    """
    Recursively parse all paths from a JSON object using dot notation.

    Args:
        doc (dict or list): The JSON document.
        path (tuple): Current path.

    Yields:
        tuple: (dot-separated path, value)
    """
    if isinstance(doc, dict):
        for key, value in doc.items():
            current_path = path + (key,)
            if not isinstance(value, (dict, list)):
                yield ".".join(current_path), value
            else:
                yield from parse_document(value, current_path)

    elif isinstance(doc, list):
        for item in doc:
            current_path = path + ("*",)
            if not isinstance(item, (dict, list)):
                yield ".".join(current_path), item
            else:
                yield from parse_document(item, current_path)

    else:
        yield ".".join(path), doc

        
def extract_paths(doc):
    """
    Extract all paths from a document.

    Args:
        doc (dict or list): The JSON document.

    Returns:
        dict: A dictionary where the keys are paths and the values are sets of corresponding values.
    """
    path_dict = defaultdict(set)
    for path, value in parse_document(doc):
        path_dict[path].add(value)
    return path_dict


def path_dict_to_df(path_dict):
    """
    Convert a path_dict to a DataFrame.

    Args:
        path_dict (dict): Dictionary of paths and their corresponding values.
    Returns:
        pd.DataFrame: A DataFrame with paths as columns and values as rows.
    """
    # Convert sets to lists and take the first element if there are any values
    flat_dict = {path: list(values)[0] if values else None for path, values in path_dict.items()}
    return pd.DataFrame([flat_dict])


def sample_documents(json_file_path, seed=1):
    """
    Select two random documents from a JSON lines file and extract paths from them.

    Args:
        json_file_path (str): Path to the JSONL file.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (source_df, target_df)
    """
    with open(json_file_path) as f:
        docs = [json.loads(line) for line in f]

    if len(docs) < 2:
        raise ValueError("Need at least 2 documents to sample source and target.")

    random.seed(seed)
    source_doc, target_doc = random.sample(docs, 2)

    # Extract paths from the source and target documents
    source_dict = extract_paths(source_doc)
    target_dict = extract_paths(target_doc)

    # Convert the path dictionaries to DataFrames
    source_df = path_dict_to_df(source_dict)
    target_df = path_dict_to_df(target_dict)

    return source_df, target_df


def normalize_json(json_file_path):
    """Normalize the JSON file by converting it into a DataFrame.

    Args:
        json_file_path (str): Path of JSON file

    Returns:
        pd.DataFrame: A DataFrame
    """
    
    df_list = []
    with open(json_file_path) as json_file:
        for line in json_file:
            json_data = json.loads(line)
            df = pd.json_normalize(json_data)
            df_list.append(df)
    return pd.concat(df_list).reset_index(drop = True)


def find_valentine(df1, df2, ground_truth):
    """Match source keys to target keys using the Valentine library's Similarity Flooding algorithm.

    Args:
        df1 (pd.DataFrame): The target dataframe containing the keys to be matched.
        df2 (pd.DataFrame): The source dataframe containing the keys to be matched.
        ground_truth (list): A list of tuples containing the ground truth matches, where each tuple is in the form (target_key, source_key).


    Returns:
        match_dict (dict): A dictionary where keys are target keys from df1 and values are lists of matching source keys from df2 based on the ground truth.
    """

    # Instantiate matcher and run the matching algorithm
    matcher = Cupid()
    matches = valentine_match(df1, df2, matcher, "target", "source")
    return matches

    
def get_key_prefix(key):
    """
    Get the prefix of a key.

    Args:
        key (str): Target or source key.

    Returns:
        str: The prefix of the key.
    """
    # If no dot in the key, the prefix is just the '$' symbol
    if '.' not in key:
        prefix = '$'
    else:
        prefix, _ = key.rsplit('.', maxsplit=1)  # Split at the last dot
        prefix = '$.' + prefix  # Add '$.' to indicate the start of the path
    return prefix


def predict_matches(source_vars, grouped_by_target):
    """
    Get the final matches.

    Args:
        source_vars (dict): Variables of source paths.
        grouped_by_target (dict): Grouped target paths.

    Returns:
        dict: Dictionary of keys that definitely match.
    """

    final_matching = {}
    # Iterate over target paths and their corresponding source paths
    #for t_path, s_paths in grouped_by_target.items():
    for s_path, var_list in source_vars.items():
        for var, t_path in var_list:
        # Check each binary variable associated with this source path
            if bool(round(var.X)):  # Check if the binary variable indicates a match
                final_matching[s_path] = t_path
                break
    return final_matching


def quadratic_programming(match_dict):
    """
    Sets up a quadratic programming model using Gurobi to find the optimal matching between 
    target and source keys based on the provided match dictionary. It creates binary variables for potential 
    matches and nested relationships, sets up constraints, and defines an objective function to maximize 
    the number of matches while penalizing deeper prefixes more heavily.

    Args:
        match_dict (dict): A dictionary where keys are target keys and values are lists of source keys 
                           that possibly match.

    Returns:
        dict: A dictionary where keys are source keys and values are target keys that definitely match.
    """

    # Initialize the quadratic model
    quadratic_model = Model("quadratic")

    # Dictionaries to store binary variables
    nested_t_vars = {}
    prefix_vars = {} 
    source_vars = collections.defaultdict(list)
    all_vars = []
    score_vars = []
    
    # Organizing data by target paths
    grouped_by_target = defaultdict(list)
    for (t_path, s_path), score in match_dict.items():
        grouped_by_target[t_path].append((s_path, score))

    # Loop over matching target and its matching source paths
    for t_path, s_paths in grouped_by_target.items():
        t_prefix = 't' + get_key_prefix(t_path)

        # Dictionary to store the binary variables for nested source paths
        nested_s_vars = {}
        
        for s_path, score in s_paths:
            s_prefix = 's' + get_key_prefix(s_path)

            # Create a binary variable for "nested" paths
            nested_path = f'{t_path}-----{s_path}'
            s_var = quadratic_model.addVar(name=nested_path, vtype=GRB.BINARY)
            source_vars[s_path].append((s_var, t_path))
            all_vars.append(s_var)
            score_vars.append(score * s_var)

            # Create a binary variable for "prefix" paths
            prefix_path = f'{t_prefix}-----{s_prefix}'
            if prefix_path not in prefix_vars:
                prefix_vars[prefix_path] = quadratic_model.addVar(name=prefix_path, vtype=GRB.BINARY)
            prefix_var = prefix_vars[prefix_path]

            # Create a binary variable for "nested" target paths
            nested_t_path = f'{t_path}-----{s_prefix}'
            if nested_t_path not in nested_t_vars:
                nested_t_vars[nested_t_path] = quadratic_model.addVar(name=nested_t_path, vtype=GRB.BINARY)
            nested_t_var = nested_t_vars[nested_t_path]

            # Create a binary variable for "nested" source paths
            nested_s_path = f'{s_path}-----{t_prefix}'
            if nested_s_path not in nested_s_vars:
                nested_s_vars[nested_s_path] = quadratic_model.addVar(name=nested_s_path, vtype=GRB.BINARY)
            nested_s_var = nested_s_vars[nested_s_path]
            
            # Add constraints
            quadratic_model.addConstr(nested_t_var <= prefix_var, name=f"nested_target_prefix_{t_path}_{s_path}")
            quadratic_model.addConstr(nested_s_var <= prefix_var, name=f"nested_source_prefix_{t_path}_{s_path}")
    
    # Ensure each source path matches at most one target path
    for s_path in source_vars:
        total = sum(s_var for s_var, _ in source_vars[s_path])
        quadratic_model.addConstr(total <= 1, name=f"source_match_once_{s_path}")

    # Update model before setting objective
    quadratic_model.update()

    # --- SMARTER OBJECTIVE FUNCTION: Adaptive Prefix Penalty ---
    prefix_penalties = []
    for prefix_path, prefix_var in prefix_vars.items():
        # Extract the raw prefix by removing '$.'
        raw_prefix = prefix_path.split('$.', 1)[1]  # Remove the '$.' at the start

        # Calculate the depth of the prefix, i.e., how many components there are in the path
        nesting_depth = raw_prefix.count('.') + 1  # The number of components separated by dots

        # Calculate the penalty weight based on the depth of the prefix
        penalty_weight = 0.001 * nesting_depth  # Penalty grows linearly with depth
        prefix_penalties.append(penalty_weight * prefix_var)

    # Set the objective: maximize match scores minus adaptive prefix penalties
    quadratic_model.setObjective(sum(score_vars) - sum(prefix_penalties), GRB.MAXIMIZE)
    quadratic_model.setParam("OutputFlag", False)
    quadratic_model.optimize()

    # Handle infeasibility
    if quadratic_model.getAttr(GRB.Attr.Status) == GRB.INFEASIBLE:
        quadratic_model.computeIIS()
        quadratic_model.write("iis.ilp")
        print("Model is infeasible. Check iis.ilp for details.")
        return {}

    quadratic_model.write("model.lp")

    return predict_matches(source_vars, grouped_by_target)

