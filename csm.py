import collections
import json
import pandas as pd
import sys

import valentine as va
import valentine.algorithms 
import valentine.metrics as valentine_metrics
from collections import defaultdict
from scipy.optimize import linprog
from gurobipy import *


def json_to_dataframe(json_file_path):
    """
    Purpose: Normalize the JSON file
    @param: json_file_path: Path of JSON file
    @returns: dataframe: Flat table 
    """
    df_list = []
    with open(json_file_path) as json_file:
        for line in json_file:
            json_data = json.loads(line)
            df = pd.json_normalize(json_data)
            df_list.append(df)
    return pd.concat(df_list).reset_index(drop = True)


def find_valentine(df1, df2, ground_truth):
    """
    Purpose: Match source keys to target keys
    @param: df1: Target dataframe, df2: Source dataframe
    @returns: match_dict: dictionary of matching keys
    """
    matcher = valentine.algorithms.SimilarityFlooding()
    matches = va.valentine_match(df1, df2, matcher, "target", "source")
    metrics = valentine_metrics.all_metrics(matches, ground_truth)
    match_dict = defaultdict(list)
    for match in ground_truth:
        match_dict[match[0]].append(match[1])
    return match_dict


def linear_programming(match_dict):
    problem  = LpProblem(name = "Match_columns", sense = LpMaximize)
    nested_pairs = {} 
    all_variables = []

    for t_key, s_keys in match_dict.items():
        target_prefix = ""
        # Get the prefix of the target key if it has one
        if '.' not in t_key:
            target_prefix = '_tROOT_'
        else:
            target_prefix, _ = t_key.rsplit('.', maxsplit = 1)
            target_prefix = '_tROOT_' + target_prefix
        
        variables = []
        nested_vars = {}
        
        # Loop over source keys that match with target key
        for s_key in s_keys:
            # Create binary variable for matching source key
            var_name = 'T_' + t_key + '-----' + 'S_' + s_key
            s_x = LpVariable(name = var_name, lowBound = 0, upBound = 1, cat = "Binary")
            all_variables.append(s_x)
            variables.append(s_x)
            
            # Get the prefix of the source key if it has one. Create a.b_O.d, ...
            if '.' not in s_key:
                source_prefix = '_sROOT_'
            else:
                source_prefix, _ = s_key.rsplit('.', maxsplit = 1)
                source_prefix = '_sROOT_' + source_prefix
            
            # Get the nested key
            nested_key = 'T_' + t_key + '_O_S' + source_prefix
            # Create a variable for each nested key
            if nested_key not in nested_vars:
                nested_vars[nested_key] = LpVariable(name = nested_key, lowBound = 0, upBound = 1, cat = "Binary")
            nested_var = nested_vars[nested_key]
            #print("nested", nested_var)
                
            # Constraint a.O_d >= a.b_d.e (and a.O_d >= a.b_d.f, ...)
            problem += (nested_var >= s_x)

            # Create a variable for pair key. P.a_O.d (and P.a_O, ...)
            pair_key = 'T' + target_prefix + 'S' + source_prefix
            if pair_key not in nested_pairs:
                nested_pairs[pair_key] = LpVariable(name = pair_key, lowBound = 0, upBound = 1, cat = "Binary")
            pair_var = nested_pairs[pair_key]
            #print("pair", pair_var)
            # Constraint P.a_O.d <= a.b_O.d (and P.a_O.d >= a.c_O.d, ...)
            problem += pair_var >= nested_var

        # At most one matching for each path in the source schema
        problem += lpSum(variables) <= 1

    # Objective function: P.a_O.d + P.a_O + ...
    # problem += lpSum(list(nested_pairs.values()))
    problem += lpSum(all_variables)
    LpSolverDefault.msg = 1    
    # Solve the objective function
    status = problem.solve(PULP_CBC_CMD(msg = 1))
    print("Status:", LpStatus[status])

    # Loop over problem variables and print their optimum values
    for variable in problem.variables():
        print(variable.name, "=", variable.varValue)
    

def get_key_prefix(key):
    """
    Purpose: Get the prefix of a key
    @param: key: target or source key
    @returns: prefix
    """
    if '.' not in key:
        prefix = 'ROOT_'
    else:
        prefix, _ = key.rsplit('.', maxsplit = 1)
        prefix = 'ROOT_' + prefix
    return prefix


def quadratic_programming(match_dict):
    """
    Purpose: Create a quadratic model to choose the best match
    @param: match_dict: Dictionary of keys that match
    @returns:
    """
    #try:

    # Initialize the quadratic model
    quadratic_model = Model("quadratic")

    # Dictionary to store the binary variables for nested target keys
    nested_t_vars = {}

    # Dictionary to store the binary variables for root target and source keys
    root_vars = {} 

    # Dictionary to store source variables
    source_vars = collections.defaultdict(list)

    all_vars = []

    # Loop over matching target and source keys
    for t_key, s_keys in match_dict.items():
        target_prefix = 't' + get_key_prefix(t_key)

        # List to store the binary variables for nested source keys
        nested_s_vars = {}
        
        # Loop over source keys
        for s_key in s_keys:
            # Create a binary variable for the match between target and source keys
            nested_var_name = 'T_' + t_key + '-----' + 'S_' + s_key
            s_var = quadratic_model.addVar(name=nested_var_name, vtype=GRB.BINARY)  
            source_vars[s_key].append((s_var, t_key))
            all_vars.append(s_var)
            
            # Create a binary variable for "nested" target keys
            nested_t_key = 'T_' + t_key + '-----' + 'S_' + 's' + get_key_prefix(s_key)
            if nested_t_key not in nested_t_vars:
                nested_t_vars[nested_t_key] = quadratic_model.addVar(name=nested_t_key, vtype=GRB.BINARY)
            nested_t_var = nested_t_vars[nested_t_key]
            
            # Create a binary variable for "root" target keys
            root_var_name = 'T_' + target_prefix + '-----' + 'S_' + 's' + get_key_prefix(s_key)
            if root_var_name not in root_vars:
                root_vars[root_var_name] = quadratic_model.addVar(name=root_var_name, vtype=GRB.BINARY)
            root_var = root_vars[root_var_name]
            
            # Create a binary variable for "nested" source keys
            nested_s_key = 'T_' + target_prefix + '-----' + 'S_' + s_key
            if nested_s_key not in nested_s_vars:
                nested_s_vars[nested_s_key] = quadratic_model.addVar(name=nested_s_key, vtype=GRB.BINARY)
            nested_s_var = nested_s_vars[nested_s_key]

            # Add constraints
            quadratic_model.addConstr(nested_t_var - root_var <= 0, name="Matching nested target implies matching root target.")
            quadratic_model.addConstr(nested_s_var - root_var <= 0, name="Matching nested source implies matching root source.")
    
    # Call update if you need to examine the model because optimization
    quadratic_model.update()
   
    # Loop over the source keys
    for s_key in s_keys:
        # Sum source variables constraints, It must be <= 1
        total = sum(s[0] for s in source_vars[s_key])
        #print('TOTAL:', total)
        # Add constraint: 
        quadratic_model.addConstr(total <= 1, name="Each source key can be matched to at most one target key.")


    # Objective function
    quadratic_model.setObjective(sum(all_vars), GRB.MAXIMIZE)
    quadratic_model.setParam("OutputFlag", False)
    quadratic_model.optimize()
    if quadratic_model.getAttr(GRB.Attr.Status) == GRB.INFEASIBLE:
        quadratic_model.computeIIS()
        quadratic_model.write('iis.ilp')
    quadratic_model.write('model.lp')

    final_match_dict = show_matches(source_vars, match_dict)
    
    #print(f"Optimal objective value: {quadratic_model.objVal}")
    # Loop over model variables and print their rounded optimum values
    ''' 
    for variable in quadratic_model.getVars():
        print(variable.varName, "=", bool(round(variable.x)))
    print()
    '''
    return final_match_dict



def show_matches(source_vars, match_dict):
    # Create a function that takes in source and target dataframes, possible matchings and final matchings
    final_matching = {}
    for s_keys in match_dict.values():
        for s_key in s_keys:
            for (s_var, t_key) in source_vars[s_key]:
                if bool(round(s_var.X)):
                    final_matching[s_key] = t_key
                    break
    return final_matching
