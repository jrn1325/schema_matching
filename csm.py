import csv
import json
import numpy as np
import pandas as pd
import sys

import valentine as va
import valentine.algorithms 
import valentine.metrics as valentine_metrics
from collections import defaultdict
from scipy.optimize import linprog
from pulp import *
from gurobipy import *


def read_json(json_file_path):
    '''
    Input: json file path
    Output: dataframe
    Purpose: Load json file into object and put in dataframe
    '''
    # Create a list of dataframes
    df_list = []
    
    with open(json_file_path) as json_file:
        # Loop over each line
        for line in json_file:
            # Load line into json
            json_data = json.loads(line)
            # Convert json to dataframe
            df = pd.json_normalize(json_data)
            df_list.append(df)
    # Return merged dataframe
    return pd.concat(df_list).reset_index(drop = True)


def find_valentine(df1, df2):
    """
    Input: target and source dataframes
    Output: dictionary of elements that match
    Purpose: Match source keys to target keys
    """
    # Pick a matcher
    matcher = valentine.algorithms.SimilarityFlooding()
    # Find matches
    matches = va.valentine_match(df1, df2, matcher, "target", "source")
    #for key, value in matches.items():
    #    print(key, value)    
    # Define ground truth
    ground_truth = [("t_name.firstname", "s_name.firstname"), ("t_name.lastname", "s_name.lastname"), ("t_sex", "s_sex"), ("t_age", "s_age"), ("t_name.firstname", "firstname"), ("t_name.lastname", "lastname")]
    
    # Create dictionary
    match_dict = defaultdict(list)
    for match in ground_truth:
        match_dict[match[0]].append(match[1])
    # Calculate metrics
    metrics = valentine_metrics.all_metrics(matches, ground_truth)
    #print()
    #for key, value in metrics.items():
    #    print(key, value)
    
    return match_dict


def linear_programming(match_dict):
    print(match_dict)
    # Create a maximization problem
    problem  = LpProblem(name = "Match_columns", sense = LpMaximize)

    # Create a quadratic model
    quadratic_model = Model("quadratic")
    nested_pairs = {} 
    all_variables = []

    
    
    # Loop over dictionary of matching keys
    for t_key, s_keys in match_dict.items():
        #print(t_key, s_keys)
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
            print(1, var_name)
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
                print(2, nested_key)
            nested_var = nested_vars[nested_key]
            #print("nested", nested_var)
                
            # Constraint a.O_d >= a.b_d.e (and a.O_d >= a.b_d.f, ...)
            problem += (nested_var >= s_x)

            # Create a variable for pair key. P.a_O.d (and P.a_O, ...)
            pair_key = 'T' + target_prefix + 'S' + source_prefix
            if pair_key not in nested_pairs:
                nested_pairs[pair_key] = LpVariable(name = pair_key, lowBound = 0, upBound = 1, cat = "Binary")
                print(3, pair_key)
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
    

def quadratic_programming(match_dict):
    try:

        # Create a quadratic model
        quadratic_model = Model("quadratic")

        nested_pairs = {} 
        all_variables = []
        
        # Loop over dictionary of matching keys
        for t_key, s_keys in match_dict.items():
            #print(t_key, s_keys)
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
                s_x = quadratic_model.addVar(name = var_name, vtype = GRB.BINARY)
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
                    nested_vars[nested_key] = quadratic_model.addVar(name = nested_key, vtype = GRB.BINARY)
                nested_var = nested_vars[nested_key]
                #print("nested", nested_var)
                    
                # Constraint a.O_d >= a.b_d.e (and a.O_d >= a.b_d.f, ...)
                quadratic_model.addConstr(nested_var >= s_x)
                
                # Create a variable for pair key. P.a_O.d (and P.a_O, ...)
                pair_key = 'T' + target_prefix + 'S' + source_prefix
                if pair_key not in nested_pairs:
                    nested_pairs[pair_key] = quadratic_model.addVar(name = pair_key, vtype = GRB.BINARY)
                pair_var = nested_pairs[pair_key]
                #print("pair", pair_var)
                # Constraint P.a_O.d <= a.b_O.d (and P.a_O.d >= a.c_O.d, ...)
                quadratic_model.addConstr(pair_var >= nested_var)

                # Add more constraints
                firstname_T1 = 'S' + skey + '_' + T_t_name_firstname + S_firstname_T_t_name_lastname (0,1)
                lastname_T1 = S_lastname_T_t_name_firstname + S_lastname_T_t_name_lastname (0,1)
                firstname_lastname_T1_count = firstname_T1 + lastname_T1 (0,1, 2)
                firstname_lastname_T1 (0,1)


        # Objective function: P.a_O.d + P.a_O + ...
        quadratic_model.setObjective(sum(all_variables), GRB.MAXIMIZE)
        # problem += lpSum(list(nested_pairs.values()))
        #problem += lpSum(all_variables)
        
        # Optimize the model
        quadratic_model.setParam("OutputFlag", False)
        quadratic_model.optimize()

        print(f"Optimal objective value: {quadratic_model.objVal}" )
        # Loop over model variables and print their optimum values
        for variable in quadratic_model.getVars():
            print(variable.varName, "=", variable.x)
    except GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except AttributeError:
        print("Encountered an attribute error")

def main():
    target_file = "schema_matching/files/target_schema.json"
    source_file = "schema_matching/files/source_schema.json"

    # Preprocess target and source data
    df_target = read_json(target_file)
    df_source = read_json(source_file)
    match_dict = find_valentine(df_target, df_source)
    #linear_programming(match_dict)
    quadratic_programming(match_dict)

if __name__ == "__main__":
    main()
     


