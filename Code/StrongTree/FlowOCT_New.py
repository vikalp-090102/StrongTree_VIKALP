#!/usr/bin/python
import sys
import os
import time
import getopt
import csv
import pandas as pd
from mip import Model, BINARY, CONTINUOUS, minimize  # python-mip with CBC
from sklearn.model_selection import train_test_split

# ---------------------------
# Logger: Redirect output to both console and a log file.
# ---------------------------
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ---------------------------
# Tree class: For storing tree properties (here, just depth)
# ---------------------------
class Tree:
    def __init__(self, depth):
        self.depth = depth

# ---------------------------
# Dummy evaluation functions.
# Replace these with your actual evaluation logic.
# ---------------------------
def get_acc(primal, data, b_value, beta_value, p_value):
    # Dummy: always returns 0.95.
    return 0.95

def get_mae(primal, data, b_value, beta_value, p_value):
    return 0.1

def get_mse(primal, data, b_value, beta_value, p_value):
    return 0.05

def get_r_squared(primal, data, b_value, beta_value, p_value):
    return 0.9

def print_tree(primal, b_value, beta_value, p_value):
    print("Decision Tree Structure:")
    print("b values:", b_value)
    print("beta values:", beta_value)
    print("p values:", p_value)

# ---------------------------
# FlowOCT class using python-mip and CBC solver.
# ---------------------------
class FlowOCT:
    def __init__(self, data, label, tree, lam, time_limit, mode):
        self.data = data
        self.label = label
        self.tree = tree
        self.lam = lam
        self.time_limit = time_limit
        self.mode = mode
        # Create a MILP model with minimization and force CBC as solver.
        self.model = Model(sense=minimize, solver_name="cbc")
        # These lists will hold decision variables. You can modify this formulation as needed.
        self.b = []      # e.g. binary variables for branch decisions
        self.beta = []   # continuous parameters (e.g. thresholds or slopes)
        self.p = []      # continuous parameters (e.g. predictions)
    
    def create_primal_problem(self):
        # For demonstration purposes, we create one variable in each list.
        self.b.append(self.model.add_var(var_type=BINARY, name="b0"))
        self.beta.append(self.model.add_var(var_type=CONTINUOUS, lb=-10, ub=10, name="beta0"))
        self.p.append(self.model.add_var(var_type=CONTINUOUS, lb=0, ub=10, name="p0"))
        
        # A dummy objective that penalizes the binary variable with lambda and adds the continuous variables.
        self.model.objective = minimize(self.lam * self.b[0] + self.beta[0] + self.p[0])
        # Dummy constraint: force beta0 + p0 to be at least 1.
        self.model.add_constr(self.beta[0] + self.p[0] >= 1)
        
        # Set a time limit (python-mip supports this via the max_seconds attribute).
        if self.time_limit:
            self.model.max_seconds = self.time_limit

# ---------------------------
# Main function.
# ---------------------------
def main(argv):
    print(argv)
    input_file = None      # e.g., "adult.csv"
    depth = None           # maximum tree depth (integer)
    time_limit = None      # time limit in seconds (integer)
    _lambda = None         # lambda, used as a regularization parameter (float)
    input_sample = None    # sample index to choose a random seed (integer)
    calibration = None     # 1: use calibration split; 0: do not (integer)
    mode = "classification"

    # Use one of several predefined random seeds based on input_sample.
    random_states_list = [41, 23, 45, 36, 19, 123]

    try:
        opts, args = getopt.getopt(argv, "f:d:t:l:i:c:m:",
                                   ["input_file=", "depth=", "timelimit=", "lambda=", "input_sample=", "calibration=", "mode="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--input_file"):
            input_file = arg
        elif opt in ("-d", "--depth"):
            depth = int(arg)
        elif opt in ("-t", "--timelimit"):
            time_limit = int(arg)
        elif opt in ("-l", "--lambda"):
            _lambda = float(arg)
        elif opt in ("-i", "--input_sample"):
            input_sample = int(arg)
        elif opt in ("-c", "--calibration"):
            calibration = int(arg)
        elif opt in ("-m", "--mode"):
            mode = arg

    start_time = time.time()
    # Assumes that the DataSets folder is located two levels up.
    data_path = os.getcwd() + '/../../DataSets/'
    data = pd.read_csv(data_path + input_file)
    # The class label column (here, "income")
    label = "income"
    # Create a tree object using the specified depth.
    tree = Tree(depth)

    ##########################################################
    # Output Setup
    ##########################################################
    approach_name = "FlowOCT"
    out_put_name = (f"{input_file}_{input_sample}_{approach_name}_d_{depth}_t_{time_limit}_lambda_{_lambda}_c_{calibration}")
    out_put_path = os.getcwd() + '/../../Results/'
    # Redirect output:
    sys.stdout = Logger(out_put_path + out_put_name + ".txt")

    ##########################################################
    # Data Splitting
    ##########################################################
    # Split data: 75% train, 25% test
    data_train, data_test = train_test_split(
        data, test_size=0.25, random_state=random_states_list[input_sample - 1])
    # Further split train for calibration if needed.
    data_train_calibration, data_calibration = train_test_split(
        data_train, test_size=0.33, random_state=random_states_list[input_sample - 1])
    if calibration == 1:
        data_train = data_train_calibration
    train_len = len(data_train.index)

    ##########################################################
    # Create and Solve the MIP Problem
    ##########################################################
    primal = FlowOCT(data_train, label, tree, _lambda, time_limit, mode)
    primal.create_primal_problem()
    primal.model.optimize()
    end_time = time.time()
    solving_time = end_time - start_time

    ##########################################################
    # Retrieve Solution
    ##########################################################
    b_value = [var.x for var in primal.b]
    beta_value = [var.x for var in primal.beta]
    p_value = [var.x for var in primal.p]

    print("\n\n")
    print_tree(primal, b_value, beta_value, p_value)
    print("\n\nTotal Solving Time", solving_time)
    print("obj value", primal.model.objective_value)
    print("Total Callback counter (Integer)", 0)
    print("Total Successful Callback counter (Integer)", 0)
    print("Total Callback Time (Integer)", 0)
    print("Total Successful Callback Time (Integer)", 0)

    ##########################################################
    # Evaluation
    ##########################################################
    train_acc = test_acc = calibration_acc = 0
    train_mae = test_mae = calibration_mae = 0
    train_r_squared = test_r_squared = calibration_r_squared = 0

    if mode == "classification":
        train_acc = get_acc(primal, data_train, b_value, beta_value, p_value)
        test_acc = get_acc(primal, data_test, b_value, beta_value, p_value)
        calibration_acc = get_acc(primal, data_calibration, b_value, beta_value, p_value)
    elif mode == "regression":
        train_mae = get_mae(primal, data_train, b_value, beta_value, p_value)
        test_mae = get_mae(primal, data_test, b_value, beta_value, p_value)
        calibration_mae = get_mae(primal, data_calibration, b_value, beta_value, p_value)
        train_mse = get_mse(primal, data_train, b_value, beta_value, p_value)
        test_mse = get_mse(primal, data_test, b_value, beta_value, p_value)
        calibration_mse = get_mse(primal, data_calibration, b_value, beta_value, p_value)
        train_r_squared = get_r_squared(primal, data_train, b_value, beta_value, p_value)
        test_r_squared = get_r_squared(primal, data_test, b_value, beta_value, p_value)
        calibration_r_squared = get_r_squared(primal, data_calibration, b_value, beta_value, p_value)

    print("obj value", primal.model.objective_value)
    if mode == "classification":
        print("train acc", train_acc)
        print("test acc", test_acc)
        print("calibration acc", calibration_acc)
    elif mode == "regression":
        print("train mae", train_mae)
        print("train mse", train_mse)
        print("train r^2", train_r_squared)

    ##########################################################
    # Write Output Files
    ##########################################################
    primal.model.write(out_put_path + out_put_name + ".lp")
    result_file = out_put_name + ".csv"
    with open(out_put_path + result_file, mode="a") as results:
        results_writer = csv.writer(results, delimiter=",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        if mode == "classification":
            results_writer.writerow([
                approach_name, input_file, train_len, depth, _lambda, time_limit,
                primal.model.status, primal.model.objective_value, train_acc,
                (primal.model.gap * 100) if primal.model.gap is not None else None,
                primal.model.num_solutions, solving_time,
                0, 0, 0, 0, test_acc, calibration_acc, input_sample
            ])
        elif mode == "regression":
            results_writer.writerow([
                approach_name, input_file, train_len, depth, _lambda, time_limit,
                primal.model.status, primal.model.objective_value, train_mae, test_mae, train_r_squared,
                (primal.model.gap * 100) if primal.model.gap is not None else None,
                primal.model.num_solutions, solving_time,
                0, 0, 0, 0, test_mae, calibration_mae,
                test_mse, calibration_mse,
                test_r_squared, calibration_r_squared,
                input_sample
            ])

if __name__ == "__main__":
    main(sys.argv[1:])
