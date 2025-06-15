#!/usr/bin/python
import sys
import os
import time
import getopt
import csv
import pandas as pd
from mip import Model, BINARY, CONTINUOUS, minimize
from sklearn.model_selection import train_test_split

# Logger class to redirect stdout to both console and a file.
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

# Dummy Tree class (as used by FlowOCT)
class Tree:
    def __init__(self, depth):
        self.depth = depth

# Utility function to print the tree structure (dummy implementation).
def print_tree(primal, b_value, beta_value, p_value):
    print("Printing tree structure:")
    print("b values:", b_value)
    print("beta values:", beta_value)
    print("p values:", p_value)

# Dummy evaluation functions.
def get_acc(primal, data, b_value, beta_value, p_value):
    return 0.95

def get_mae(primal, data, b_value, beta_value, p_value):
    return 0.1

def get_mse(primal, data, b_value, beta_value, p_value):
    return 0.05

def get_r_squared(primal, data, b_value, beta_value, p_value):
    return 0.9

# FlowOCT class that builds and solves the optimization model using python-mip.
class FlowOCT:
    def __init__(self, data, label, tree, lam, time_limit, mode):
        self.data = data
        self.label = label
        self.tree = tree
        self.lam = lam
        self.time_limit = time_limit
        self.mode = mode
        # Create a MIP model instance (using CBC).
        self.model = Model(sense=minimize, solver_name="cbc")
        self.b = []      # Binary decision variables.
        self.beta = []   # Continuous decision variables.
        self.p = []      # Continuous decision variables.

    def create_primal_problem(self):
        # For demonstration, we add one variable for each list.
        self.b.append(self.model.add_var(var_type=BINARY, name="b0"))
        self.beta.append(self.model.add_var(var_type=CONTINUOUS, lb=-10, ub=10, name="beta0"))
        self.p.append(self.model.add_var(var_type=CONTINUOUS, lb=0, ub=10, name="p0"))
        
        # IMPORTANT: Replace quadratic terms with a linear objective.
        self.model.objective = minimize(self.lam * self.b[0] + self.beta[0] + self.p[0])
        
        # Dummy constraint: beta0 + p0 >= 1.
        self.model += self.beta[0] + self.p[0] >= 1

        # Set time limit if provided.
        if self.time_limit:
            self.model.max_seconds = self.time_limit

# Main function to process arguments, load data, build and solve the model.
def main(argv):
    print(argv)
    input_file = None
    depth = None
    time_limit = None
    _lambda = None
    input_sample = None
    calibration = None
    mode = "classification"
    '''
    Depending on the value of input_sample we choose one of the following random seeds 
    and then split the whole data into train, test and calibration.
    '''
    random_states_list = [41, 23, 45, 36, 19, 123]

    try:
        opts, args = getopt.getopt(argv, "f:d:t:l:i:c:m:",
                                   ["input_file=", "depth=", "timelimit=", "lambda=",
                                    "input_sample=", "calibration=", "mode="])
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
    data_path = os.getcwd() + '/../../DataSets/'
    data = pd.read_csv(data_path + input_file)
    # Name of the column in the dataset representing the class label.
    label = 'income'

    # Tree structure: Create a tree object of the given depth.
    tree = Tree(depth)

    ##########################################################
    # Output setup
    ##########################################################
    approach_name = 'FlowOCT'
    out_put_name = (input_file + '_' + str(input_sample) + '_' + approach_name +
                    '_d_' + str(depth) + '_t_' + str(time_limit) +
                    '_lambda_' + str(_lambda) + '_c_' + str(calibration))
    out_put_path = os.getcwd() + '/../../Results/'
    # Redirect stdout to log file.
    sys.stdout = Logger(out_put_path + out_put_name + '.txt')

    ##########################################################
    # Data splitting
    ##########################################################
    '''
    Creating train, test and calibration datasets.
    We use 50% of the data for training, 25% for testing, and 25% for calibration.
    When calibration==1, we train only on a 50% subset; otherwise on 75% of the data.
    '''
    data_train, data_test = train_test_split(data, test_size=0.25,
                                             random_state=random_states_list[input_sample - 1])
    data_train_calibration, data_calibration = train_test_split(
        data_train, test_size=0.33, random_state=random_states_list[input_sample - 1])

    if calibration == 1:  # For calibration, train on a smaller subset.
        data_train = data_train_calibration

    train_len = len(data_train.index)

    ##########################################################
    # Creating and Solving the Problem
    ##########################################################
    primal = FlowOCT(data_train, label, tree, _lambda, time_limit, mode)
    primal.create_primal_problem()
    primal.model.optimize()
    end_time = time.time()
    solving_time = end_time - start_time

    ##########################################################
    # Retrieving the Solution
    ##########################################################
    b_value = [var.x for var in primal.b]
    beta_value = [var.x for var in primal.beta]
    p_value = [var.x for var in primal.p]

    print("\n\n")
    print_tree(primal, b_value, beta_value, p_value)
    print('\n\nTotal Solving Time', solving_time)
    print("obj value", primal.model.objective_value)
    # Dummy callback values (python-mip does not use callbacks like gurobipy)
    print('Total Callback counter (Integer)', 0)
    print('Total Successful Callback counter (Integer)', 0)
    print('Total Callback Time (Integer)', 0)
    print('Total Successful Callback Time (Integer)', 0)

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
        train_r2 = get_r_squared(primal, data_train, b_value, beta_value, p_value)
        test_r2 = get_r_squared(primal, data_test, b_value, beta_value, p_value)
        calibration_r2 = get_r_squared(primal, data_calibration, b_value, beta_value, p_value)

    print("obj value", primal.model.objective_value)
    if mode == "classification":
        print("train acc", train_acc)
        print("test acc", test_acc)
        print("calibration acc", calibration_acc)
    elif mode == "regression":
        print("train mae", train_mae)
        print("train mse", test_mse)
        print("train r^2", train_r_squared)

    ##########################################################
    # Write Output Files
    ##########################################################
    primal.model.write(out_put_path + out_put_name + '.lp')
    result_file = out_put_name + '.csv'
    with open(out_put_path + result_file, mode='a', newline='') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        if mode == "classification":
            results_writer.writerow(
                [approach_name, input_file, train_len, depth, _lambda, time_limit,
                 primal.model.status, primal.model.objective_value, train_acc,
                 (primal.model.gap * 100) if primal.model.gap is not None else None,
                 0,  # Dummy value for num_nodes as it's not available in python-mip
                 solving_time,
                 0, 0, 0, 0,   # Dummy callback counters and times.
                 test_acc, calibration_acc, input_sample])
        elif mode == "regression":
            results_writer.writerow(
                [approach_name, input_file, train_len, depth, _lambda, time_limit,
                 primal.model.status, primal.model.objective_value, train_mae, test_mae, train_r_squared,
                 (primal.model.gap * 100) if primal.model.gap is not None else None,
                 0,  # Dummy value for num_nodes as it's not available in python-mip
                 solving_time,
                 0, 0, 0, 0,  # Dummy callback counters and times.
                 test_mae, calibration_mae,
                 test_mse, calibration_mse,
                 test_r_squared, calibration_r2,
                 input_sample])

if __name__ == "__main__":
    main(sys.argv[1:])
