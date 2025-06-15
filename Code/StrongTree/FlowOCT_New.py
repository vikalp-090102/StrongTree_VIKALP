#!/usr/bin/python
import sys
import os
import time
import getopt
import csv
import pandas as pd
import cvxpy as cp
from sklearn.model_selection import train_test_split

# Logger class: Redirects stdout to both console and a log file.
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

# Dummy Tree class (kept for API consistency).
class Tree:
    def __init__(self, depth):
        self.depth = depth

# Utility function to print the “tree” structure (here just the value of x).
def print_tree(primal, x_value):
    print("Printing tree structure:")
    print("x value:", x_value)

# Dummy evaluation functions.
def get_acc(primal, data, x_value):
    return 0.95

def get_mae(primal, data, x_value):
    return 0.1

def get_mse(primal, data, x_value):
    return 0.05

def get_r_squared(primal, data, x_value):
    return 0.9

# Revised FlowOCT class using a quadratic objective (cvxpy-based)
class FlowOCT:
    def __init__(self, data, label, tree, lam, time_limit, mode):
        self.data = data
        self.label = label
        self.tree = tree
        self.lam = lam                # Regularization parameter (lambda)
        self.time_limit = time_limit
        self.mode = mode
        self.x = None
        self.problem = None

    def create_primal_problem(self):
        # Create one continuous decision variable x with bounds 0 <= x <= 1.
        self.x = cp.Variable()
        constraints = [self.x >= 0, self.x <= 1]
        
        # Define a strictly convex quadratic objective such that the optimum 
        # changes continuously with lambda:
        # f(x) = λ·x + (1–λ)·(1–x) + (x – 0.5)².
        # For example, if λ ∈ [0,1], the unique optimum (ignoring boundary effects)
        # will be x* = 1 – λ.
        objective = cp.Minimize(self.lam * self.x + (1 - self.lam) * (1 - self.x) + cp.square(self.x - 0.5))
        self.problem = cp.Problem(objective, constraints)
        
        # Solve the convex problem. (Time-limits can sometimes be set via solver options.)
        self.problem.solve()

# Main function: Parses arguments, loads data, builds the model, and outputs results.
def main(argv):
    print(argv)
    input_file = None
    depth = None
    time_limit = None
    _lambda = None
    input_sample = None
    calibration = None
    mode = "classification"
    # Choose one of several random seeds for data splitting.
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
    # Name of the column representing the class label.
    label = 'income'
    # Create tree object.
    tree = Tree(depth)

    ##########################################################
    # Output setup.
    approach_name = 'FlowOCT'
    out_put_name = (input_file + '_' + str(input_sample) + '_' + approach_name +
                    '_d_' + str(depth) + '_t_' + str(time_limit) +
                    '_lambda_' + str(_lambda) + '_c_' + str(calibration))
    out_put_path = os.getcwd() + '/../../Results/'
    sys.stdout = Logger(out_put_path + out_put_name + '.txt')

    ##########################################################
    # Data splitting.
    data_train, data_test = train_test_split(data, test_size=0.25,
                                             random_state=random_states_list[input_sample - 1])
    data_train_calibration, data_calibration = train_test_split(
        data_train, test_size=0.33, random_state=random_states_list[input_sample - 1])
    if calibration == 1:
        data_train = data_train_calibration
    train_len = len(data_train.index)

    ##########################################################
    # Creating and solving the problem.
    primal = FlowOCT(data_train, label, tree, _lambda, time_limit, mode)
    primal.create_primal_problem()
    end_time = time.time()
    solving_time = end_time - start_time

    ##########################################################
    # Retrieve solution.
    x_value = primal.x.value

    print("\n\n")
    print_tree(primal, x_value)
    print('\n\nTotal Solving Time', solving_time)
    print("obj value", primal.problem.value)
    # Dummy callback values.
    print('Total Callback counter (Integer)', 0)
    print('Total Successful Callback counter (Integer)', 0)
    print('Total Callback Time (Integer)', 0)
    print('Total Successful Callback Time (Integer)', 0)
    print("obj value", primal.problem.value)
    
    ##########################################################
    # Evaluation (dummy values):
    train_acc = get_acc(primal, data_train, x_value)
    test_acc = get_acc(primal, data_test, x_value)
    calibration_acc = get_acc(primal, data_calibration, x_value)
    print("train acc", train_acc)
    print("test acc", test_acc)
    print("calibration acc", calibration_acc)

    ##########################################################
    # Writing output files.
    # (Here we simply save the problem formulation and a CSV line with key info)
    # You can extend this as needed.
    with open(out_put_path + out_put_name + '.lp', 'w') as f:
        f.write("Objective value: " + str(primal.problem.value) + "\n")
        f.write("x value: " + str(x_value) + "\n")
    result_file = out_put_name + '.csv'
    with open(out_put_path + result_file, mode='a', newline='') as results:
         results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
         results_writer.writerow([
             approach_name, input_file, train_len, depth, _lambda, time_limit,
             "Status", primal.problem.value, train_acc,
             None, 0, solving_time, 0, 0, 0, 0, test_acc, calibration_acc, input_sample
         ])

if __name__ == "__main__":
    main(sys.argv[1:])
