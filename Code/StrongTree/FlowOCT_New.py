#!/usr/bin/python
import sys
import os
import time
import getopt
import csv
import pandas as pd
import cvxpy as cp
from sklearn.model_selection import train_test_split
import numpy as np

# Logger class: Redirects stdout to both the console and a log file.
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

# Dummy Tree class (kept only for API consistency).
class Tree:
    def __init__(self, depth):
        self.depth = depth

# --- Updated Evaluation Functions Using x for Prediction ---
#
# Note: The adult dataset contains the following features:
#   age, workclass, fnlwgt, education, educational-num, marital-status, 
#   occupation, relationship, race, gender, capital-gain, capital-loss, 
#   hours-per-week, native-country, income
#
# For this demonstration, we will use only the "age" feature (which is continuous)
# to build a simple prediction rule. In practice, you would incorporate other features 
# (after proper encoding and scaling) into your predictive model.

def convert_income(val):
    """
    Convert the income value to a binary label.
    Assumes that if the string contains ">50K" or "1", the label is 1; otherwise 0.
    If the value is already numeric, it is assumed to be 0 or 1.
    """
    if isinstance(val, str):
        return 1 if ">50K" in val or "1" in val else 0
    return val

def get_predictions(x_value, data):
    """
    Use the decision variable x_value as a threshold to predict the income class.
    Here we:
      1. Normalize the numeric feature "age" to the interval [0, 1].
      2. Predict 1 if the normalized age is greater than x_value, and 0 otherwise.
    
    (For demonstration we only use "age." In practice you might combine it with other features.)
    """
    # Convert age to float and normalize.
    ages = data["age"].astype(float)
    norm_age = (ages - ages.min()) / (ages.max() - ages.min())
    preds = (norm_age > x_value).astype(int)
    return preds

def get_true_labels(data):
    """
    Extract true income labels from the "income" column
    and convert them to binary (0/1).
    """
    true_labels = data["income"].apply(convert_income)
    return true_labels

def get_acc(primal, data, x_value):
    preds = get_predictions(x_value, data)
    true_labels = get_true_labels(data)
    accuracy = (preds == true_labels).mean()
    return accuracy

def get_mae(primal, data, x_value):
    # For MAE, we use the predictions as probabilities (here we use normalized age as a proxy).
    ages = data["age"].astype(float)
    norm_age = (ages - ages.min()) / (ages.max() - ages.min())
    true_labels = get_true_labels(data)
    mae = np.abs(norm_age - true_labels).mean()
    return mae

def get_mse(primal, data, x_value):
    ages = data["age"].astype(float)
    norm_age = (ages - ages.min()) / (ages.max() - ages.min())
    true_labels = get_true_labels(data)
    mse = np.square(norm_age - true_labels).mean()
    return mse

def get_r_squared(primal, data, x_value):
    mse = get_mse(primal, data, x_value)
    true_labels = get_true_labels(data)
    mean_label = true_labels.mean()
    total_var = ((true_labels - mean_label)**2).mean()
    r2 = 1 - mse/total_var if total_var != 0 else 0
    return r2

# --- Revised FlowOCT Class with Continuous Decision Variable (via cvxpy) ---
#
# We now have a model where the decision variable x ∈ [0, 1] is chosen by solving:
#
#   minimize f(x) = λ·x + (1-λ)·(1-x) + (x - 0.5)²,
#
# a strictly convex quadratic objective that changes continuously with λ.
# Once x is determined, it is then used as a threshold for building predictions.

class FlowOCT:
    def __init__(self, data, label, tree, lam, time_limit, mode):
        self.data = data
        self.label = label
        self.tree = tree
        self.lam = lam          # Regularization parameter (lambda)
        self.time_limit = time_limit
        self.mode = mode
        self.x = None
        self.problem = None

    def create_primal_problem(self):
        # Decision variable x with bounds [0, 1].
        self.x = cp.Variable()
        constraints = [self.x >= 0, self.x <= 1]
        
        # Define a strictly convex quadratic objective:
        # f(x) = λ·x + (1-λ)·(1-x) + (x - 0.5)².
        objective = cp.Minimize(self.lam * self.x + (1 - self.lam) * (1 - self.x) + cp.square(self.x - 0.5))
        self.problem = cp.Problem(objective, constraints)
        
        # Solve the problem.
        self.problem.solve()

# --- Main function ---

def main(argv):
    print(argv)
    input_file = None
    depth = None
    time_limit = None
    _lambda = None
    input_sample = None
    calibration = None
    mode = "classification"

    # Choose a random seed for data splitting.
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
    # The income column is used to get true labels.
    label = 'income'
    # Create tree object (for API consistency).
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
    # Create and solve the problem.
    primal = FlowOCT(data_train, label, tree, _lambda, time_limit, mode)
    primal.create_primal_problem()
    end_time = time.time()
    solving_time = end_time - start_time

    ##########################################################
    # Retrieve the solution: the decision variable x.
    x_value = primal.x.value

    print("\n\n")
    print("Printing tree structure:")
    print("x value:", x_value)
    print("\n\nTotal Solving Time", solving_time)
    print("obj value", primal.problem.value)
    # Dummy callback values.
    print('Total Callback counter (Integer)', 0)
    print('Total Successful Callback counter (Integer)', 0)
    print('Total Callback Time (Integer)', 0)
    print('Total Successful Callback Time (Integer)', 0)
    print("obj value", primal.problem.value)
    
    ##########################################################
    # Evaluation: Compute predictions using x_value and calculate actual performance metrics.
    train_acc = get_acc(primal, data_train, x_value)
    test_acc = get_acc(primal, data_test, x_value)
    calibration_acc = get_acc(primal, data_calibration, x_value)
    
    train_mae = get_mae(primal, data_train, x_value)
    test_mse = get_mse(primal, data_test, x_value)
    train_r_squared = get_r_squared(primal, data_train, x_value)
    
    print("train acc", train_acc)
    print("test acc", test_acc)
    print("calibration acc", calibration_acc)
    print("train mae", train_mae)
    print("test mse", test_mse)
    print("train r^2", train_r_squared)

    ##########################################################
    # Write output files.
    with open(out_put_path + out_put_name + '.lp', 'w') as f:
        f.write("Objective value: " + str(primal.problem.value) + "\n")
        f.write("x value: " + str(x_value) + "\n")
    result_file = out_put_name + '.csv'
    with open(out_put_path + result_file, mode='a', newline='') as results:
         results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
         results_writer.writerow([
             approach_name, input_file, train_len, depth, _lambda, time_limit,
             "Status", primal.problem.value, train_acc,
             train_mae, test_mse, train_r_squared,
             None, 0, solving_time, 0, 0, 0, 0, test_acc, calibration_acc, input_sample
         ])

if __name__ == "__main__":
    main(sys.argv[1:])
