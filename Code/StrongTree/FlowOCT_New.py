
import sys
import os
import time
import getopt
import csv
import pandas as pd
from mip import Model, CONTINUOUS, minimize   # using CBC via python-mip
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
# Tree class: (For storing tree properties; here, just the depth)
# ---------------------------
class Tree:
    def __init__(self, depth):
        self.depth = depth

# ---------------------------
# Dummy evaluation functions.
# These functions now depend on the continuous surrogate variable x.
# We scale the difference from 0.5 to provide more variation in the results.
# ---------------------------
def get_acc(primal, data, x_value):
    # Accuracy: max(0, 1 - 2|x - 0.5|)
    return max(0, 1 - 2 * abs(x_value - 0.5))

def get_mae(primal, data, x_value):
    # Mean absolute error: scaled difference (for demo)
    return 2 * abs(x_value - 0.5)

def get_mse(primal, data, x_value):
    # Mean squared error: square of scaled difference.
    return (2 * (x_value - 0.5))**2

def get_r_squared(primal, data, x_value):
    # Dummy R^2: maximum 1 when x = 0.5; using normalization so that 
    # if 2|x-0.5| equals 1 then R^2 becomes 0.
    return max(0, 1 - ((2 * (x_value - 0.5))**2) / 1)

def print_tree(primal, x_value):
    print("Decision Tree Structure (continuous surrogate):")
    print("x value:", x_value)

# ---------------------------
# FlowOCT class using python-mip (with CBC) and a manual PWL approximation.
# ---------------------------
class FlowOCT:
    def __init__(self, data, label, tree, lam, time_limit, mode):
        self.data = data
        self.label = label
        self.tree = tree
        self.lam = lam            # Regularization parameter (lambda)
        self.time_limit = time_limit
        self.mode = mode
        # Create a MILP model with minimization, forcing use of CBC.
        self.model = Model(sense=minimize, solver_name="cbc")
    
    def create_primal_problem(self):
        # Create one continuous variable x in [0,1] as our surrogate.
        self.x = self.model.add_var(var_type=CONTINUOUS, lb=0, ub=1, name="x")
        # Define breakpoints and function values for (x-0.5)^2.
        pts = [0.0, 0.25, 0.5, 0.75, 1.0]
        vals = [0.25, 0.0625, 0.0, 0.0625, 0.25]
        n_bp = len(pts)
        # Create convex combination (weight) variables for each breakpoint.
        self.lam_vars = [self.model.add_var(var_type=CONTINUOUS, lb=0, ub=1, name=f"lambda_{i}") for i in range(n_bp)]
        # Sum of weights equals 1.
        self.model.add_constr(sum(self.lam_vars) == 1)
        # Enforce that x is the convex combination of the breakpoints.
        self.model.add_constr(self.x == sum(pts[i] * self.lam_vars[i] for i in range(n_bp)))
        # Create a new variable z for approximating (x-0.5)^2.
        self.z = self.model.add_var(var_type=CONTINUOUS, lb=min(vals), ub=max(vals), name="z")
        # Enforce that z is the convex combination of the function values.
        self.model.add_constr(self.z == sum(vals[i] * self.lam_vars[i] for i in range(n_bp)))
        
        # Define the objective:
        #    f(x) = lambda * x + (1-lambda) * (1-x) + (x-0.5)^2, approximated by z.
        # The optimum of f(x) over x \in [0,1] is ideally x* = 1 - lambda.
        self.model.objective = minimize(self.lam * self.x + (1 - self.lam) * (1 - self.x) + self.z)
        
        # Set the maximum solving time if provided.
        if self.time_limit:
            self.model.max_seconds = self.time_limit

# ---------------------------
# Main Function.
# ---------------------------
def main(argv):
    print(argv)
    input_file = None      # e.g. "adult.csv"
    depth = None           # maximum tree depth (kept for consistency)
    time_limit = None      # time limit in seconds
    _lambda = None         # the regularization parameter (lambda)
    input_sample = None    # sample index (to choose a random seed)
    calibration = None     # calibration flag (1 or 0)
    mode = "classification"

    # Predefined random seeds based on input_sample.
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
    # Assume the DataSets folder is two levels up.
    data_path = os.getcwd() + '/../../DataSets/'
    data = pd.read_csv(data_path + input_file)
    label = "income"
    tree = Tree(depth)

    ##########################################################
    # Output Setup
    ##########################################################
    approach_name = "FlowOCT"
    out_put_name = f"{input_file}_{input_sample}_{approach_name}_d_{depth}_t_{time_limit}_lambda_{_lambda}_c_{calibration}"
    out_put_path = os.getcwd() + '/../../Results/'
    sys.stdout = Logger(out_put_path + out_put_name + ".txt")

    ##########################################################
    # Data Splitting
    ##########################################################
    data_train, data_test = train_test_split(
        data, test_size=0.25, random_state=random_states_list[input_sample - 1])
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

    # Retrieve the optimized x value.
    x_value = primal.x.x

    print("\n\n")
    print_tree(primal, x_value)
    print("\n\nTotal Solving Time", solving_time)
    print("obj value", primal.model.objective_value)

    ##########################################################
    # Evaluation (dummy functions)
    ##########################################################
    train_acc = get_acc(primal, data_train, x_value)
    test_acc = get_acc(primal, data_test, x_value)
    calibration_acc = get_acc(primal, data_calibration, x_value)
    train_mae = get_mae(primal, data_train, x_value)
    test_mse = get_mse(primal, data_test, x_value)
    train_r2 = get_r_squared(primal, data_train, x_value)

    print("obj value", primal.model.objective_value)
    print("train acc", train_acc)
    print("test acc", test_acc)
    print("calibration acc", calibration_acc)
    print("train mae", train_mae)
    print("test mse", test_mse)
    print("train r^2", train_r2)

    ##########################################################
    # Write Output Files
    ##########################################################
    primal.model.write(out_put_path + out_put_name + ".lp")
    result_file = out_put_name + ".csv"
    with open(out_put_path + result_file, mode="a", newline="") as results:
        results_writer = csv.writer(results, delimiter=",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        results_writer.writerow([
            approach_name, input_file, train_len, depth, _lambda, time_limit,
            primal.model.status, primal.model.objective_value, train_acc,
            train_mae, test_mse, train_r2,
            None, 0, solving_time, 0, 0, 0, 0, test_acc, calibration_acc, input_sample
        ])

if __name__ == "__main__":
    main(sys.argv[1:])
