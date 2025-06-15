#!/usr/bin/python
import sys
import os
import time
import getopt
import csv
import pandas as pd
from mip import Model, BINARY, CONTINUOUS, xsum, maximize
from sklearn.model_selection import train_test_split
from utils import *            # your utility routines
from logger import logger      # your custom logger that redirects output
from Tree import Tree          # your Tree class

#####################################################################
# FlowOCT Class: Formulate the decision-tree MIP problem.
# (This formulation “mimics” your original approach but now uses python-mip.)
#####################################################################
class FlowOCT:
    def __init__(self, data, label, tree, _lambda, time_limit, mode):
        """
        :param data: training DataFrame.
        :param label: column name for the class label.
        :param tree: Tree object (defines tree structure, Nodes, Leaves, etc.).
        :param _lambda: regularization parameter.
        :param time_limit: time limit (in seconds) for solving.
        :param mode: "classification" or "regression".
        """
        self.mode = mode
        self.data = data
        self.datapoints = list(data.index)
        self.label = label

        if self.mode == "classification":
            self.labels = list(data[label].unique())
        elif self.mode == "regression":
            self.labels = [1]
        # All features except the label are treated as categorical.
        self.cat_features = [col for col in data.columns if col != label]

        self.tree = tree
        self._lambda = _lambda

        # Parameter m: for each data point i, set m[i]=1 for classification.
        # (For regression, m[i] is chosen based on the target value.)
        self.m = { i: 1 for i in self.datapoints }
        if self.mode == "regression":
            for i in self.datapoints:
                y_i = data.at[i, label]
                self.m[i] = max(y_i, 1 - y_i)

        # Placeholders for decision variables:
        #   b[n,f] : binary branch decision at node n on feature f.
        #   p[n]   : binary variable indicating if node n is a prediction (leaf) node.
        #   beta[n,k] : continuous prediction parameter at node n for class k (or for regression, node n's value).
        #   zeta[i,n] : continuous variable representing outgoing "flow" from node n for data point i.
        #   z[i,n] : continuous variable representing incoming flow at node n for data point i.
        self.b = {}       # Indexed by (n, f)
        self.p = {}       # Indexed by node n (for both self.tree.Nodes and self.tree.Leaves)
        self.beta = {}    # Indexed by (n, k)
        self.zeta = {}    # Indexed by (i, n)
        self.z = {}       # Indexed by (i, n)

        # Create the model using python-mip with the CBC solver.
        self.model = Model("FlowOCT", sense=maximize, solver_name="cbc")
        if time_limit:
            self.model.max_seconds = time_limit

        # (Advanced callback tracking variables; these are placeholders to mimic your original code.)
        self.model._total_callback_time_integer = 0
        self.model._total_callback_time_integer_success = 0
        self.model._callback_counter_integer = 0
        self.model._callback_counter_integer_success = 0

    def create_primal_problem(self):
        """Create the MIP formulation for the FlowOCT problem."""
        # --- Decision Variables ---
        # b[n,f]: branching decisions, binary.
        for n in self.tree.Nodes:
            for f in self.cat_features:
                self.b[(n, f)] = self.model.add_var(var_type=BINARY, name=f"b_{n}_{f}")

        # p[n]: prediction indicator, binary, defined on both Nodes and Leaves.
        for n in (self.tree.Nodes + self.tree.Leaves):
            self.p[n] = self.model.add_var(var_type=BINARY, name=f"p_{n}")

        # beta[n,k]: prediction parameters (continuous, lower bound 0).
        for n in (self.tree.Nodes + self.tree.Leaves):
            for k in self.labels:
                self.beta[(n, k)] = self.model.add_var(var_type=CONTINUOUS, lb=0, name=f"beta_{n}_{k}")

        # Flow variables: For every data point and every node:
        for i in self.datapoints:
            for n in (self.tree.Nodes + self.tree.Leaves):
                self.zeta[(i, n)] = self.model.add_var(var_type=CONTINUOUS, lb=0, name=f"zeta_{i}_{n}")
                self.z[(i, n)] = self.model.add_var(var_type=CONTINUOUS, lb=0, name=f"z_{i}_{n}")

        # --- Constraints ---
        # Flow conservation at each internal node: 
        # For each node n in Nodes and each data point i:
        for n in self.tree.Nodes:
            # Get left and right children node indices; assume tree provides these methods.
            n_left = int(self.tree.get_left_children(n))
            n_right = int(self.tree.get_right_children(n))
            for i in self.datapoints:
                self.model.add_constr(self.z[(i, n)] == self.z[(i, n_left)] + self.z[(i, n_right)] + self.zeta[(i, n)],
                                      name=f"flow_cons_{i}_{n}")

        # Left branch flow constraints: For every data point i and node n,
        # z[i, n_left] <= m[i] * sum( b[n, f] for f if data[i, f]==0 )
        for n in self.tree.Nodes:
            n_left = int(self.tree.get_left_children(n))
            for i in self.datapoints:
                expr = xsum(self.b[(n, f)] for f in self.cat_features if self.data.at[i, f] == 0)
                self.model.add_constr(self.z[(i, n_left)] <= self.m[i] * expr, name=f"left_flow_{i}_{n}")

        # Right branch flow constraints:
        for n in self.tree.Nodes:
            n_right = int(self.tree.get_right_children(n))
            for i in self.datapoints:
                expr = xsum(self.b[(n, f)] for f in self.cat_features if self.data.at[i, f] == 1)
                self.model.add_constr(self.z[(i, n_right)] <= self.m[i] * expr, name=f"right_flow_{i}_{n}")

        # Node decision partition constraints for internal nodes:
        # sum_{f} b[n,f] + p[n] + sum_{m in A(n)} p[m] = 1, forall n in Nodes.
        for n in self.tree.Nodes:
            anc = self.tree.get_ancestors(n)
            self.model.add_constr(xsum(self.b[(n, f)] for f in self.cat_features) + self.p[n] + xsum(self.p[m] for m in anc) == 1,
                                  name=f"node_part_{n}")
        # For leaves: p[n] + sum_{m in A(n)} p[m] = 1
        for n in self.tree.Leaves:
            anc = self.tree.get_ancestors(n)
            self.model.add_constr(self.p[n] + xsum(self.p[m] for m in anc) == 1, name=f"leaf_part_{n}")

        # Loss reduction constraints:
        if self.mode == "classification":
            # For each data point i and each node n, enforce:
            # zeta[i, n] <= beta[n, y[i]]  where y[i] is the true class of datapoint i.
            for n in (self.tree.Nodes + self.tree.Leaves):
                for i in self.datapoints:
                    y_val = self.data.at[i, self.label]
                    self.model.add_constr(self.zeta[(i, n)] <= self.beta[(n, y_val)], name=f"loss_red_{i}_{n}")
            # For each node, sum_{k in labels} beta[n,k] = p[n]
            for n in (self.tree.Nodes + self.tree.Leaves):
                self.model.add_constr(xsum(self.beta[(n, k)] for k in self.labels) == self.p[n], name=f"beta_sum_{n}")
        elif self.mode == "regression":
            # For regression, add constraints adapted to prediction error.
            for n in (self.tree.Nodes + self.tree.Leaves):
                for i in self.datapoints:
                    self.model.add_constr(self.zeta[(i, n)] <= self.m[i]*self.p[n] - self.data.at[i, self.label]*self.p[n] + self.beta[(n, 1)],
                                          name=f"reg_loss1_{i}_{n}")
                    self.model.add_constr(self.zeta[(i, n)] <= self.m[i]*self.p[n] + self.data.at[i, self.label]*self.p[n] - self.beta[(n, 1)],
                                          name=f"reg_loss2_{i}_{n}")
            for n in (self.tree.Nodes + self.tree.Leaves):
                self.model.add_constr(self.beta[(n, 1)] <= self.p[n], name=f"reg_beta_{n}")

        # For each leaf node n and each data point i, enforce zeta[i, n] == z[i, n]:
        for n in self.tree.Leaves:
            for i in self.datapoints:
                self.model.add_constr(self.zeta[(i, n)] == self.z[(i, n)], name=f"leaf_flow_{i}_{n}")

        # --- Objective Function ---
        # Our objective is a linear expression:
        #   maximize: sum_{i in datapoints} (1 - lambda) * (z[i, 1] - m[i])
        #             - lambda * sum_{n in Nodes} sum_{f in cat_features} b[n,f]
        # Here we assume node "1" is the root.
        obj_expr = 0
        for i in self.datapoints:
            obj_expr += (1 - self._lambda) * (self.z[(i, 1)] - self.m[i])
        for n in self.tree.Nodes:
            for f in self.cat_features:
                obj_expr += - self._lambda * self.b[(n, f)]
        self.model.objective = maximize(obj_expr)
        # End of create_primal_problem()

#####################################################################
# Main function: parses command-line arguments, splits data,
# builds and solves the FlowOCT MIP, evaluates and writes output.
#####################################################################
def main(argv):
    print(argv)
    input_file = None      # e.g., "adult.csv"
    depth = None           # maximum tree depth (integer)
    time_limit = None      # time limit (seconds)
    _lambda = None         # regularization parameter (float)
    input_sample = None    # sample index (to choose a random seed)
    calibration = None     # calibration flag (1 for calibration, 0 otherwise)
    mode = "classification"

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
    data_path = os.getcwd() + '/../../DataSets/'
    data = pd.read_csv(data_path + input_file)
    label = 'income'
    tree = Tree(depth)

    approach_name = "FlowOCT"
    out_put_name = f"{input_file}_{input_sample}_{approach_name}_d_{depth}_t_{time_limit}_lambda_{_lambda}_c_{calibration}"
    out_put_path = os.getcwd() + '/../../Results/'
    sys.stdout = logger(out_put_path + out_put_name + ".txt")

    # Data splitting: 75% train, 25% test; and further calibration split.
    data_train, data_test = train_test_split(data, test_size=0.25, random_state=random_states_list[input_sample - 1])
    data_train_calibration, data_calibration = train_test_split(data_train, test_size=0.33, random_state=random_states_list[input_sample - 1])
    if calibration == 1:
        data_train = data_train_calibration
    train_len = len(data_train.index)

    # Create and solve the MIP problem.
    primal = FlowOCT(data_train, label, tree, _lambda, time_limit, mode)
    primal.create_primal_problem()
    # In python-mip, update is implicit; we call optimize().
    primal.model.optimize()
    end_time = time.time()
    solving_time = end_time - start_time

    # Retrieve solution: extract x values from b, beta, and p.
    b_value = { k: v.x for k, v in primal.b.items() }
    beta_value = { k: v.x for k, v in primal.beta.items() }
    p_value = { n: v.x for n, v in primal.p.items() }

    print("\n\n")
    print_tree(primal, b_value, beta_value, p_value)
    print("\n\nTotal Solving Time", solving_time)
    print("obj value", primal.model.objective_value)
    print('Total Callback counter (Integer)', primal.model._callback_counter_integer)
    print('Total Successful Callback counter (Integer)', primal.model._callback_counter_integer_success)
    print('Total Callback Time (Integer)', primal.model._total_callback_time_integer)
    print('Total Successful Callback Time (Integer)', primal.model._total_callback_time_integer_success)

    # --- Evaluation ---
    # Dummy evaluation functions (replace with advanced ones later).
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

    # Write model file and results.
    primal.model.write(out_put_path + out_put_name + ".lp")
    result_file = out_put_name + ".csv"
    with open(out_put_path + result_file, mode="a", newline="") as results:
        results_writer = csv.writer(results, delimiter=",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        if mode == "classification":
            results_writer.writerow([
                approach_name, input_file, train_len, depth, _lambda, time_limit,
                primal.model.status, primal.model.objective_value, train_acc,
                (primal.model.gap * 100) if primal.model.gap is not None else None,
                primal.model.num_solutions, solving_time,
                primal.model._total_callback_time_integer, primal.model._total_callback_time_integer_success,
                primal.model._callback_counter_integer, primal.model._callback_counter_integer_success,
                test_acc, calibration_acc, input_sample
            ])
        elif mode == "regression":
            results_writer.writerow([
                approach_name, input_file, train_len, depth, _lambda, time_limit,
                primal.model.status, primal.model.objective_value, train_mae, test_mae, train_r_squared,
                (primal.model.gap * 100) if primal.model.gap is not None else None,
                primal.model.num_solutions, solving_time,
                primal.model._total_callback_time_integer, primal.model._total_callback_time_integer_success,
                primal.model._callback_counter_integer, primal.model._callback_counter_integer_success,
                test_mae, calibration_mae,
                test_mse, calibration_mse,
                test_r_squared, calibration_r_squared,
                input_sample
            ])

if __name__ == "__main__":
    main(sys.argv[1:])
