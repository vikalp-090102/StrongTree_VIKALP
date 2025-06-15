#!/usr/bin/python
import sys
import os
import time
import getopt
import csv
import pandas as pd
import cvxpy as cp
import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------
# Logger: Redirects stdout to file and console.
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
# Dummy Tree class for API consistency.
# ---------------------------
class Tree:
    def __init__(self, depth):
        self.depth = depth

# ---------------------------
# Data Preprocessing Functions.
# ---------------------------
def preprocess_data(data):
    """
    Process the raw Adult dataset into a feature matrix X.
    Numeric features are min–max scaled and categorical features are one–hot encoded.
    """
    # Define numeric columns.
    numeric_cols = ["age", "fnlwgt", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]
    # All other columns (except 'income') are considered categorical.
    categorical_cols = [col for col in data.columns if col not in numeric_cols + ["income"]]
    
    data_numeric = data[numeric_cols].copy()
    for col in numeric_cols:
        data_numeric[col] = (data_numeric[col] - data_numeric[col].min()) / \
                            (data_numeric[col].max() - data_numeric[col].min())
    
    data_categorical = pd.get_dummies(data[categorical_cols], drop_first=True)
    X = pd.concat([data_numeric, data_categorical], axis=1)
    return X

def get_true_labels_lr(data):
    """
    Convert the income column to a binary label for logistic regression.
    Returns 1 if income >50K, and -1 otherwise.
    """
    def convert(val):
        if isinstance(val, str):
            return 1 if ">50K" in val else -1
        return 1 if val == 1 or val == "1" else -1
    y = data["income"].apply(convert)
    return y.values  # as a NumPy array

# ---------------------------
# Logistic Regression Helper Functions.
# ---------------------------
def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))

def predict_logreg(model, X):
    """
    Predict labels using the trained logistic regression model.
    Returns predictions in {-1, 1}.
    """
    z = X @ model.w.value + model.b.value
    preds = np.where(z >= 0, 1, -1)
    return preds

def get_accuracy(true, pred):
    return np.mean(true == pred)

def get_regression_metrics(model, X, y):
    """
    Compute regression-style metrics using the sigmoid probability output
    as a continuous prediction.
    True labels are mapped from {-1, 1} to {0, 1}.
    """
    y_reg = (y + 1) / 2.0
    z = X @ model.w.value + model.b.value
    prob = sigmoid(z)
    mae = np.mean(np.abs(prob - y_reg))
    mse = np.mean((prob - y_reg)**2)
    total_var = np.mean((y_reg - y_reg.mean())**2)
    r2 = 1 - mse/total_var if total_var != 0 else 0
    return mae, mse, r2

# ---------------------------
# Advanced Logistic Regression Class using cvxpy.
# ---------------------------
class FlowOCT_LogReg:
    def __init__(self, X, y, lam, time_limit, mode):
        self.X = X  # Feature matrix (NumPy array)
        self.y = y  # Labels in {-1, 1} (NumPy array)
        self.lam = lam            # Regularization parameter λ
        self.time_limit = time_limit  # (Not used in ECOS below)
        self.mode = mode
        self.n, self.d = self.X.shape
        self.w = None
        self.b = None
        self.problem = None
    
    def create_primal_problem(self):
        # Define decision variables: w (vector) and b (scalar)
        self.w = cp.Variable(self.d)
        self.b = cp.Variable()
        
        # Logistic loss: for each sample, loss = logistic(-y*(wᵀx+b)).
        losses = cp.logistic(-cp.multiply(self.y, self.X @ self.w + self.b))
        loss = cp.sum(losses) / self.n
        
        # Regularization: (λ/2)*||w||²
        reg = (self.lam / 2) * cp.sum_squares(self.w)
        objective = cp.Minimize(loss + reg)
        
        self.problem = cp.Problem(objective)
        # Use the ECOS solver (which does not require a time limit parameter).
        self.problem.solve(solver=cp.ECOS)

# ---------------------------
# Main Function.
# ---------------------------
def main(argv):
    print(argv)
    input_file = None
    depth = None
    time_limit = None
    _lambda = None
    input_sample = None
    calibration = None
    mode = "classification"

    # Random seed list for data splitting.
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
    
    # Preprocess features and obtain the label vector.
    X_full = preprocess_data(data)
    y_full = get_true_labels_lr(data)
    X_full = X_full.values  # Convert to NumPy array.
    
    # Split data into train, test, and calibration sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.25, random_state=random_states_list[input_sample - 1])
    X_train_cal, X_cal, y_train_cal, y_cal = train_test_split(
        X_train, y_train, test_size=0.33, random_state=random_states_list[input_sample - 1])
    if calibration == 1:
        X_train_use, y_train_use = X_train_cal, y_train_cal
    else:
        X_train_use, y_train_use = X_train, y_train
    train_len = X_train_use.shape[0]

    ##########################################################
    # Output setup.
    approach_name = 'FlowOCT_LogReg'
    out_put_name = (input_file + '_' + str(input_sample) + '_' + approach_name +
                    '_d_' + str(depth) + '_t_' + str(time_limit) +
                    '_lambda_' + str(_lambda) + '_c_' + str(calibration))
    out_put_path = os.getcwd() + '/../../Results/'
    sys.stdout = Logger(out_put_path + out_put_name + '.txt')

    ##########################################################
    # Create and solve the logistic regression problem.
    model = FlowOCT_LogReg(X_train_use, y_train_use, _lambda, time_limit, mode)
    model.create_primal_problem()
    end_time = time.time()
    solving_time = end_time - start_time

    # Retrieve model parameters.
    w_val, b_val = model.w.value, model.b.value

    print("\n\nPrinting model parameters:")
    print("w:", w_val)
    print("b:", b_val)
    print("\nTotal Solving Time:", solving_time)
    print("Objective value:", model.problem.value)
    print('Total Callback counter (Integer):', 0)
    print('Total Successful Callback counter (Integer):', 0)
    print('Total Callback Time (Integer):', 0)
    print('Total Successful Callback Time (Integer):', 0)
    
    ##########################################################
    # Evaluation: Predictions and metric computation.
    train_preds = predict_logreg(model, X_train_use)
    test_preds = predict_logreg(model, X_test)
    cal_preds = predict_logreg(model, X_cal)
    
    train_acc = get_accuracy(y_train_use, train_preds)
    test_acc = get_accuracy(y_test, test_preds)
    cal_acc = get_accuracy(y_cal, cal_preds)
    
    train_mae, train_mse, train_r2 = get_regression_metrics(model, X_train_use, y_train_use)
    test_mae, test_mse, test_r2 = get_regression_metrics(model, X_test, y_test)
    
    print("train acc:", train_acc)
    print("test acc:", test_acc)
    print("calibration acc:", cal_acc)
    print("train mae:", train_mae)
    print("test mse:", test_mse)
    print("train r^2:", train_r2)

    ##########################################################
    # Write output files.
    with open(out_put_path + out_put_name + '.lp', 'w') as f:
        f.write("Objective value: " + str(model.problem.value) + "\n")
        f.write("w: " + str(w_val) + "\n")
        f.write("b: " + str(b_val) + "\n")
    result_file = out_put_name + '.csv'
    with open(out_put_path + result_file, mode='a', newline='') as results:
         results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
         results_writer.writerow([
             approach_name, input_file, train_len, depth, _lambda, time_limit,
             "Status", model.problem.value, train_acc,
             train_mae, test_mse, train_r2,
             None, 0, solving_time, 0, 0, 0, 0, test_acc, cal_acc, input_sample
         ])

if __name__ == "__main__":
    main(sys.argv[1:])
