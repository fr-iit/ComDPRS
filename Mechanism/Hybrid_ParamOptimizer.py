from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import numpy as np
from Mechanism.Mapping import *
from Mechanism.Evaluation import *
import Perturbation_Mechanism as PM
import RecSys_ALS as als

# Define the parameter search space
search_space = [
    Real(0.001, 1.0, name='k'),
    Real(0.001, 2.0, name='m'),
    Real(0.001, 1.0, name='y')
]

def perturb_matrix(ep, train_matrix, sensitivity, lower, upper, index, k, m, y, L, Cp_Sen):
    """
    Applies Composite DP perturbation to each nonzero entry in the matrix.
    """
    R_perturbed = np.zeros_like(train_matrix)
    n_users, n_items = train_matrix.shape

    for i in range(n_users):
        # changes
        num_ratings = np.count_nonzero(train_matrix[i, :])
        user_ep = ep/num_ratings
        print('user: ', i, ' num_ratings: ', num_ratings, ' ep: ', user_ep)
        for j in range(n_items):
            if train_matrix[i, j] > 0:

                R_perturbed[i, j] = np.round(PM.perturbation_fun_optimized(
                    user_ep, train_matrix[i, j], sensitivity, lower, upper,index, k, m, y, L, Cp_Sen))


    return R_perturbed

def calculate_rmse(test_matrix, P, Q):

    # Check if test_matrix dimensions match P and Q
    num_users, num_items = test_matrix.shape
    if num_users > P.shape[0] or num_items > Q.shape[0]:
        test_matrix = test_matrix[:P.shape[0], :Q.shape[0]]

    # Compute the predicted ratings
    predictions = P @ Q.T  # predictions is a NumPy array with shape (6040, 3952)

    # Extract non-zero entries from the test matrix
    row_indices, col_indices = test_matrix.nonzero()

    # Debug: Check maximum indices
    max_row_index = np.max(row_indices)
    max_col_index = np.max(col_indices)
    # print(f"Max row index: {max_row_index}, Max col index: {max_col_index}")
    # print(f"Predictions shape: {predictions.shape}")

    # Ensure indices are within bounds
    assert max_row_index < predictions.shape[0], "Row index out of bounds"
    assert max_col_index < predictions.shape[1], "Column index out of bounds"

    # Get the actual test ratings
    # test_values = test_matrix[row_indices, col_indices].A1  # Convert to 1D array using .A1
    test_values = test_matrix[row_indices, col_indices].flatten()

    # Predict ratings for the corresponding user-item pairs
    predicted_values = predictions[row_indices, col_indices]

    # Calculate the Root Mean Squared Error
    rmse = np.sqrt(np.mean((test_values - predicted_values) ** 2))

    return rmse

def op_hybrid_parameter_optimization(ep, index, train, test, n_calls=30):
    lambda_ = 0.5  # trade-off between var and rmse
    var_max = 1.0  # normalize variance
    rmse_max = 5.0  # normalize RMSE based on 5-star rating
    l, u = 1, 5
    sensitivity = 4

    @use_named_args(search_space)
    def objective(k, m, y):
        if y <= 0:
            return 10.0  # skip invalid region

        sensitivity_Cp = sensitivity_Cp_fun(ep, k, m, y, index)
        Cp_assume = 0  # or sensitivity_Cp / 2

        if checkConstraints(ep, k, m, y, Cp_assume, index) != 0:
            return 10.0  # infeasible

        try:
            var_tmp = theory_var_fun(ep, k, m, y, Cp_assume, index)
            L = PM.LValue(ep, k, m, y, index)
            R_perturbed = perturb_matrix(ep, train, sensitivity, l, u, index, k, m, y, L, sensitivity_Cp)

            # Use lighter ALS config for speed
            P, Q = als.als_explicit(R_perturbed, num_factors=20, regularization=0.1, num_iterations=15)
            rmse_val = calculate_rmse(test, P, Q)

            norm_var = var_tmp / var_max
            norm_rmse = rmse_val / rmse_max
            J = lambda_ * norm_var + (1 - lambda_) * norm_rmse

            return J

        except Exception as e:
            print("Error during optimization at (k, m, y):", k, m, y, "→", str(e))
            return 10.0  # penalize failure

    # Run Bayesian Optimization
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        acq_func="EI",  # Expected Improvement
        n_calls=n_calls,
        random_state=42,
        verbose=True
    )

    # Extract best result
    k_best, m_best, y_best = result.x
    print(f"Best parameters found: k={k_best:.4f}, m={m_best:.4f}, y={y_best:.4f} with hybrid loss {result.fun:.4f}")
    return k_best, m_best, y_best


def parameter_optimization(ep, index, n_calls=30):
    best_reduce_rate = -1000

    @use_named_args(search_space)
    def objective(k, m, y):
        if y <= 0:
            return 1000  # Penalize invalid y

        sensitivity_Cp = sensitivity_Cp_fun(ep, k, m, y, index)
        Cp_assume = 0  # Can try sensitivity_Cp / 2 as a variant

        if checkConstraints(ep, k, m, y, Cp_assume, index) != 0:
            return 1000  # Penalize infeasible regions

        try:
            var_tmp = theory_var_fun(ep, k, m, y, Cp_assume, index)
            var_lap = (sensitivity_Cp / ep) ** 2 * 2
            reduce_rate = reduceRate(var_lap, var_tmp)

            # We return -reduceRate because gp_minimize minimizes
            return -reduce_rate

        except Exception as e:
            print("Error at (k, m, y):", k, m, y, "→", str(e))
            return 1000

    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        acq_func="EI",  # Expected Improvement
        n_calls=n_calls,
        random_state=42,
        verbose=True
    )

    k_best, m_best, y_best = result.x
    best_reduce_rate = -result.fun

    print(f"Best parameters found: k={k_best:.4f}, m={m_best:.4f}, y={y_best:.4f} with reduceRate={best_reduce_rate:.4f}")
    return k_best, m_best, y_best