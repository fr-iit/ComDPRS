from Mechanism.Mapping import *
from Mechanism.Evaluation import *
import Perturbation_Mechanism as PM
import RecSys_ALS as als

# ==========Parameter Optimization==========#
def main_parameter_optimization(ep, index):
    step1 = 0.1
    step2 = 0.01
    step3 = 0.001
    k_best = 0
    m_best = 0
    y_best = 0
    reduceRate_best = -1000

    y_upper = 1
    k_upper = 1
    m_upper = 2

    # First round step1
    y_count = 0
    while (y_count < y_upper):
        k_count = 0
        while (k_count < k_upper):
            m_count = 0
            while (m_count <= m_upper):
                if y_count == 0:
                    m_count = m_count + step1
                    continue
                sensitivity_Cp = sensitivity_Cp_fun(ep, k_count, m_count, y_count, index)
                # Cp_assume = sensitivity_Cp / 2
                Cp_assume = 0
                if (checkConstraints(ep, k_count, m_count, y_count, Cp_assume, index) == 0):
                    var_tmp = theory_var_fun(ep, k_count, m_count, y_count, Cp_assume, index)
                    var_lap = (sensitivity_Cp / ep) ** 2 * 2
                    reduceRate_tmp = reduceRate(var_lap, var_tmp)

                    if (reduceRate_tmp > reduceRate_best):
                        reduceRate_best = reduceRate_tmp
                        k_best = k_count
                        m_best = m_count
                        y_best = y_count

                m_count = m_count + step1
            k_count = k_count + step1
        y_count = y_count + step1

    # Second round step2
    if (y_best == 0):
        y_count = 0
        y_count_end = 0.1
    else:
        y_count = y_best - step1
        y_count_end = y_best + step1
    while (y_count < y_count_end):
        if (k_best == 0):
            k_count = 0
            k_count_end = 0.1
        else:
            k_count = k_best - step1
            k_count_end = k_best + step1
        while (k_count < k_count_end):
            if (m_best == 0):
                m_count = 0
                m_count_end = 0.1
            else:
                m_count = m_best - step1
                m_count_end = m_best + step1
            while (m_count < m_count_end):
                if y_count == 0:
                    m_count = m_count + step2
                    continue
                sensitivity_Cp = sensitivity_Cp_fun(ep, k_count, m_count, y_count, index)
                # Cp_assume = sensitivity_Cp / 2
                Cp_assume = 0
                if (checkConstraints(ep, k_count, m_count, y_count, Cp_assume, index) == 0):
                    var_tmp = theory_var_fun(ep, k_count, m_count, y_count, Cp_assume, index)
                    var_lap = (sensitivity_Cp / ep) ** 2 * 2
                    reduceRate_tmp = reduceRate(var_lap, var_tmp)

                    if (reduceRate_tmp > reduceRate_best):
                        reduceRate_best = reduceRate_tmp
                        k_best = k_count
                        m_best = m_count
                        y_best = y_count

                m_count = m_count + step2
            k_count = k_count + step2
        y_count = y_count + step2

    # Third round step3
    if (y_best == 0):
        y_count = 0
        y_count_end = 0.01
    else:
        y_count = y_best - step2
        y_count_end = y_best + step2
    while (y_count < y_count_end):
        if (k_best == 0):
            k_count = 0
            k_count_end = 0.01
        else:
            k_count = k_best - step2
            k_count_end = k_best + step2
        while (k_count < k_count_end):
            if (m_best == 0):
                m_count = 0
                m_count_end = 0.01
            else:
                m_count = m_best - step2
                m_count_end = m_best + step2
            while (m_count < m_count_end):
                if y_count == 0:
                    m_count = m_count + step3
                    continue
                sensitivity_Cp = sensitivity_Cp_fun(ep, k_count, m_count, y_count, index)
                # Cp_assume = sensitivity_Cp / 2
                Cp_assume = 0
                if (checkConstraints(ep, k_count, m_count, y_count, Cp_assume, index) == 0):
                    var_tmp = theory_var_fun(ep, k_count, m_count, y_count, Cp_assume, index)
                    var_lap = (sensitivity_Cp / ep) ** 2 * 2
                    reduceRate_tmp = reduceRate(var_lap, var_tmp)

                    if (reduceRate_tmp > reduceRate_best):
                        reduceRate_best = reduceRate_tmp
                        k_best = k_count
                        m_best = m_count
                        y_best = y_count

                m_count = m_count + step3
            k_count = k_count + step3
        y_count = y_count + step3

    return k_best, m_best, y_best

def hybrid_parameter_optimization(ep, index, train, test):
    step1 = 0.1
    step2 = 0.01
    step3 = 0.001
    k_best = 0
    m_best = 0
    y_best = 0
    reduceRate_best = -1000

    best_J = float('inf')
    lambda_ = 0.5  # adjust as needed
    var_max = 1.0  # estimate or tune
    rmse_max = 5.0  # based on rating scale

    y_upper = 1
    k_upper = 1
    m_upper = 2

    l = 1
    u = 5
    sensitivity = 4

    # First round step1
    y_count = 0
    print('step1 :', step1)
    while (y_count < y_upper):
        k_count = 0
        while (k_count < k_upper):
            m_count = 0
            while (m_count <= m_upper):
                if y_count == 0:
                    m_count = m_count + step1
                    continue
                sensitivity_Cp = sensitivity_Cp_fun(ep, k_count, m_count, y_count, index)
                # Cp_assume = sensitivity_Cp / 2
                Cp_assume = 0

                if (checkConstraints(ep, k_count, m_count, y_count, Cp_assume, index) == 0):
                    var_tmp = theory_var_fun(ep, k_count, m_count, y_count, Cp_assume, index)
                    L = PM.LValue(ep, k_count, m_count, y_count, index)
                    R_perturbed = perturb_matrix(ep, train, sensitivity, l, u, index, k_count, m_count, y_count, L, sensitivity_Cp)

                    P, Q = als.als_explicit(R_perturbed, num_factors=10, regularization=0.1, num_iterations=10)
                    # als_predict = als.als_predict_func(P, Q)

                    rmse_val = calculate_rmse(test, P, Q)

                    # Normalize both variance and RMSE
                    norm_var = var_tmp / var_max
                    norm_rmse = rmse_val / rmse_max

                    # Hybrid loss function
                    J = lambda_ * norm_var + (1 - lambda_) * norm_rmse

                    # Update best parameters
                    if J < best_J:
                        best_J = J
                        k_best = k_count
                        m_best = m_count
                        y_best = y_count

                    # var_lap = (sensitivity_Cp / ep) ** 2 * 2
                    # reduceRate_tmp = reduceRate(var_lap, var_tmp)
                    #
                    # if (reduceRate_tmp > reduceRate_best):
                    #     reduceRate_best = reduceRate_tmp
                    #     k_best = k_count
                    #     m_best = m_count
                    #     y_best = y_count

                m_count = m_count + step1
            k_count = k_count + step1
        y_count = y_count + step1

    # Second round step2
    print('step2 :', step2)
    if (y_best == 0):
        y_count = 0
        y_count_end = 0.1
    else:

        y_count = y_best - step1
        y_count_end = y_best + step1
    while (y_count < y_count_end):
        if (k_best == 0):
            k_count = 0
            k_count_end = 0.1
        else:
            k_count = k_best - step1
            k_count_end = k_best + step1
        while (k_count < k_count_end):
            if (m_best == 0):
                m_count = 0
                m_count_end = 0.1
            else:
                m_count = m_best - step1
                m_count_end = m_best + step1
            while (m_count < m_count_end):
                if y_count == 0:
                    m_count = m_count + step2
                    continue
                sensitivity_Cp = sensitivity_Cp_fun(ep, k_count, m_count, y_count, index)
                # Cp_assume = sensitivity_Cp / 2
                Cp_assume = 0
                if (checkConstraints(ep, k_count, m_count, y_count, Cp_assume, index) == 0):
                    var_tmp = theory_var_fun(ep, k_count, m_count, y_count, Cp_assume, index)
                    L = PM.LValue(ep, k_count, m_count, y_count, index)
                    R_perturbed = perturb_matrix(ep, train, sensitivity, l, u, index, k_count, m_count, y_count, L, sensitivity_Cp)

                    # P, Q = als.als_explicit(R_perturbed, num_factors=30, regularization=0.1, num_iterations=30)
                    P, Q = als.als_explicit(R_perturbed, num_factors=10, regularization=0.1, num_iterations=10)
                    # als_predict = als.als_predict_func(P, Q)

                    rmse_val = calculate_rmse(test, P, Q)

                    # Normalize both variance and RMSE
                    norm_var = var_tmp / var_max
                    norm_rmse = rmse_val / rmse_max

                    # Hybrid loss function
                    J = lambda_ * norm_var + (1 - lambda_) * norm_rmse

                    # Update best parameters
                    if J < best_J:
                        best_J = J
                        k_best = k_count
                        m_best = m_count
                        y_best = y_count



                    # var_lap = (sensitivity_Cp / ep) ** 2 * 2
                    # reduceRate_tmp = reduceRate(var_lap, var_tmp)
                    #
                    # if (reduceRate_tmp > reduceRate_best):
                    #     reduceRate_best = reduceRate_tmp
                    #     k_best = k_count
                    #     m_best = m_count
                    #     y_best = y_count

                m_count = m_count + step2
            k_count = k_count + step2
        y_count = y_count + step2

    # Third round step3
    print('step3 :', step3)
    if (y_best == 0):
        y_count = 0
        y_count_end = 0.01
    else:
        y_count = y_best - step2
        y_count_end = y_best + step2
    while (y_count < y_count_end):
        if (k_best == 0):
            k_count = 0
            k_count_end = 0.01
        else:
            k_count = k_best - step2
            k_count_end = k_best + step2
        while (k_count < k_count_end):
            if (m_best == 0):
                m_count = 0
                m_count_end = 0.01
            else:
                m_count = m_best - step2
                m_count_end = m_best + step2
            while (m_count < m_count_end):
                if y_count == 0:
                    m_count = m_count + step3
                    continue
                sensitivity_Cp = sensitivity_Cp_fun(ep, k_count, m_count, y_count, index)
                # Cp_assume = sensitivity_Cp / 2
                Cp_assume = 0
                if (checkConstraints(ep, k_count, m_count, y_count, Cp_assume, index) == 0):
                    var_tmp = theory_var_fun(ep, k_count, m_count, y_count, Cp_assume, index)

                    L = PM.LValue(ep, k_count, m_count, y_count, index)
                    R_perturbed = perturb_matrix(ep, train, sensitivity, l, u, index, k_count, m_count, y_count, L, sensitivity_Cp)

                    P, Q = als.als_explicit(R_perturbed, num_factors=10, regularization=0.1, num_iterations=10)
                    # als_predict = als.als_predict_func(P, Q)

                    rmse_val = calculate_rmse(test, P, Q)

                    # Normalize both variance and RMSE
                    norm_var = var_tmp / var_max
                    norm_rmse = rmse_val / rmse_max

                    # Hybrid loss function
                    J = lambda_ * norm_var + (1 - lambda_) * norm_rmse

                    # Update best parameters
                    if J < best_J:
                        best_J = J
                        k_best = k_count
                        m_best = m_count
                        y_best = y_count

                    # var_lap = (sensitivity_Cp / ep) ** 2 * 2
                    # reduceRate_tmp = reduceRate(var_lap, var_tmp)
                    #
                    # if (reduceRate_tmp > reduceRate_best):
                    #     reduceRate_best = reduceRate_tmp
                    #     k_best = k_count
                    #     m_best = m_count
                    #     y_best = y_count

                m_count = m_count + step3
            k_count = k_count + step3
        y_count = y_count + step3

    return k_best, m_best, y_best

# ==========================================#

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


def perturb_matrix(ep, train_matrix, sensitivity, lower, upper, index, k, m, y, L, Cp_Sen):
    """
    Applies Composite DP perturbation to each nonzero entry in the matrix.
    """
    R_perturbed = np.zeros_like(train_matrix)
    n_users, n_items = train_matrix.shape

    for i in range(n_users):
        for j in range(n_items):
            if train_matrix[i, j] > 0:
                R_perturbed[i, j] = np.round(PM.perturbation_fun_optimized(
                    ep, train_matrix[i, j], sensitivity, lower, upper,index, k, m, y, L, Cp_Sen))
                print("Optimized Perturbed Value:", R_perturbed[i, j], ' for original value: ', train_matrix[i, j])

    return R_perturbed
