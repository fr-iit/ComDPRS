import numpy as np
import DataLoader as DL
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error, mean_absolute_error
import csv


def compute_item_averages(R, beta_m, epsilon_1, epsilon_2, r_min, r_max):

    n_users, m_items = R.shape

    delta_r = r_max - r_min
    global_avg = (np.sum(R) / (n_users * m_items)) + np.random.laplace(0, delta_r / epsilon_1)

    IAvg = np.zeros(m_items)
    for j in range(m_items):

        R_j = R[:, j]
        num_ratings_j = np.count_nonzero(R_j)

        sum_ratings_j = np.sum(R_j)

        IAvg[j] = (sum_ratings_j + beta_m * global_avg + np.random.laplace(0, delta_r / epsilon_2)) / (
                    num_ratings_j + beta_m)

        IAvg[j] = np.clip(IAvg[j], r_min, r_max)

    return IAvg


def compute_user_averages(R, IAvg, beta_u, epsilon_1, epsilon_2):

    n_users, m_items = R.shape

    R_prime = R - IAvg

    delta_r = 2  # Sensitivity (maximum rating difference for clamping to [-2, 2])
    global_avg_prime = (np.sum(R_prime) / (n_users * m_items)) + np.random.laplace(0, delta_r / epsilon_1)

    UAvg = np.zeros(n_users)
    for u in range(n_users):
        # Extract adjusted ratings for user u
        R_u = R_prime[u, :]
        num_ratings_u = np.count_nonzero(R_u)  # Number of non-zero ratings for user u

        # Sum of adjusted ratings for user u
        sum_ratings_u = np.sum(R_u)

        # Compute the differentially private user average
        UAvg[u] = (sum_ratings_u + beta_u * global_avg_prime + np.random.laplace(0, delta_r / epsilon_2)) / (
                    num_ratings_u + beta_u)

        # Clamp the user average to the range [-2, 2]
        UAvg[u] = np.clip(UAvg[u], -2, 2)

    return UAvg


def calculate_discounted_and_clamped_matrix(R, IAvg, UAvg, B):

    n_users, n_items = R.shape

    # Initialize the adjusted matrix
    R_discounted = np.zeros_like(R)

    # Loop through users and items
    for u in range(n_users):
        for i in range(n_items):
            # Discount the item and user averages
            R_discounted[u, i] = R[u, i] - (IAvg[i] + UAvg[u])
            # Clamp the result to the range [-B, B]
            R_discounted[u, i] = np.clip(R_discounted[u, i], -B, B)

    return R_discounted

def input_perturbation(R, delta_r, epsilon, B):
    noise = np.random.laplace(0, delta_r / epsilon)
    noise_R = R + noise
    return np.clip(noise_R, -B, B)

def als_matrix_factorization(R, d, lambda_reg, max_iters=150, tol=1e-5):

    n_users, n_items = R.shape
    P = np.random.rand(n_users, d)
    Q = np.random.rand(n_items, d)

    prev_rmse = float('inf')
    for iteration in range(max_iters):
        # Solve for P
        for u in range(n_users):
            rated_items = np.where(R[u, :] > 0)[0]
            if len(rated_items) == 0:
                continue
            Q_rated = Q[rated_items, :]
            R_u = R[u, rated_items]
            P[u, :] = solve(Q_rated.T @ Q_rated + lambda_reg * np.eye(d), Q_rated.T @ R_u)

        # Solve for Q
        for i in range(n_items):
            rated_users = np.where(R[:, i] > 0)[0]
            if len(rated_users) == 0:
                continue
            P_rated = P[rated_users, :]
            R_i = R[rated_users, i]
            Q[i, :] = solve(P_rated.T @ P_rated + lambda_reg * np.eye(d), P_rated.T @ R_i)

        # Compute RMSE
        R_pred = np.dot(P, Q.T)
        mask = R > 0
        rmse = np.sqrt(np.mean((R[mask] - R_pred[mask]) ** 2))

        if abs(prev_rmse - rmse) < tol:
            print(f"Iteration {iteration + 1} Converged!")
            break
        prev_rmse = rmse

    return P, Q

def predict_ratings(P, Q, avgI, avgU):
    return np.dot(P, Q.T) + avgI.reshape(1, -1) + avgU.reshape(-1, 1)

def predict_ratings_original(P, Q):
    return np.dot(P, Q.T)


def save_perturbed_ratings(R, filename):
    with open(filename, 'w') as f:
        for user_id in range(R.shape[0]):
            for item_id in range(R.shape[1]):
                if R[user_id, item_id] > 0:  # Only save rated items
                    f.write(f"{user_id + 1}::{item_id + 1}::{R[user_id, item_id]:.2f}::000000000\n")

    print(f"Perturbed ratings saved to {filename}")

def compute_item_popularity(interaction_matrix):
    num_users = interaction_matrix.shape[0]
    return np.sum(interaction_matrix > 0, axis=0) / num_users  # Fraction of users who interacted


def calculate_mae(true_ratings, predicted_ratings, min_val=1.0, max_val=5.0):
    mae = mean_absolute_error(true_ratings, predicted_ratings)
    return mae

def hit_rate(predicted_scores, test_items, k=10):
    hits = 0
    for pred, true_item in zip(predicted_scores, test_items):
        top_k_items = np.argsort(pred)[::-1][:k]
        if true_item in top_k_items:
            hits += 1
    return hits / len(test_items)

# Example usage
if __name__ == "__main__":

    num_runs = 5  # Number of times to run the experiment

    # Store results for each metric
    mae_list = []
    HR_list = []

    topN = 10
    lambda_reg = 1.9
    K = 10
    # total ep should be
    epsilon = 1.0
    dataset = 'beauty'

    # Load example data
    if dataset == '100k':
        R = DL.load_user_item_matrix_100k_DP()
    elif dataset == '1m':
        R = DL.load_user_item_matrix_1m_DP()
    elif dataset == 'yahoo':
        R = DL.load_user_item_matrix_yahoo_DP()
    elif dataset == 'beauty':
        R = DL.load_user_item_matrix_beauty_DP()

    user_profiles = DL.user_profile(R)
    item_popularity = compute_item_popularity(R)

    beta_m = 1.0
    beta_u = 1.0

    G_ep = 0.3 * epsilon  # Global average privacy parameter
    I_ep = 0.25 * epsilon  # Item average privacy parameter
    U_ep = 0.25 * epsilon  # Global average privacy parameter for user
    Input_ep = 0.2 * epsilon

    print(f'G_ep: {G_ep}, I_ep: {I_ep}, U_ep: {U_ep}, Input_ep: {Input_ep}')

    r_min, r_max = 1, 5  # Rating limits

    # Step 1: Compute Item Averages
    # IAvg = compute_item_averages(R, beta_m, G_ep, I_ep, r_min, r_max)
    # print("Item Averages len:", len(IAvg))

    # Step 2: Compute User Averages
    # UAvg = compute_user_averages(R, IAvg, beta_u, G_ep, U_ep)
    # print("User Averages len:", len(UAvg))

    B = 1

    # Compute the discounted and clamped matrix
    # R_discounted_clamped = calculate_discounted_and_clamped_matrix(R, IAvg, UAvg, B)
    #
    # input_R = input_perturbation(R_discounted_clamped, delta_r= 2, epsilon=Input_ep, B = 1)

    # Variables to track the best run
    best_rmse = float('inf')  # Initialize with a high value
    best_recall = 0  # Initialize with the lowest value
    best_run = -1
    best_R_predicted = None  # Store the best predicted rating matrix

    for run in range(num_runs):
        print(f"===== Run {run + 1}/{num_runs} =====")

        R_train, R_test = DL.split_data(R, test_size=0.2)

        IAvg = compute_item_averages(R_train, beta_m, G_ep, I_ep, r_min, r_max)
        print("Item Averages len:", len(IAvg))

        UAvg = compute_user_averages(R_train, IAvg, beta_u, G_ep, U_ep)
        print("User Averages len:", len(UAvg))

        R_discounted_clamped = calculate_discounted_and_clamped_matrix(R_train, IAvg, UAvg, B)

        input_R = input_perturbation(R_discounted_clamped, delta_r=2, epsilon=Input_ep, B=1)

        # Train ALS
        P, Q = als_matrix_factorization(input_R, d=K, lambda_reg=lambda_reg)
        # P, Q = als_matrix_factorization(R_train, d=K, lambda_reg=lambda_reg) # no DP

        # Predict ratings
        R_predicted = predict_ratings(P, Q, IAvg, UAvg)
        # R_predicted = predict_ratings_original(P, Q) # no DP

        # Compute MAE
        test_mask = R_test > 0
        true_ratings = R_test[test_mask]
        predicted_ratings = R_predicted[test_mask]
        mae = calculate_mae(true_ratings, predicted_ratings)
        mae_list.append(mae)

        # Compute Hit Rate@topN
        test_users, test_items = np.nonzero(R_test)
        test_true_items = test_items.tolist()

        predicted_scores = R_predicted[test_users, :]
        hr = hit_rate(predicted_scores, test_true_items, k=topN)
        HR_list.append(hr)

    # Report average results
    avg_mae = np.mean(mae_list)
    avg_hr = np.mean(HR_list)

    print(f"\n=== Final Report After {num_runs} Runs ===")
    print(f'dataset: {dataset}')
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average Hit Rate@{topN}: {avg_hr:.4f}")
