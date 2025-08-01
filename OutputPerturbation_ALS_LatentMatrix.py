import numpy as np
import DataLoader as DL
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_rmse(R, P, Q):
    R_pred = np.dot(P, Q.T)
    mask = R > 0  # Only consider observed ratings
    mse = mean_squared_error(R[mask], R_pred[mask])
    return np.sqrt(mse)

def matrix_factorization_with_output_perturbation_als(R, K=10, lambda_reg=0.1, epsilon=1.0, max_iter=150, tol=1e-5):

    n_users, n_items = R.shape

    P = np.random.rand(n_users, K)
    Q = np.random.rand(n_items, K)

    prev_rmse = float('inf')

    # ALS optimization
    for iteration in range(max_iter):
        # print(iteration)
        # Solve for P (users)
        for u in range(n_users):
            rated_items = np.where(R[u, :] > 0)[0]  # Get indices of rated items
            if len(rated_items) == 0:
                continue
            Q_rated = Q[rated_items, :]
            R_u = R[u, rated_items]
            P[u, :] = solve(Q_rated.T @ Q_rated + lambda_reg * np.eye(K), Q_rated.T @ R_u)

        # Solve for Q (items)
        for i in range(n_items):
            rated_users = np.where(R[:, i] > 0)[0]  # Get indices of users who rated item i
            if len(rated_users) == 0:
                continue
            P_rated = P[rated_users, :]
            R_i = R[rated_users, i]
            Q[i, :] = solve(P_rated.T @ P_rated + lambda_reg * np.eye(K), P_rated.T @ R_i)

        # Compute RMSE
        rmse = compute_rmse(R, P, Q)
        if abs(prev_rmse - rmse) < tol:
            print("Converged!")
            break
        prev_rmse = rmse

    R_predicted = np.dot(P, Q.T)

    # Measure sensitivity
    R_perturbed = R + np.random.normal(0, 0.1, R.shape)  # Small Gaussian noise
    P_perturbed = np.random.rand(n_users, K)
    Q_perturbed = np.random.rand(n_items, K)

    # Recompute with perturbed data using ALS
    for iteration in range(max_iter):
        # Solve for P_perturbed (users)
        for u in range(n_users):
            rated_items = np.where(R_perturbed[u, :] > 0)[0]
            if len(rated_items) == 0:
                continue
            Q_rated = Q_perturbed[rated_items, :]
            R_u = R_perturbed[u, rated_items]
            P_perturbed[u, :] = solve(Q_rated.T @ Q_rated + lambda_reg * np.eye(K), Q_rated.T @ R_u)

        # Solve for Q_perturbed (items)
        for i in range(n_items):
            rated_users = np.where(R_perturbed[:, i] > 0)[0]
            if len(rated_users) == 0:
                continue
            P_rated = P_perturbed[rated_users, :]
            R_i = R_perturbed[rated_users, i]
            Q_perturbed[i, :] = solve(P_rated.T @ P_rated + lambda_reg * np.eye(K), P_rated.T @ R_i)

    R_predicted_perturbed = np.dot(P_perturbed, Q_perturbed.T)

    # Calculate the sensitivity as the maximum absolute difference
    sensitivity = np.max(np.abs(R_predicted - R_predicted_perturbed))

    # Add Laplace noise to P and Q for output perturbation
    noise_scale = sensitivity / epsilon

    P_noisy = P + np.random.laplace(0, noise_scale, P.shape)
    Q_noisy = Q + np.random.laplace(0, noise_scale, Q.shape)

    # Compute the noisy predicted ratings matrix
    R_predicted_noisy = np.dot(P_noisy, Q_noisy.T)
    R_predicted_noisy = np.clip(R_predicted_noisy, 0, 5)  # Clip ratings to [0, 5]

    return P_noisy, Q_noisy, R_predicted_noisy, sensitivity


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

if __name__ == "__main__":

    epsilon = 0.1
    dataset = '100k'

    if dataset == '100k':
        R = DL.load_user_item_matrix_100k()  # DL.load_user_item_matrix_100k() # --- this should be original data
    elif dataset == '1m':
        R = DL.load_user_item_matrix_1m()
    elif dataset == 'yahoo':
        R = DL.load_user_item_matrix_yahoo()  # DL.load_user_item_matrix_yahoo()
    elif dataset == 'beauty':
        R = DL.load_user_item_matrix_beauty()  # DL.load_user_item_matrix_beauty()

    R_train, R_test = DL.split_data(R)

    K = 10  # Number of latent factors
    lambda_Q = 0.1
    lambda_reg = 0.017
    # epsilon = 1.0  # Privacy budget
    delta_r = 1.0  # Sensitivity of ratings

    P_noisy, Q_noisy, R_predicted_noisy, sensitivity = matrix_factorization_with_output_perturbation_als(
        R_train, K, lambda_reg, epsilon)

    print("Sensitivity:", sensitivity)

    topN = 10

    # Compute MAE
    test_mask = R_test > 0
    true_ratings = R_test[test_mask]
    predicted_ratings = R_predicted_noisy[test_mask]
    mae = calculate_mae(true_ratings, predicted_ratings)

    # Compute Hit Rate@topN
    test_users, test_items = np.nonzero(R_test)
    test_true_items = test_items.tolist()

    predicted_scores = R_predicted_noisy[test_users, :]
    hr = hit_rate(predicted_scores, test_true_items, k=topN)

    print(f"\n=== Final Report of LMMF ===")
    print(f'ep: {epsilon}, dataset: {dataset}')
    print(f"MAE: {mae:.4f}")
    print(f"Hit Rate@{topN}: {hr:.4f}")


