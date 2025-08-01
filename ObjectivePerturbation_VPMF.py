import numpy as np
from numpy.linalg import inv
from numpy.linalg import solve
import DataLoader as DL
from sklearn.metrics import mean_squared_error, mean_absolute_error

def laplace_noise(scale, shape):
    return np.random.laplace(0, scale, shape)

def vp_dpmf(P, R, max_q, min_q, lambda_Q, epsilon, delta_r):
    n_users, n_items = R.shape
    K = P.shape[1]  # Number of latent features
    Q = np.random.rand(n_items, K)  # Initialize Q randomly
    Q_perturbed = np.zeros_like(Q)
    max_norm = 5.0

    for i in range(n_items):

        users_who_rated = np.where(R[:, i] > 0)[0]
        if len(users_who_rated) == 0:
            continue

        P_ui = P[users_who_rated, :]  # Subset of P for users who rated item i
        r_i = R[users_who_rated, i]  # Ratings for item i

        # Compute Delta
        max_user_factor_norm = max(np.linalg.norm(P[u, :], ord=1) for u in users_who_rated)
        Delta = max_user_factor_norm * delta_r

        # Generate Laplace noise
        noise = laplace_noise(Delta / epsilon, (K, 1))

        q_i_perturbed = inv(P_ui.T @ P_ui + lambda_Q * np.eye(K)) @ (P_ui.T @ r_i - noise.flatten())

        if np.linalg.norm(q_i_perturbed) > max_norm:
            q_i_perturbed = (q_i_perturbed / np.linalg.norm(q_i_perturbed)) * max_norm

        q_i_perturbed = np.clip(q_i_perturbed, min_q, max_q)

        Q_perturbed[i, :] = q_i_perturbed


    print(f"original, max: {max_q}, min: {min_q}")
    print(f'max Q: {np.max(Q_perturbed)} and min Q for perturb: {np.min(Q_perturbed)}')
    return Q_perturbed

def als_matrix_factorization(R, K, lambda_reg, max_iters=150, tol=1e-5):

    n_users, n_items = R.shape
    P = np.random.rand(n_users, K)
    Q = np.random.rand(n_items, K)

    prev_rmse = float('inf')

    for iteration in range(max_iters):
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
        R_pred = np.dot(P, Q.T)
        mask = R > 0  # Only consider observed ratings
        rmse = np.sqrt(np.mean((R[mask] - R_pred[mask]) ** 2))

        print(f"Iteration {iteration + 1}: RMSE = {rmse:.4f}")

        # Check for convergence
        if abs(prev_rmse - rmse) < tol:
            print("Converged!")
            break
        prev_rmse = rmse

    return P, Q

def predict_ratings(P, Q):
    return np.dot(P, Q.T)

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

    epsilon = 0.1
    dataset = 'beauty'

    if dataset == '100k':
        R = DL.load_user_item_matrix_100k()
    elif dataset == '1m':
        R = DL.load_user_item_matrix_1m()
    elif dataset == 'yahoo':
        R = DL.load_user_item_matrix_yahoo()
    elif dataset == 'beauty':
        R = DL.load_user_item_matrix_beauty()

    R_train, R_test = DL.split_data(R)

    K = 10  # Number of latent factors
    lambda_Q = 0.1
    lambda_reg = 0.017
    # epsilon = 1.0  # Privacy budget
    delta_r = 1.0  # Sensitivity of ratings

    P, Q = als_matrix_factorization(R_train, K, lambda_reg)
    max_q = np.max(Q)
    min_q = np.min(Q)

    pred = predict_ratings(P,Q)


    # Apply VP-DPMF to obtain perturbed item latent factor matrix
    Q_perturbed = vp_dpmf(P, R_train, max_q, min_q, lambda_Q, epsilon, delta_r)

    # Predict the full ratings matrix
    R_predicted = predict_ratings(P, Q_perturbed)

    topN = 10

    # Compute MAE
    test_mask = R_test > 0
    true_ratings = R_test[test_mask]
    predicted_ratings = R_predicted[test_mask]
    mae = calculate_mae(true_ratings, predicted_ratings)

    # Compute Hit Rate@topN
    test_users, test_items = np.nonzero(R_test)
    test_true_items = test_items.tolist()

    predicted_scores = R_predicted[test_users, :]
    hr = hit_rate(predicted_scores, test_true_items, k=topN)

    print(f"\n=== Final Report of VPMF ===")
    print(f'ep: {epsilon}, dataset: {dataset}')
    print(f"MAE: {mae:.4f}")
    print(f"Hit Rate@{topN}: {hr:.4f}")
