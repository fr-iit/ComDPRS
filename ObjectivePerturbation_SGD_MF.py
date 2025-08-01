import numpy as np
import DataLoader as DL
from sklearn.metrics import mean_squared_error, mean_absolute_error

def dp_matrix_factorization(R, K, gamma_init, emax, epsilon, lamb, itr=150, tol=1e-5, decay=0.01):
    n_users, n_items = R.shape
    P = np.random.rand(n_users, K)
    Q = np.random.rand(n_items, K)

    delta_r = 2  # Sensitivity of ratings
    prev_rmse = float('inf')  # Store previous RMSE
    gamma = gamma_init  # Initialize learning rate

    for step in range(itr):
        total_squared_error = 0  # Track sum of squared errors
        count = 0  # Count the number of rated entries

        for u in range(n_users):
            non_zero_indices = np.where(R[u, :] > 0)[0]  # Get rated items by user `u`
            for i in non_zero_indices:
                predicted_rating = np.dot(P[u, :], Q[i, :])
                noisy_error = R[u, i] - predicted_rating + np.random.laplace(0, itr * delta_r / epsilon)

                # Clamp noisy error
                noisy_error = np.clip(noisy_error, -emax, emax)

                # Compute gradients
                P_update = gamma * (noisy_error * Q[i, :] - lamb * P[u, :])
                Q_update = gamma * (noisy_error * P[u, :] - lamb * Q[i, :])

                # Apply updates
                P[u, :] += P_update
                Q[i, :] += Q_update

                # Compute squared error for RMSE calculation
                total_squared_error += noisy_error ** 2
                count += 1  # Count the number of rated elements

        # Compute RMSE for this iteration
        current_rmse = np.sqrt(total_squared_error / count)

        if abs(prev_rmse - current_rmse) < tol:
            gamma *= 0.9  # Reduce learning rate by 10% if RMSE change is too small
        elif abs(prev_rmse - current_rmse) > 5 * tol:
            gamma *= 1.05  # Increase learning rate by 5% if RMSE is improving fast

        # Apply Learning Rate Decay Over Time
        gamma = gamma_init / (1 + decay * step)

        if abs(prev_rmse - current_rmse) < tol:
            print(f"Converged at iteration {step}: RMSE = {current_rmse:.4f}, Learning Rate = {gamma:.4f}")
            break

        prev_rmse = current_rmse  # Update previous RMSE

    return P, Q

def split_data(R, test_size=0.2):
    """Split rating matrix into train and test sets."""
    train = R.copy()
    test = np.zeros_like(R)

    for user_idx in range(R.shape[0]):
        non_zero_indices = np.where(R[user_idx] > 0)[0]
        if len(non_zero_indices) == 0:
            continue
        test_indices = np.random.choice(non_zero_indices, size=int(test_size * len(non_zero_indices)), replace=False)
        train[user_idx, test_indices] = 0  # Mask test ratings in train set
        test[user_idx, test_indices] = R[user_idx, test_indices]  # Save original test ratings

    return train, test

def compute_rmse(R_test, R_predicted):
    """Compute RMSE for the test set."""
    non_zero_indices = R_test > 0
    mse = mean_squared_error(R_test[non_zero_indices], R_predicted[non_zero_indices])
    return np.sqrt(mse)

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
    dataset = 'beauty'

    if dataset == '100k':
        R = DL.load_user_item_matrix_100k()  # DL.load_user_item_matrix_100k() # --- this should be original data
    elif dataset == '1m':
        R = DL.load_user_item_matrix_1m()
    elif dataset == 'yahoo':
        R = DL.load_user_item_matrix_yahoo()  # DL.load_user_item_matrix_yahoo()
    elif dataset == 'beauty':
        R = DL.load_user_item_matrix_beauty()  # DL.load_user_item_matrix_beauty()

    train, test = split_data(R)

    # Parameters
    d = 10  # Latent factors
    gamma = 0.005  # Learning rate
    lamb = 1.9  # Regularization
    emax = 2  # Error bound


    # Perform DP Matrix Factorization
    P, Q = dp_matrix_factorization(train, d, gamma, emax, epsilon, lamb)

    # Predicted ratings matrix
    R_predicted = np.dot(P, Q.T)

    topN = 10

    # Compute MAE
    test_mask = test > 0
    true_ratings = test[test_mask]
    predicted_ratings = R_predicted[test_mask]
    mae = calculate_mae(true_ratings, predicted_ratings)

    # Compute Hit Rate@topN
    test_users, test_items = np.nonzero(test)
    test_true_items = test_items.tolist()

    predicted_scores = R_predicted[test_users, :]
    hr = hit_rate(predicted_scores, test_true_items, k=topN)

    print(f"\n=== Final Report ===")
    print(f'ep: {epsilon}, dataset: {dataset}')
    print(f"MAE: {mae:.4f}")
    print(f"Hit Rate@{topN}: {hr:.4f}")