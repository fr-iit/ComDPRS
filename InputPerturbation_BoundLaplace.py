import numpy as np
import DataLoader as DL
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error

def laplace_noise(scale):
    return np.random.laplace(loc=0, scale=scale)

def blp_mechanism(r, l, u, epsilon):
    b = (u - l) / epsilon  # Calculate the sensitivity parameter b
    while True:
        noise = laplace_noise(b)
        r_star = r + noise

        if l <= r_star <= u:
            return r_star
        else:
            continue

def compute_rmseALS(R, P, Q):

    R_pred = np.dot(P, Q.T)
    mask = R > 0
    mse = mean_squared_error(R[mask], R_pred[mask])
    return np.sqrt(mse)


def als_explicit(R, K, lambda_reg, max_iter=150, tol=1e-4):

    n_users, n_items = R.shape
    P = np.random.rand(n_users, K)
    Q = np.random.rand(n_items, K)

    prev_rmse = float('inf')

    for iteration in range(max_iter):
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
        rmse = compute_rmseALS(R, P, Q)
        print(f"Iteration {iteration + 1}: RMSE = {rmse:.4f}")

        # Check for convergence
        if abs(prev_rmse - rmse) < tol:
            print("Converged!")
            break
        prev_rmse = rmse

    return P, Q


# Function to compute RMSE
def compute_rmse(R_test, R_predicted):
    # Only consider non-zero ratings in the test set
    non_zero_indices = R_test > 0
    mse = mean_squared_error(R_test[non_zero_indices], R_predicted[non_zero_indices])
    return np.sqrt(mse)

def compute_recall_at_k(R_test, R_predicted, k):
    recall_values = []
    num_users_with_recall = 0
    for user_idx in range(R_test.shape[0]):
        actual_items = np.where(R_test[user_idx] > 0)[0]
        if len(actual_items) == 0:
            continue  # Skip users with no test data

        predicted_items = np.argsort(R_predicted[user_idx])[-k:][::-1]
        hits = len(set(predicted_items).intersection(actual_items))
        recall = hits / len(actual_items)

        if recall > 0:
            num_users_with_recall += 1

        recall_values.append(recall)

    print(f"Users with non-zero recall: {num_users_with_recall}/{R_test.shape[0]}")
    return np.mean(recall_values)


def predict_ratings(P, Q):
    return np.dot(P, Q.T)


def save_perturbed_ratings(R, filename):
    with open(filename, 'w') as f:
        for user_id in range(R.shape[0]):
            for item_id in range(R.shape[1]):
                if R[user_id, item_id] > 0:  # Only save rated items
                    f.write(f"{user_id + 1}::{item_id + 1}::{R[user_id, item_id]:.2f}::000000000\n")

    print(f"Perturbed ratings saved to {filename}")


# Example usage
if __name__ == "__main__":

    # R = DL.load_user_item_matrix_1m()
    # R = DL.load_user_item_matrix_100k()
    # R = DL.load_user_item_matrix_yahoo()
    R = DL.load_user_item_matrix_10m()
    print(R.shape)
    n_users = R.shape[0]
    n_items = R.shape[1]
    print(n_items)
    # Rating scale bounds (e.g., 1 to 5 stars)
    l = 1.0
    u = 5.0

    epsilon = 0.01  # Privacy budget

    # Apply BLP mechanism to each rating in the matrix
    perturbed_ratings = np.zeros_like(R)
    for i in range(n_users):
        num_ratings = np.count_nonzero(R[i, :])
        user_ep = epsilon / num_ratings
        for j in range(n_items):
            if R[i, j] > 0 :
                perturbed_ratings[i, j] = np.round(blp_mechanism(R[i, j], l, u, user_ep), 2)
                # print(perturbed_ratings[i, j])

    output_file = 'ml-10m/InputPerturbation/BoundDP_ep'+str(epsilon)+'.dat'
    save_perturbed_ratings(perturbed_ratings, output_file)
    # Run ALS
    # P, Q = als_explicit(R_train, K=10, lambda_reg=0.017)
    #
    # R_predicted = predict_ratings(P, Q)
    #
    # topN = 10

    # rmse = compute_rmse(R_test, R_predicted)
    # print("RMSE on Test Set:", rmse)

