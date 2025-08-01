import numpy as np
from scipy.sparse import csr_matrix
import DataLoader as DL
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random

def als_explicit(R, num_factors=10, regularization=0.1, num_iterations=20):
    num_users, num_items = R.shape
    P = np.random.rand(num_users, num_factors)
    Q = np.random.rand(num_items, num_factors)

    for iteration in range(num_iterations):
        for u in range(num_users):
            rated_items = np.nonzero(R[u, :])[0]
            if len(rated_items) > 0:
                Q_rated = Q[rated_items, :]
                R_u = R[u, rated_items].flatten()
                A = Q_rated.T @ Q_rated + regularization * np.eye(num_factors)
                b = Q_rated.T @ R_u
                P[u, :] = np.linalg.solve(A, b)

        for i in range(num_items):
            rated_users = np.nonzero(R[:, i])[0]
            if len(rated_users) > 0:
                P_rated = P[rated_users, :]
                R_i = R[rated_users, i].flatten()
                A = P_rated.T @ P_rated + regularization * np.eye(num_factors)
                b = P_rated.T @ R_i
                Q[i, :] = np.linalg.solve(A, b)

        # Compute RMSE for current iteration
        prediction = P @ Q.T
        non_zero_indices = R.nonzero()
        actual = R[non_zero_indices]
        predicted = prediction[non_zero_indices]
        error = np.sqrt(np.mean((actual - predicted) ** 2))
        print(f"Iteration {iteration + 1}/{num_iterations}, RMSE: {error:.4f}")

    return P, Q

def als_predict_func(P, Q):
    def predict(user_indices, item_indices):
        return np.array([P[u, :] @ Q[i, :].T for u, i in zip(user_indices, item_indices)])
    return predict

# evaluation

def calculate_nmae(true_ratings, predicted_ratings, min_val=1.0, max_val=5.0):
    mae = mean_absolute_error(true_ratings, predicted_ratings)
    # return mae / (max_val - min_val)
    return mae

def hit_rate(predicted_scores, test_items, k=10):
    hits = 0
    for pred, true_item in zip(predicted_scores, test_items):
        top_k_items = np.argsort(pred)[::-1][:k]
        if true_item in top_k_items:
            hits += 1
    return hits / len(test_items)

def ndcg(predicted_scores, test_items, k=10):
    def dcg(relevance):
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance))

    ndcg_total = 0.0
    for pred, true_item in zip(predicted_scores, test_items):
        top_k_items = np.argsort(pred)[::-1][:k]
        relevance = [1 if item == true_item else 0 for item in top_k_items]
        ideal_relevance = sorted(relevance, reverse=True)
        dcg_val = dcg(relevance)
        idcg_val = dcg(ideal_relevance)
        ndcg_total += (dcg_val / idcg_val) if idcg_val > 0 else 0
    return ndcg_total / len(test_items)

########################################
# ------- call function and load data

dataset = '1m'
Topk = 10

# Load example data
if dataset == '100k':
    Factors = 10
    Iterations = 20
    user_item_matrix = DL.load_user_item_matrix_100k_DP() # DL.load_user_item_matrix_100k() # --- this should be original data
elif dataset == '1m':
    Factors = 10
    Iterations = 20 #20
    user_item_matrix = DL.load_user_item_matrix_1m_DP()
    #  DL.load_user_item_matrix_1m()
elif dataset == 'yahoo':
    Factors = 10
    Iterations = 100 #20
    user_item_matrix = DL.load_user_item_matrix_yahoo_DP() #  DL.load_user_item_matrix_yahoo()
elif dataset == 'beauty':
    Factors = 40
    Iterations = 100
    user_item_matrix = DL.load_user_item_matrix_beauty_DP() #DL.load_user_item_matrix_beauty()

num_runs = 5
all_rmse, all_nmae, all_hr, all_ndcg = [], [], [], []

for run in range(num_runs):
    print(f"\n--- Run {run + 1}/{num_runs} ---")
    train_matrix, test_matrix = DL.split_data(user_item_matrix)

    P, Q = als_explicit(train_matrix, num_factors=Factors, regularization=0.1, num_iterations=Iterations)
    als_predict = als_predict_func(P, Q)

    true_ratings = []
    predicted_ratings = []

    user_indices, item_indices = test_matrix.nonzero()
    predicted_scores_matrix = P @ Q.T
    predicted_scores = []

    test_items_for_hr = []

    for u, i in zip(user_indices, item_indices):
        true_ratings.append(test_matrix[u, i])
        predicted_ratings.append(predicted_scores_matrix[u, i])

        # HR & nDCG
        user_row = train_matrix[u].flatten()
        unseen_items = np.where(user_row == 0)[0]
        if len(unseen_items) >= Topk:
            sampled_items = random.sample(list(unseen_items), Topk - 1)
            candidate_items = sampled_items + [i]
            random.shuffle(candidate_items)
            scores = [predicted_scores_matrix[u, j] for j in candidate_items]
            predicted_scores.append(scores)
            test_items_for_hr.append(i)

    rmse_val = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    nmae_val = calculate_nmae(true_ratings, predicted_ratings)
    hr_val = hit_rate(predicted_scores, test_items_for_hr, k=Topk)
    ndcg_val = ndcg(predicted_scores, test_items_for_hr, k=Topk)

    print(f"RMSE: {rmse_val:.4f}, nMAE: {nmae_val:.4f}, HR@{Topk}: {hr_val:.4f}, nDCG@{Topk}: {ndcg_val:.4f}")

    all_rmse.append(rmse_val)
    all_nmae.append(nmae_val)
    all_hr.append(hr_val)
    all_ndcg.append(ndcg_val)

# Final average results
print("\n===== Final Results (Averaged over 5 runs) =====")
print(dataset)
print(f"Avg MAE: {(nmae_val):.4f}")
print(f"Avg HR@{Topk}: {np.mean(all_hr):.4f}")
print(f"Avg nDCG@{Topk}: {np.mean(all_ndcg):.4f}")

