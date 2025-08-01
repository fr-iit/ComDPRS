import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
# import MovieLensData as MD
# import RecSys_Evaluation as RSE
import pandas as pd
import DataLoader as DL


def split_train_test(R, test_size=0.2):

    num_users, num_items = R.shape
    train_data = []
    test_data = []

    for user in range(num_users):
        rated_items = np.nonzero(R[user, :].toarray())[1]  # Extract nonzero items
        if len(rated_items) > 9:
            train_items, test_items = train_test_split(rated_items, test_size=test_size, random_state=42)

            train_row = np.zeros(num_items)
            test_row = np.zeros(num_items)

            train_row[train_items] = R[user, train_items].toarray().flatten()
            test_row[test_items] = R[user, test_items].toarray().flatten()

            train_data.append(train_row)
            test_data.append(test_row)
        else:
            continue

    train_matrix = np.array(train_data)
    test_matrix = np.array(test_data)

    return csr_matrix(train_matrix), csr_matrix(test_matrix)

def compute_user_similarity(R_train):

    return cosine_similarity(R_train) #similarity_matrix #


def predict_userknn_rating(user_index, item_index, R_train, similarity_matrix, neighbor_count, threshold=0.15):

    user_similarity = similarity_matrix[user_index]

    rated_by_neighbors = R_train[:, item_index].nonzero()[0]  # Users who have rated this item
    neighbor_similarities = user_similarity[rated_by_neighbors]

    valid_neighbors = rated_by_neighbors[neighbor_similarities >= threshold]
    valid_similarities = neighbor_similarities[neighbor_similarities >= threshold]

    if len(valid_neighbors) == 0:
        return np.mean(R_train[user_index].toarray())  # Default to user's average rating

    top_k_neighbors = np.argsort(-valid_similarities)[:neighbor_count]  # Get top-k indices of valid neighbors

    numerator = 0
    denominator = 0
    for idx in top_k_neighbors:
        neighbor_index = valid_neighbors[idx]
        neighbor_rating = R_train[neighbor_index, item_index]

        if neighbor_rating > 0:
            similarity = valid_similarities[idx]
            numerator += similarity * neighbor_rating
            denominator += abs(similarity)

    if denominator > 0:
        return numerator / denominator
    else:
        return np.mean(R_train[user_index].toarray())


def evaluate_userknn(R_train, R_test, similarity_matrix, k=10):

    num_users, num_items = R_test.shape
    hit_count = 0
    total_count = 0
    candidate_item = int(num_items * 0.25)
    print(f'candidate_item: {candidate_item}')

    all_true_ratings = []
    all_predicted_ratings = []
    all_ndcg_scores = []

    for user in range(num_users):
        print(user)
        rated_items = R_test[user, :].nonzero()[1]
        if len(rated_items) == 0:
            continue

        for item in rated_items:

            predicted_rating = predict_userknn_rating(user, item, R_train, similarity_matrix, neighbor_count=30)

            true_rating = R_test[user, item]
            all_true_ratings.append(true_rating)
            all_predicted_ratings.append(predicted_rating)

    return 0, all_true_ratings, all_predicted_ratings, 0 # for only mae calculation

# --- HR: without 1+random
def evaluate_userknn_norandom(R_train, R_test, similarity_matrix, k):
    print(f'K' + str(k))
    num_users, num_items = R_test.shape
    hit_count = 0
    total_count = 0

    all_true_ratings = []
    all_predicted_ratings = []
    all_ndcg_scores = []

    for user in range(num_users):
        # print(f'user: {user}')
        rated_items = R_test[user, :].nonzero()[1]  # Extract the indices of rated items for this user
        if len(rated_items) == 0:
            continue  # Skip users with no rated items in the test set

        # Predict ratings for the rated items
        user_true_ratings = []
        user_predicted_ratings = []

        for item in rated_items:
            predicted_rating = predict_userknn_rating(user, item, R_train, similarity_matrix, neighbor_count=30)

            true_rating = R_test[user, item]
            user_true_ratings.append(true_rating)
            user_predicted_ratings.append(predicted_rating)

            all_true_ratings.append(true_rating)
            all_predicted_ratings.append(predicted_rating)

        # Calculate nDCG for this user based on the true and predicted ratings
        ndcg_score = calculate_ndcg(user_true_ratings, user_predicted_ratings, k=k)
        all_ndcg_scores.append(ndcg_score)

        # Calculate HitRate for the top-k items
        top_k_items = np.argsort(user_predicted_ratings)[-k:]  # Get indices of top-k predicted items
        if 0 in top_k_items:  # Check if the true item is in the top-k predictions
            hit_count += 1
        total_count += 1

        # print(f'user: {user}, hit count: {hit_count}, total_count: {total_count}')
    hit_rate_at_k = hit_count / total_count if total_count > 0 else 0

    return hit_rate_at_k, all_true_ratings, all_predicted_ratings, all_ndcg_scores


def calculate_nmae(true_ratings, predicted_ratings, max = 5, min = 1):
    true_ratings = np.array(true_ratings)
    predicted_ratings = np.array(predicted_ratings)
    mae = np.mean(np.abs(true_ratings - predicted_ratings))
    # nmae = mae/(max-min)
    return mae


def calculate_rmse(true_ratings, predicted_ratings):
    true_ratings = np.array(true_ratings)
    predicted_ratings = np.array(predicted_ratings)

    mse = np.mean((true_ratings - predicted_ratings) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def calculate_ndcg(true_ratings, predicted_scores, k=10):

    sorted_indices = np.argsort(predicted_scores)[::-1]
    sorted_true_ratings = np.take(true_ratings, sorted_indices[:k])

    dcg = 0.0
    for i in range(len(sorted_true_ratings)):
        dcg += (2 ** sorted_true_ratings[i] - 1) / np.log2(i + 2)  # log2(i+2) to avoid division by 0

    sorted_ideal_ratings = np.sort(true_ratings)[::-1][:k]
    idcg = 0.0
    for i in range(len(sorted_ideal_ratings)):
        idcg += (2 ** sorted_ideal_ratings[i] - 1) / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg  # Normalized DCG


def evaluate_userknn_1plusRandom(R_train, R_test, similarity_matrix, k=10, candidate_fraction=0.2, neighbor_count=30):
    num_users, num_items = R_test.shape
    hit_count = 0
    total_count = 0
    ndcg_scores = []

    num_candidates = int(num_items * candidate_fraction)

    for user in range(num_users):
        print(user)
        test_items = R_test[user, :].nonzero()[1]
        if len(test_items) == 0:
            continue

        train_items = set(R_train[user, :].nonzero()[1])
        all_items = np.arange(num_items)

        for true_item in test_items:
            # Sample negative items that the user did not rate in training
            candidate_pool = np.setdiff1d(all_items, list(train_items))
            if len(candidate_pool) < num_candidates - 1:
                continue  # Skip if not enough negatives

            negatives = np.random.choice(candidate_pool, size=num_candidates - 1, replace=False)
            candidates = np.concatenate(([true_item], negatives))

            # Predict scores for all candidate items
            predicted_scores = []
            true_relevance = []

            for item in candidates:
                score = predict_userknn_rating(user, item, R_train, similarity_matrix, neighbor_count=neighbor_count)
                predicted_scores.append(score)
                true_relevance.append(R_test[user, item] if item == true_item else 0)

            # Compute Hit@k
            top_k_indices = np.argsort(predicted_scores)[-k:]
            if 0 in top_k_indices:  # true_item is at index 0 in candidates
                hit_count += 1
            total_count += 1

            # Compute nDCG@k
            ndcg = calculate_ndcg(true_relevance, predicted_scores, k)
            ndcg_scores.append(ndcg)

    hit_rate = hit_count / total_count if total_count > 0 else 0
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0

    return hit_rate, avg_ndcg



# Main logic to run UserKNN

dataset = '1m'

# Load example data
if dataset == '100k':
    user_item_matrix = DL.load_user_item_matrix_100k_DP() #  DL.load_user_item_matrix_100k()
elif dataset == '1m':
    user_item_matrix = DL.load_user_item_matrix_1m_DP()
elif dataset == 'yahoo':
    user_item_matrix =  DL.load_user_item_matrix_yahoo_DP() #DL.load_user_item_matrix_yahoo() #DL.load_user_item_matrix_yahoo_masked()#
elif dataset == 'beauty':
    user_item_matrix = DL.load_user_item_matrix_beauty_DP()  #  DL.load_user_item_matrix_beauty() #DL.load_user_item_matrix_yahoo_masked()#


user_item_matrix = csr_matrix(user_item_matrix)

# Split the data
train_matrix, test_matrix = split_train_test(user_item_matrix)

similarity_matrix = compute_user_similarity(train_matrix)
k_neighbors = 30
hit_rate, true_rate, predicted_rate, nDCG = evaluate_userknn_norandom(train_matrix, test_matrix, similarity_matrix, k=k_neighbors)
mae = calculate_nmae(true_rate, predicted_rate)
average_ndcg = np.mean(nDCG)

print(f'top-{k_neighbors}')
print(dataset)
print(f'MAE: {mae:.4f}')
print(f'HR: {(hit_rate):.4f}, nDCG: {(average_ndcg):.4f}')
