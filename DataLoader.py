import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

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

# Function to split the data into train and test sets
def split_data_save(R, ep, name, test_size=0.2):
    train = R.copy()
    test = np.zeros_like(R)

    countR = np.count_nonzero(R)

    for user_idx in range(R.shape[0]):
        non_zero_indices = np.where(R[user_idx] > 0)[0]
        test_indices = np.random.choice(non_zero_indices,
                                         size=int(test_size * len(non_zero_indices)),
                                         replace=False)
        train[user_idx, test_indices] = 0
        test[user_idx, test_indices] = R[user_idx, test_indices]

    if name == 'Bound':
        print('saving: ', name)
        with open('ml-1m/SplitData/'+str(name)+'TrainData'+str(ep)+'.dat', 'w', newline='') as train_file:
            writer = csv.writer(train_file)
            writer.writerows(train)

        with open('ml-1m/SplitData/'+str(name)+'TestData'+str(ep)+'.dat', 'w', newline='') as test_file:
            writer = csv.writer(test_file)
            writer.writerows(test)
    elif name == 'Unbound':
        print('saving: ', name)
        with open('ml-1m/SplitData/'+str(name)+'TrainData'+str(ep)+'.dat', 'w', newline='') as train_file:
            writer = csv.writer(train_file)
            writer.writerows(train)

        with open('ml-1m/SplitData/'+str(name)+'TestData'+str(ep)+'.dat', 'w', newline='') as test_file:
            writer = csv.writer(test_file)
            writer.writerows(test)
    else:
        print('saving: ', name)
        with open('ml-1m/SplitData/TrainData.dat', 'w', newline='') as train_file:
            writer = csv.writer(train_file)
            writer.writerows(train)

        with open('ml-1m/SplitData/TestData.dat', 'w', newline='') as test_file:
            writer = csv.writer(test_file)
            writer.writerows(test)

    # Count non-zero entries in train and test datasets
    train_non_zero = np.count_nonzero(train)
    test_non_zero = np.count_nonzero(test)

    print(f"Data has been split: {len(train)} rows for training and {len(test)} rows for testing.")
    print("Non-zero entries in training dataset:", train_non_zero)
    print("Non-zero entries in testing dataset:", test_non_zero)
    print("Non-zero entries in dataset:", countR)
    # return train, test

def save_test_data(test_data, filename):

    with open(filename, "w") as f:
        for user, items in test_data.items():
            line = f"{user}," + ",".join(map(str, items))
            f.write(line + "\n")
    print(f"Test data saved to {filename}")

def GroundTruth_ranking(test_matrix):
    test_dict = {}
    for user_id in range(test_matrix.shape[0]):  # Iterate through users
        interacted_items = np.where(test_matrix[user_id] > 0)[0]  # Non-zero ratings
        if len(interacted_items) > 0:
            # test_dict[user_id] = set(interacted_items)
            test_dict[user_id] = {str(item) for item in interacted_items}
    return test_dict

def save_topN_recommendations(R_predicted, filename, topN):
    filewrite = 'ml-10m/TopN-RecommendedList/'+filename
    with open(filewrite, mode='w', newline='') as file:
        writer = csv.writer(file)

        for user_idx in range(R_predicted.shape[0]):
            # Get top-10 recommended item indices sorted by highest predicted rating
            topN_items = np.argsort(R_predicted[user_idx])[-topN:][::-1]

            # Write to CSV (user_id, top10_item1, top10_item2, ...)
            writer.writerow([user_idx] + topN_items.tolist())

    print(f"Top-N recommendations saved to {filewrite}")

# dataset: yahoo
def load_user_item_matrix_yahoo():

    movies = set()  # Using set to automatically deduplicate
    users = set()  # Using set to automatically deduplicate
    ratings = []

    with open('Data/ml-yahoo/yahoo_mergerating.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header

        for row in reader:
            userid, movieid, Rating, Genres, gender = row

            user_id = int(userid)
            movie_id = int(movieid)
            rating = float(Rating)

            movies.add(movie_id)
            users.add(user_id)
            ratings.append((user_id, movie_id, rating))

    # Map unique user and movie IDs to their indices
    num_unique_users = len(users)
    num_unique_movies = len(movies)

    print(f'Number of unique users: {num_unique_users}, Number of unique movies: {num_unique_movies}')

    # Create a user-item matrix with dimensions (num_users, num_movies)
    df = np.zeros(shape=(num_unique_users, num_unique_movies))

    # Fill the matrix with ratings
    for user_id, movie_id, rating in ratings:
        df[user_id - 1, movie_id - 1] = rating  # Subtracting 1 to align with 0-indexing

    # Calculate density
    count_non_zero = np.count_nonzero(df)
    density = (count_non_zero / df.size) * 100

    print(f'Yahoo Data Density: {density:.2f}%')

    return df

def load_UIM_yahoo_implicit():

    movies = set()  # Using set to automatically deduplicate
    users = set()  # Using set to automatically deduplicate
    ratings = []

    with open('Data/ml-yahoo/yahoo_mergerating.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header

        for row in reader:
            userid, movieid, Rating, Genres, gender = row

            user_id = int(userid)
            movie_id = int(movieid)
            rating = float(Rating)

            movies.add(movie_id)
            users.add(user_id)
            ratings.append((user_id, movie_id, rating))

    # Map unique user and movie IDs to their indices
    num_unique_users = len(users)
    num_unique_movies = len(movies)

    print(f'Number of unique users: {num_unique_users}, Number of unique movies: {num_unique_movies}')

    # Create a user-item matrix with dimensions (num_users, num_movies)
    df = np.zeros(shape=(num_unique_users, num_unique_movies))

    # Fill the matrix with ratings
    for user_id, movie_id, rating in ratings:
        df[user_id - 1, movie_id - 1] = 1.0  # Subtracting 1 to align with 0-indexing

    # Calculate density
    count_non_zero = np.count_nonzero(df)
    density = (count_non_zero / df.size) * 100

    print(f'Yahoo Data Density: {density:.2f}%')

    return df

def load_user_item_matrix_yahoo_DP(max_user=2837, max_item=8584):

    df = np.zeros(shape=(max_user, max_item))
    print('original data')
    with open("Data/ml-yahoo/Com_DP/HybridOP_5.0.dat", 'r') as f: #u.data u1.base
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id - 1, movie_id - 1] = rating

    return df



# --- end yahoo

# dataset: ml100k
def load_user_item_matrix_100k(max_user=943, max_item=1682):

    df = np.zeros(shape=(max_user, max_item))
    print('original data')
    with open("Data/ml-100k/u.data", 'r') as f: #u.data u1.base
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split()
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            df[user_id-1, movie_id-1] = rating

    return df


def load_user_item_matrix_100k_DP(max_user=943, max_item=1682):

    df = np.zeros(shape=(max_user, max_item))
    print('original data')
    with open("Data/ml-100k/Com_DP/BoundDP_ep5.0.dat", 'r') as f: #u.data u1.base
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id - 1, movie_id - 1] = rating

    return df


def load_UIM_100k_implicit(max_user=943, max_item=1682):

    df = np.zeros(shape=(max_user, max_item))
    print('original data')
    with open("Data/ml-100k/u.data", 'r') as f: #u.data u1.base
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split()
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if rating > 0:
                df[user_id-1, movie_id-1] = 1.0

    return df

# --- end ml100k

# dataset: ml1m
def load_user_item_matrix_1m(max_user=6040, max_item=3952):

    df = np.zeros(shape=(max_user, max_item))
    with open("Data/ml-1m/ratings.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df

def load_user_item_matrix_1m_DP(max_user=6040, max_item=3952):

    df = np.zeros(shape=(max_user, max_item))
    with open("Data/ml-1m/Com_DP/TVariance_5.0.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df

def load_UIM_1m_implicit(max_user=6040, max_item=3952):

    df = np.zeros(shape=(max_user, max_item))
    with open("ml-1m/ratings.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item and rating > 0:
                df[user_id-1, movie_id-1] = 1.0

    return df

# --- end ml1m

# --- start ml-10m
def load_user_item_matrix_10m(max_user=72000, max_item=10000):

    df = np.zeros(shape=(max_user, max_item))
    with open("ml-10m/ratings.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating
    print(df.shape)
    return df
# --- end ml-10m

# --- beauty
def load_user_item_matrix_beauty(max_user=719, max_item=1142):

    df = np.zeros(shape=(max_user, max_item))
    with open("Data/ml-beauty/rating.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating = line.split("\t")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df

def load_user_item_matrix_beauty_DP(max_user=719, max_item=1142):

    df = np.zeros(shape=(max_user, max_item))
    with open("Data/ml-beauty/Com_DP/HybridOP_5.0.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df


def load_UIM_beauty_implicit(max_user=719, max_item=1142):
    df = np.zeros(shape=(max_user, max_item))
    with open("Data/ml-beauty/rating.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("\t")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)

            if user_id <= max_user and movie_id <= max_item and rating > 0:
                # Convert to implicit: 1 if rated, 0 otherwise
                df[user_id - 1, movie_id - 1] = 1.0  # Interaction occurred

    return df

# --- end beauty

def DensityCount(data = 'beauty'):

    if data == '1m':
        X = load_user_item_matrix_1m()
    elif data == '100k':
        X = load_user_item_matrix_100k()
    elif data == 'yahoo':
        X = load_user_item_matrix_yahoo()
        print(f'user: {X.shape[0]}, items: {X.shape[1]}')
    elif data == '10m':
        X = load_user_item_matrix_10m()
    elif data == 'beauty':
        X = load_user_item_matrix_beauty()
        print(f'user: {X.shape[0]}, items: {X.shape[1]}')

    total_entries = X.shape[0] * X.shape[1]
    no_of_ratings = np.count_nonzero(X)
    # print('rating no: ', no_of_ratings)
    density = (no_of_ratings/total_entries) * 100

    # obs_total_entries = X_obs.shape[0] * X_obs.shape[1]
    # obs_ratings_no = np.count_nonzero(X_obs)
    # obs_density = (obs_ratings_no/obs_total_entries) * 100

    # print(f"data: {data} , density: {density}, Obs_density: {obs_density}")
    print(f"data: {data} , density: {density}")
    print(f'no_of_ratings: {no_of_ratings}')

def find_max_min_rating_users(rating_matrix):

    # Count the number of non-zero ratings for each user
    user_ratings_count = np.count_nonzero(rating_matrix, axis=1)

    # Find the user with the maximum and minimum ratings
    max_user = np.argmax(user_ratings_count)  # User index with the most ratings
    max_ratings_count = user_ratings_count[max_user]

    min_user = np.argmin(user_ratings_count)  # User index with the least ratings
    min_ratings_count = user_ratings_count[min_user]



    # Find the minimum number of ratings
    min_rating_count = np.min(user_ratings_count)
    max_rating_count = np.max(user_ratings_count)

    # Count how many users have the minimum number of ratings
    # user_count_min = np.sum(user_ratings_count == min_rating_count)
    user_count_min = np.sum(user_ratings_count <= 100)
    user_count_max = np.sum(user_ratings_count > 100)

    print(f"User with maximum ratings: User {max_user}, Number of ratings: {max_ratings_count}")
    print(f"User with minimum ratings: User {min_user}, Number of ratings: {min_ratings_count}")
    print(f'number of user who have min rating: {user_count_min}')
    print(f'number of user who have max rating: {user_count_max}')

    # return max_user, max_ratings_count, min_user, min_ratings_count

def identify_cold_start_users(rating_matrix, cold_start_threshold=0.05):

    # Count the number of non-zero ratings for each user
    user_ratings_count = np.count_nonzero(rating_matrix, axis=1)

    # Find the maximum number of ratings by any user
    max_ratings = np.max(user_ratings_count)

    print(f'max_ratings: {max_ratings}')

    # Define the threshold for cold-start users
    cold_start_limit = cold_start_threshold * max_ratings

    print(f'cold_start_limit: {cold_start_limit}')

    # Identify cold-start users
    cold_start_users = [user for user, count in enumerate(user_ratings_count) if count <= cold_start_limit]

    # Identify normal users
    normal_users = [user for user, count in enumerate(user_ratings_count) if count > cold_start_limit]

    # Count the number of cold-start users
    cold_start_count = len(cold_start_users)

    print(f"Cold-Start Users: {len(cold_start_users)}")
    print(f"Normal Users: {len(normal_users)}")
    # print(f"Count of Cold-Start Users: {cold_start_count}")

    return cold_start_users, normal_users #, cold_start_count

def matrix_to_interaction_data(user_item_matrix):

    user_ids, item_ids = np.where(user_item_matrix > 0)  # Find non-zero interactions
    data = {
        "user_id": user_ids + 1,  # Convert to 1-based indexing
        "item_id": item_ids + 1  # Convert to 1-based indexing
    }
    return pd.DataFrame(data)


def user_profile(R):

    num_users = R.shape[0]
    user_profiles = [np.where(R[u] > 0)[0] for u in range(num_users)]
    return user_profiles

########## call functions

# load_gender_vector_100k()
# load_gender_vector_1m()
DensityCount()

# find_max_min_rating_users(R)
# identify_cold_start_users(R)

# output_file = "ml-1m/item_popularity.dat"
# R = load_user_item_matrix_100k()
# find_max_min_rating_users(R)
# train, test = save_split(R, 0.2)
# save_test_data(test, filename="ml-yahoo/test_data.dat")
# interaction = matrix_to_interaction_data(R)
# calculate_and_save_item_popularity(interactioload_user_item_matrix_100k_Pertubation(n, output_file)


# -------------------------------------------
# user_item_matrix = load_user_item_matrix_100k_Pertubation() # load_user_item_matrix_100k() #
# max_ratings = user_item_matrix.max(axis=1)
# max_rating_in_system = user_item_matrix.max()
# # Now, find the users who have this maximum rating
# users_with_max_rating = np.where(user_item_matrix == max_rating_in_system)[0]
#
# # Extract ratings for these users
# ratings_for_max_users = user_item_matrix[users_with_max_rating].flatten()
# #
# # Step 1: Flatten the user-item matrix to get all ratings
# all_ratings = user_item_matrix[user_item_matrix > 0].flatten()
#
# unique_ratings, counts = np.unique(all_ratings, return_counts=True)
#
# # Print each rating and its corresponding count
# for rating, count in zip(unique_ratings, counts):
#     print(f"Rating: {rating}, Count: {count}")

# plt.figure(figsize=(8, 6))
# plt.hist(all_ratings, bins=np.arange(1, 7) - 0.5, edgecolor='black', alpha=0.7)
# plt.title('Rating Distribution for Users with Maximum Rating in the System')
# plt.xlabel('Rating')
# plt.ylabel('Frequency')
# plt.xticks(range(1, 6))
# plt.grid(True)
# plt.show()