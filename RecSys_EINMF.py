# einmf.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import DataLoader as DL
from sklearn.metrics import mean_squared_error
import sys
import math
import random


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def matrix_to_df(mat):
    rows, cols = np.nonzero(mat)
    ratings = mat[rows, cols]
    return pd.DataFrame(
        {'user': rows.astype(np.int64),
         'item': cols.astype(np.int64),
         'rating': ratings.astype(np.float32)}
    )


def init_weights(m):
    if isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, data, num_users, num_items):
        self.users = data['user'].values
        self.items = data['item'].values
        self.ratings = data['rating'].values
        self.implicit = (self.ratings > 0).astype(np.float32)  # Binary implicit

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float32),
            torch.tensor(self.implicit[idx], dtype=torch.float32)
        )


class EINMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(EINMF, self).__init__()
        # Explicit and Implicit Embeddings
        self.user_explicit = nn.Embedding(num_users, embedding_dim)
        self.user_implicit = nn.Embedding(num_users, embedding_dim)
        self.item_explicit = nn.Embedding(num_items, embedding_dim)
        self.item_implicit = nn.Embedding(num_items, embedding_dim)

        # MLP for deep interaction
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.output_layer = nn.Linear(32 + embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        # Embedding Lookups
        u_e = self.user_explicit(user)
        u_i = self.user_implicit(user)
        i_e = self.item_explicit(item)
        i_i = self.item_implicit(item)

        shallow = (u_e + u_i) * (i_e + i_i)  # element-wise
        shallow_score = torch.sum(shallow, dim=1)

        deep_input = torch.cat([u_e + u_i, i_e + i_i], dim=1)
        deep_feature = self.mlp(deep_input)

        final_input = torch.cat([deep_feature, shallow], dim=1)
        output = self.output_layer(final_input)
        return self.sigmoid(output).squeeze()


def train_model(model, train_loader, val_loader, epochs, lr=0.0001, eta=0.6, patience=3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for user, item, rating, implicit in train_loader:
            optimizer.zero_grad()
            preds = model(user, item)
            rating_norm = rating / 5.0
            loss = eta * bce(preds, implicit) + (1 - eta) * bce(preds, rating_norm)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for user, item, rating, implicit in val_loader:
                preds = model(user, item)
                rating_norm = rating / 5.0
                loss = eta * bce(preds, implicit) + (1 - eta) * bce(preds, rating_norm)
                total_val_loss += loss.item()

        print(f"Epoch {epoch + 1} | Train Loss: {total_train_loss:.4f} | Val Loss: {total_val_loss:.4f}")

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_epoch = epoch
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best epoch was {best_epoch + 1} with val loss {best_val_loss:.4f}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)


def evaluate_hit_at_10(model, data, num_users, num_items):
    model.eval()
    hits = 0
    for user_id in range(num_users):
        # Items the user interacted with
        user_data = data[data['user'] == user_id]
        interacted_items = set(user_data['item'].values)

        if len(interacted_items) == 0:
            continue

        # Score all items
        items = torch.arange(num_items)
        user = torch.tensor([user_id] * num_items)
        with torch.no_grad():
            scores = model(user, items).numpy()

        top_items = np.argsort(scores)[-10:]
        hits += any(item in interacted_items for item in top_items)

    print(f"Hit@10: {hits / num_users:.4f}")

def evaluate_hit_at_10_1plusRandom(model, data, train_data, num_users, num_items, top_k=10, candidate_fraction=0.2):
    model.eval()
    hit_count = 0
    test_count = 0
    num_candidates = int(num_items * candidate_fraction)

    for user_id in range(num_users):

        user_test_items = data[data['user'] == user_id]['item'].values
        if len(user_test_items) == 0:
            continue

        user_train_items = set(train_data[train_data['user'] == user_id]['item'].values)
        all_items = np.arange(num_items)

        for true_item in user_test_items:

            candidate_pool = np.setdiff1d(all_items, list(user_train_items))

            if len(candidate_pool) < num_candidates - 1:
                continue  # Skip if not enough negatives

            negatives = np.random.choice(candidate_pool, size=num_candidates - 1, replace=False)
            candidates = np.concatenate(([true_item], negatives))

            # Compute model scores
            user_tensor = torch.tensor([user_id] * len(candidates))
            item_tensor = torch.tensor(candidates)
            with torch.no_grad():
                scores = model(user_tensor, item_tensor).numpy()

            top_indices = np.argsort(scores)[-top_k:]
            top_items = candidates[top_indices]

            if true_item in top_items:
                hit_count += 1
            test_count += 1

    hr_at_k = hit_count / test_count if test_count > 0 else 0
    print(f"Hit@{top_k} (1+Random): {hr_at_k:.4f}")
    return hr_at_k


def evaluate_rmse(model, data):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for _, row in data.iterrows():
            user = torch.tensor([int(row['user'])], dtype=torch.long)
            item = torch.tensor([int(row['item'])], dtype=torch.long)
            rating = float(row['rating'])

            pred = model(user, item).item()
            pred = pred * 5.0  # De-normalize prediction back to [0, 5]
            y_true.append(rating)
            y_pred.append(pred)

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(f"RMSE: {rmse:.4f}")

def normalized_mse(model, val_loader, max_rating=5.0):
    model.eval()
    mse_loss = nn.MSELoss()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for user, item, rating, _ in val_loader:
            preds = model(user, item)
            rating_norm = rating / max_rating
            loss = mse_loss(preds, rating_norm)
            total_loss += loss.item() * len(user)
            count += len(user)

    nmse = total_loss / count
    print(f"Normalized MSE: {nmse:.4f}")
    # return nmse

def mse(model, val_loader):
    model.eval()
    mse_loss = nn.MSELoss()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for user, item, rating, _ in val_loader:
            preds = model(user, item) * 5.0  # De-normalize prediction to [0, 5]
            loss = mse_loss(preds, rating)
            total_loss += loss.item() * len(user)
            count += len(user)

    mse_score = total_loss / count
    print(f"MSE: {(mse_score):.4f}")
    # print(f"MSE/2: {(mse_score/2):.4f}")
    # return mse_score

def dcg(relevance_scores):
    return np.sum([
        (2**rel - 1) / np.log2(idx + 2)
        for idx, rel in enumerate(relevance_scores)
    ])

def ndcg_at_k(model, df, num_users, num_items, k=10):
    model.eval()
    total_ndcg = 0
    user_count = 0

    for user_id in range(num_users):
        user_data = df[df['user'] == user_id]
        if user_data.empty:
            continue

        true_items = user_data['item'].values
        true_ratings = user_data['rating'].values

        # Predict scores for all items
        items = torch.arange(num_items)
        user_tensor = torch.tensor([user_id] * num_items)
        with torch.no_grad():
            scores = model(user_tensor, items).numpy()

        # Get top-k predicted item indices
        top_k_indices = np.argsort(scores)[-k:][::-1]

        # Relevance of top-k items (0 if not in true, else rating)
        relevance = [user_data[user_data['item'] == item]['rating'].values[0] if item in true_items else 0
                     for item in top_k_indices]

        # Ideal relevance: top-k true ratings sorted descending
        ideal_relevance = sorted(true_ratings, reverse=True)[:k]

        dcg_score = dcg(relevance)
        idcg_score = dcg(ideal_relevance)

        ndcg = dcg_score / idcg_score if idcg_score > 0 else 0
        total_ndcg += ndcg
        user_count += 1

    average_ndcg = total_ndcg / user_count
    print(f"nDCG@{k}: {average_ndcg:.4f}")
    # return average_ndcg


def ndcg_at_k_1plusRandom(model, test_df, train_df, num_users, num_items, k=10, candidate_fraction=0.2):
    model.eval()
    total_ndcg = 0
    test_count = 0
    num_candidates = int(num_items * candidate_fraction)

    for user_id in range(num_users):
        # Get test items
        user_test_items = test_df[test_df['user'] == user_id]
        if user_test_items.empty:
            continue

        user_train_items = set(train_df[train_df['user'] == user_id]['item'].values)
        all_items = np.arange(num_items)

        for _, row in user_test_items.iterrows():
            true_item = int(row['item'])
            true_rating = float(row['rating'])

            candidate_pool = np.setdiff1d(all_items, list(user_train_items))
            if len(candidate_pool) < num_candidates - 1:
                continue  # Skip if not enough negatives

            negatives = np.random.choice(candidate_pool, size=num_candidates - 1, replace=False)
            candidates = np.concatenate(([true_item], negatives))

            # Predict scores
            user_tensor = torch.tensor([user_id] * len(candidates))
            item_tensor = torch.tensor(candidates)
            with torch.no_grad():
                scores = model(user_tensor, item_tensor).numpy()

            # Rank top-k items
            top_k_indices = np.argsort(scores)[-k:][::-1]
            top_k_items = candidates[top_k_indices]

            relevance = [true_rating if item == true_item else 0 for item in top_k_items]
            ideal_relevance = [true_rating] + [0] * (k - 1)

            dcg_score = dcg(relevance)
            idcg_score = dcg(ideal_relevance)

            ndcg = dcg_score / idcg_score if idcg_score > 0 else 0
            total_ndcg += ndcg
            test_count += 1

    avg_ndcg = total_ndcg / test_count if test_count > 0 else 0
    print(f"nDCG@{k} (1+Random): {avg_ndcg:.4f}")
    return avg_ndcg

if __name__ == "__main__":
    R = DL.load_user_item_matrix_1m_DP()  # shape: (num_users, num_items)
    trainR, testR = DL.split_data(R)  # same shape, zeros where held-out
    num_users, num_items = R.shape

    train_df = matrix_to_df(trainR)
    val_df = matrix_to_df(testR)

    train_dataset = InteractionDataset(train_df, num_users, num_items)
    val_dataset = InteractionDataset(val_df, num_users, num_items)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128)

    model = EINMF(num_users, num_items)

    train_model(model, train_loader, val_loader, epochs=100, eta=0.6, patience=5)
    mse(model, val_loader)
    evaluate_hit_at_10(model, val_df, num_users, num_items)
    ndcg_at_k(model, val_df, num_users, num_items, k=10)