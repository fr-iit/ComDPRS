import numpy as np
import DataLoader as DL
# import Evaluation as EV
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error
import Perturbation_Mechanism as PM

def comDP_pertubation(R, epsilon, k, m, y):
    l = 1.0
    u = 5.0
    sensitivity = u - l
    index = 1
    # k, m, y = PM.hybrid_parameter_optimization(epsilon, index, train, test)
    L = PM.LValue(epsilon, k, m, y, index)
    sensitivity_Cp = PM.sensitivity_Cp_fun(epsilon, k, m, y, index)

    n_users = R.shape[0]
    n_items = R.shape[1]

    perturbed_ratings = np.zeros_like(R)
    for i in range(n_users):

        num_ratings = np.count_nonzero(R[i, :])
        user_ep = epsilon / num_ratings
        print('user: ', i)
        for j in range(n_items):
            if R[i, j] > 0:
                # perturbed_ratings[i, j] = np.round(blp_mechanism(R[i, j], l, u, epsilon))
                perturbed_ratings[i, j] = np.round(PM.perturbation_fun_optimized
                                                   (user_ep, R[i, j], sensitivity, l, u, index, k, m, y, L, sensitivity_Cp))

    return perturbed_ratings


def save_perturbed_ratings(R, filename):
    with open(filename, 'w') as f:
        for user_id in range(R.shape[0]):
            for item_id in range(R.shape[1]):
                if R[user_id, item_id] > 0:  # Only save rated items
                    f.write(f"{user_id + 1}::{item_id + 1}::{R[user_id, item_id]}::000000000\n")

    print(f"Perturbed ratings saved to {filename}")


# Main code to load data, apply perturbation, and save results
if __name__ == "__main__":

    R = DL.load_user_item_matrix_1m()
    # train, test = DL.split_data(R)

    # Set the privacy budget (epsilon)
    epsilon = 0.1

    # hyper params
    k=0.0236
    m=0.9380
    y=0.3093

    # Apply the comDP perturbation
    perturbed_ratings = comDP_pertubation(R, epsilon, k, m, y)

    # Save the perturbed ratings to a file
    save_perturbed_ratings(perturbed_ratings, "Data/ml-1m/Com_DP/TVariance_" + str(epsilon) + ".dat")
