import numpy as np
import DataLoader as DL
import Perturbation_Mechanism as PM

np.random.seed(42)  # For reproducibility

# ------- function call -------------
R = DL.load_user_item_matrix_beauty()
train, test = DL.split_data(R)
epsilon = 3.0
index = 1
k, m, y = PM.op_hybrid_parameter_optimization(epsilon, index, train, test)

# k, m, y = PM.parameter_optimization(epsilon, index)

print('k :', k)
print('m :', m)
print('y :', y)