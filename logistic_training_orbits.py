import numpy as np
from logistic_coefficient_map import *

dataFileName = 'taylor_logistic_training_test.npz'  # N = 10, with 1000 training points

nTrainingOrbit = int(1e3)
nTestingOrbit = nTrainingOrbit // 10  # testing set is 10% of the training set size
N = 10
K = np.array([-1, 1])  # choose a compact subset of phase space to sample the parameterization manifold
NN_target_map = manifold_map(logistic_taylor_orbit, N, tau_from_last_coef)


# generate training and testing orbit parameterizations
def sample_parameterization_manifold(nOrbit, grid='random'):
    if grid == 'random':
        initialData = K[0] + np.diff(K) * np.random.rand(nOrbit, 1)
        orbitData = np.zeros((nOrbit, N + 1))
        for idx, x0 in enumerate(initialData):
            tau, seq = NN_target_map(x0)
            orbitData[idx] = ezcat(tau, seq.coef)

    elif grid == 'uniform':
        pass

    return initialData, orbitData


# generate testing data
x_train, y_train = sample_parameterization_manifold(nTrainingOrbit)
x_test, y_test = sample_parameterization_manifold(nTestingOrbit)

# save training data in npz format
np.savez(dataFileName, N, x_train, y_train, x_test, y_test)
