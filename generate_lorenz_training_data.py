"""
Sample initial conditions and generate coefficient data for training the Lorenz neural network
   
    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 1/20/21; Last revision: 1/20/21
"""
import numpy as np
from lorenz_coefficient_map import coefficient_map

dataFileName = 'lorenz_data_100_1e4.npz'  # N = 100, with 1e4 training points
nTrainingSet = int(1e5)
nTestingSet = nTrainingSet // 10  # generate additional data for testing set which is 10% of the training set size
nSample = nTrainingSet + nTestingSet
N = 100
K = np.array([[-10, 10], [-10, 10], [-10, 10]])
mu = np.finfo(float).eps  # set desired last coefficient for each segment to machine precision
imgSize = 3 * N + 1


def TaylorManifoldMap(x0, N, mu, initialTau=float(-1 / 10)):
    """Embedding of R^3 into a manifold in R^{3N + 1}"""
    coefData, tau = coefficient_map(x0, N, initialTau, lastCoefficientNorm=mu)
    return np.append(coefData.flatten(), tau)


# generate data and save training data in npz format
X = K[:, 0] + np.diff(K).squeeze() * np.random.rand(nSample, 3)  # initial data
Y = np.zeros((nSample, imgSize))  # images on the Taylor manifold
for i, x in enumerate(X):
    Y[i, :] = TaylorManifoldMap(x, N, mu)

np.savez(dataFileName, X, Y, nTrainingSet)
