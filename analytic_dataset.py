from torch.utils.data import Dataset
from logistic_coefficient_map import logistic_taylor_orbit, tau_from_last_coef, parameterization_manifold_map
import torch
import numpy as np
from parameterization_tools import ezcat


class AnalyticDataset(Dataset):
    def __init__(self, dataSetSize, dataSampler, outputSize) -> None:
        super().__init__()
        self.size = dataSetSize
        self.initial_values = dataSampler(dataSetSize)
        self.N = outputSize - 1

        # generate training and testing orbit parameterizations
        NN_target_map = parameterization_manifold_map(logistic_taylor_orbit, self.N, tau_from_last_coef)
        orbitData = np.zeros((self.size, self.N + 1))
        for idx, x0 in enumerate(self.initial_values):
            tau, seq = NN_target_map(x0)
            orbitData[idx] = ezcat(tau, seq.coef)
        self.data = orbitData

    def __len__(self):
        return self.size

    def __getitem__(self, index, withTau=False):
        iv = np.array([self.initial_values[index]])
        if withTau:
            coefficients = self.data[index]
        else:
            coefficients = self.data[index][1:]

        iv = torch.from_numpy(iv.astype(np.float32))
        coefficients = torch.from_numpy(coefficients.astype(np.float32))

        return iv, coefficients
