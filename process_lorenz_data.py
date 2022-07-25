"""
Import local stable manifold for Lorenz at the origin from the MATLAB file and save into csv. First origin_local_data.mat
is downloaded from the analytic continuation github, The variable "P" in that worksapce is a (51,51,3) interval matrix. The
midpoint of that matrix is saved to lorenz_local_parm.mat

    Output: output
    Other files required: none

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 1/17/21; Last revision: 1/17/21
"""

import scipy.io
import numpy as np
import pandas as pd

mat = scipy.io.loadmat('lorenz_local_parm.mat')  # load MATLAB data file
P = np.shape(mat['midP'])  # convert to numpy array


# mat = {k:v for k, v in mat.items() if k[0] != '_'}
# data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
# data.to_csv("lorenz_origin_local_manifold.csv")