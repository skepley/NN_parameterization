"""
Make plots of the first few coefficients for the logistic function
    Output: output
    Other files required: none
    See also: OTHER_SCRIPT_NAME,  OTHER_FUNCTION_NAME

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 1/16/21; Last revision: 1/16/21
"""
# IMPORT MAP TO LEARN
import numpy as np
from logistic_coefficient_map import logistic_coefficient_map, manifold_map
import matplotlib.pyplot as plt

plt.close('all')
K = np.array([-1, 2])
withTime = False  # specify if NN should try to learn the timestep as well or fix it
fig, ax = plt.subplots()
N = 10
imgSize, TaylorCoefficientMap = manifold_map(logistic_coefficient_map, N, withTime)
plt_list = []
x = np.linspace(*K, 200)
tau_singular = np.argwhere((x == 0.5) | (x == 1.0) | (x == 0))
x = np.delete(x, tau_singular)
y = np.array([TaylorCoefficientMap(val) for val in x])
for idx in range(N):
    N_plot, = ax.plot(x, y[:, idx], linewidth=1, label=r"$a_{0}$".format(idx))
    plt_list.append(N_plot)

ax.set(xlabel=r"$x$", ylabel=R"$a_j(x)$")
ax.legend(ncol=2, loc='lower right', shadow=True)
plt.show()
fig.savefig('/users/shane/dropbox/0DiscreteConvolutionAI/coef_plots.png')
