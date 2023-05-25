"""
Make plots of the tau map for the logistic function
    Output: output
    Other files required: none
    See also: OTHER_SCRIPT_NAME,  OTHER_FUNCTION_NAME
   
    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 1/16/21; Last revision: 1/16/21
"""
# IMPORT MAP TO LEARN
import numpy as np
from logistic_map import logistic_taylor_orbit, parameterization_manifold_map, tau_from_last_coef
import matplotlib.pyplot as plt

# matplotlib.rcParams['text.usetex'] = True


plt.close('all')

N_values = np.array([10, 20, 30, 40, 50, 75, 100])
# N_values = np.array([10, 20])

K = np.array([-1, 2])
withTime = True  # specify if NN should try to learn the timestep as well or fix it
fig, ax = plt.subplots()
plt_list = []
for N in N_values:
    print(N)
    TaylorCoefficientMap = parameterization_manifold_map(logistic_taylor_orbit, N, tau_from_last_coef)
    # sample tau map
    x = np.linspace(*K, 1000)
    tau_singular = np.argwhere((x == 0.5) | (x == 1.0) | (x == 0))
    x = np.delete(x, tau_singular)
    y = np.array([TaylorCoefficientMap(val) for val in x])
    tau = y[:, -1]
    N_plot, = ax.plot(x, tau, linewidth=1.0, label=r"$N = {0}$".format(N))
    plt_list.append(N_plot)

ax.set(xlabel=r"$x$", ylabel=R"$\tau_\mu$")
# ax.legend((muPlot for muPlot in plt_list), ('mu = ' + str(N) for N in N_values), loc='upper right', shadow=True)
ax.legend(loc='upper right', shadow=True)
plt.show()
# fig.savefig('/users/shane/dropbox/0DiscreteConvolutionAI/logistic_tau_plot.png')
