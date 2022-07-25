"""
Implementation of the recursive parameterization method for the Lorenz example. This is used to generate data for training
the neural network

    Output: output
    Other files required: none
    See also: OTHER_SCRIPT_NAME,  OTHER_FUNCTION_NAME
   
    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 1/17/21; Last revision: 1/17/21
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D

plt.close('all')





def coefficient_gen(a, b, c, params):
    """The recursive Cauchy product terms for the lorenz flow coefficient map"""

    # compute nonlinear terms
    xz = np.dot(a, np.flip(c))
    xy = np.dot(a, np.flip(b))

    # compose vector field with taylor coefficients and project
    x = params[0] * (b[-1] - a[-1])
    y = params[1] * a[-1] - xz - b[-1]
    z = xy - params[2] * c[-1]
    return np.array([[x], [y], [z]])  # f(a,b,c) as column vector (in function space)


def coefficient_map(x0, nCoefficient, initialTau, lastCoefficientNorm=None, params=np.array([10, 28, 8 / 3])):
    """Taylor coefficient map for the flow of the lorenz equations over the interval [0,tau]. This takes the form
    (x,y,z) ---> coef which is a 3-by-N matrix i.e. coef = (a,b,c) where a,b,c are vectors of length N"""

    def sup_norm(v):
        """Return the sup norm on R^3"""
        return np.linalg.norm(v, np.inf)

    taylorCoefficient = np.array([[float(xi)] for xi in x0])  # initialize Taylor coefficient vector
    for m in range(nCoefficient - 1):
        tNext = (initialTau / (m + 1)) * coefficient_gen(taylorCoefficient[0, :], taylorCoefficient[1, :],
                                                         taylorCoefficient[2, :], params)
        taylorCoefficient = np.append(taylorCoefficient, tNext, 1)

    if lastCoefficientNorm is not None:  # Set integration time by desired final coefficient norm
        lastNonzeroIdx = nCoefficient - 1
        while sup_norm(taylorCoefficient[:, lastNonzeroIdx]) == 0:  # find index of highest order nonzero coefficient
            lastNonzeroIdx += -1
        rescaleBy = (lastCoefficientNorm / sup_norm(taylorCoefficient[:, lastNonzeroIdx])) ** (1 / lastNonzeroIdx)
        powerVector = np.array([pow(rescaleBy, i) for i in range(nCoefficient)])
        taylorCoefficient = taylorCoefficient * powerVector
        tau = rescaleBy * initialTau
    else:
        tau = initialTau

    return taylorCoefficient, tau



def sample_orbit(obCoef, timeNodes):
    """Evaluate a Lorenz orbit given as a 3-by-N matrix of coefficients at specified time nodes given as
    t = (t_0,...,t_k). Output is a 3-by-(k+1) matrix whose columns are points along the orbit"""

    N = np.shape(obCoef)[1]
    evalTensor = np.array([np.power(t, np.arange(N)) for t in timeNodes])
    return np.einsum('ij, kj -> ki', evalTensor, obCoef)


def bivar_polyval(sigma, p):
    """Evaluate a polynomial in 2 variables P(s,t) where powers of t increase across rows and powers of s along columns of p"""

    N1, N2 = np.shape(p)
    S = np.array(np.power(sigma[0], np.arange(N1)))
    T = np.array(np.power(sigma[1], np.arange(N2)))

    # grid eval for 2 variables
    # X: A vector of length N1
    # Y: A vector of length N2

    # N1 = np.shape(P)[0]
    # N2 = np.shape(P)[1]
    # sigma1 = np.array([np.power(x, np.arange(N1)) for x in X])
    # sigma2 = np.array([np.power(y, np.arange(N2)) for y in Y])
    # evl = np.einsum('ij, jkl, mk -> lim', sigma1, P, sigma2)

    return np.einsum('i, ij, j', S, p, T)


def lorenz_orbit(x0, nCoefficient, t_final, K=np.array([[-100, 100], [-100, 100], [-100, 100]]), t_initial=0):
    """Return an orbit for an initial condition"""

    def is_between(val, intval):
        """return True if val is in the interval defined"""

        return (val > intval[0]) and (val < intval[1])

    def in_K(v, K):
        """return True if v lies inside the box in R^3 defined by K = [a1, b1]x[a2,b2]x[a3,b3]"""
        return all(map(lambda val, intval: is_between(val, intval), v, K))

    # function start
    orbit_in_K = True  # placeholder function
    mu = np.finfo(float).eps  # set desired last coefficient for each segment to machine precision
    integration_time = np.abs(t_final - t_initial)  # directionless time units to integrate
    t_direction = np.sign(t_final - t_initial)  # direction of the integration
    timeSteps = [t_initial]  # initialize time step vector
    initialTau = t_direction * float(1 / 10)  # initial time rescaling with direction
    t = t_initial  # initialize time
    orbit = []  # initialize list of orbit segments
    while np.abs(t - t_initial) < integration_time and orbit_in_K:
        a, tau = coefficient_map(x0, nCoefficient, initialTau,
                                 lastCoefficientNorm=mu)  # compute parameterization of orbit segment
        t = t + tau  # advance time
        timeSteps.append(t)
        orbit.append(a)
        x1 = np.sum(a, 1)  # evaluate rescaled flow map at time = 1.
        orbit_in_K = in_K(x1, K)
        if not orbit_in_K:
            print('exit')
            for idx in range(3):
                if not is_between(x1[idx], K[idx]):
                    if x1[idx] < K[idx, 0]:
                        exitTau = exit_box(orbit[-1], idx, K[idx, 0], x1[idx])
                    else:
                        exitTau = exit_box(orbit[-1], idx, K[idx, 1], x1[idx])
                    break  # stop checking other coordinates

            tauScaling = exitTau[0]*tau  # choose new relative rescaling to terminate at exit time
            a, tau = coefficient_map(x0, nCoefficient,
                                     tauScaling)  # compute parameterization of orbit segment without rescaling via last coefficient
            orbit[-1] = a
        else:
            x0 = x1
    print(timeSteps)
    return orbit


def exit_box(orbit, idx, xIdxBound, xIdxFinal):
    """Given an orbit which has exited a box through the idx face (0, 1, or 2), determine the (local) exit time. xIdxBound
    is the boundary value which was violated on the coordinate of the box on the exit face and x_final is the value of
    the bad coordinate after evaluation
    map"""

    x0 = orbit[idx, 0]  # initial value of bad coordinate
    poly = orbit[idx, :4]  # use cubic approximation for rootfinding
    poly[0] += -xIdxBound  # we want to solve x(t) - xBound = 0

    def f(x):
        return np.polyval(np.flip(poly), x)  # numpy polynomial evaluation has coefficients in decreasing order

    def Df(x):
        polyder = np.polyder(np.flip(poly)) # derivative coefficient vector
        return np.polyval(polyder, x)  # derviative polynomial evaluation

    exitGuess = (xIdxBound - x0) / (xIdxFinal - x0)  # linear approximation of root
    exitTau = find_root(f, Df, exitGuess)  # approximate exit time.
    return exitTau

# get local manifold data
matlab_data = scipy.io.loadmat('lorenz_local_parm.mat')  # load MATLAB data file
P = matlab_data['midP']  # convert to numpy array
nFaceNodes = 50  # number of nodes on each face of the boundary. This should be chosen even so that 0 is never allowed for either coordinate
X = Y = np.linspace(-1, 1, nFaceNodes)
# bdNodes = np.array([np.concatenate([X, np.ones(nFaceNodes - 2), np.flip(X), -1 * np.ones(nFaceNodes - 2)]), tau = coefficient_map(x0, N, initTau)

stopHere
# # check orbit function
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# nCoefficient = 100
# evalTimes = np.linspace(0, 1, 100)
# t_final = -2
#
# # x0 = np.array([10, 11, 12])
# slowBd1 = np.array([1, 0.01])  # get a point near the local slow manifold boundary
# u1 = np.array([bivar_polyval(slowBd1, P[:, :, j]) for j in range(3)])
# slowBd2 = np.array([-1, 0.01])
# u2 = np.array([bivar_polyval(slowBd2, P[:, :, j]) for j in range(3)])
# fastBd1 = np.array([0.01, 1])  # get a point near the local slow manifold boundary
# u3 = np.array([bivar_polyval(fastBd1, P[:, :, j]) for j in range(3)])
# fastBd2 = np.array([0.01, -1])  # get a point near the local slow manifold boundary
# u4 = np.array([bivar_polyval(fastBd2, P[:, :, j]) for j in range(3)])
#
# initData = np.row_stack([u1, u2])
# colors = ['r', 'r', 'g', 'g']
# for idx, iData in enumerate(initData):
#     orbit = lorenz_orbit(iData, nCoefficient, t_final)
#     for ob in orbit:
#         obSample = sample_orbit(ob, evalTimes)
#         x, y, z = obSample
#         ax.plot(x, y, z, color=colors[idx])
# plt.show()
