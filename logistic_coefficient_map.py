import numpy as np
from parameterization_tools import *
import matplotlib.pyplot as plt


# EXAMPLE 1:
# The map for the first N Taylor coefficients of the flow for ODE x' = x^2 - x
# This is a map of the form, T: R ---> R^N. In this simple case the map can be computed with any discrete convolution
# since the nonlinearity is only quadratic. Pretending this isn't the case, the best algorithm for computing T would
# require approximately N^2 floating point multiplications.


def logistic_taylor_orbit(x0, nCoefficient, tau=1.0):
    """Taylor coefficient map for the flow of the logistic equation over the interval [0,tau]. If lastCoefficientNorm
    is specified then the tau value is appended to the end of the vector"""

    def recursive_map(a):
        """The recursive Cauchy product terms for the logistic Taylor coefficient map"""
        return np.dot(a, np.flip(a)) - a[-1]

    taylorCoefficient = np.array([float(x0)])  # initialize Taylor coefficient vector
    for j in range(nCoefficient - 1):
        tNext = (tau / (j + 1)) * recursive_map(taylorCoefficient)
        taylorCoefficient = np.append(taylorCoefficient, tNext)

    return Taylor(taylorCoefficient)


def rescale(sequence, tau):
    """Generic function for rescaling a given Taylor sequence parameterizing an orbit of
    x' = f(x) into a parameterization of x' = tau*f(x)"""

    if tau == 1:
        return sequence

    elif sequence.is_taylor():
        powerVector = np.array([pow(tau, j) for j in range(sequence.N)])
        return Taylor(sequence.coef * powerVector)

    else:
        return


def tau_from_last_coef(sequence, mu=np.finfo(float).eps):
    """Set domain parameter by last coefficient decay. Return the rescaled Taylor coefficients which have last coefficient norm
    equal to mu, and the domain parameter which achieves this decay."""

    maxIdx = np.max(np.nonzero(sequence.coef))  # index of the last nonzero coefficient
    tau = (mu / np.abs(sequence(maxIdx))) ** (1 / maxIdx)
    return tau


def tau_from_exp_reg(sequence):
    """Set domain parameter by the slope of the best fit exponential regression. Return the rescaled Taylor/Chebyshev
    coefficients and the domain parameter"""

    X = np.nonzero(sequence.coef)  # indices of nonzero coefficients are the predictors for the regression
    Y = np.log(np.abs(sequence.coef[X]))  # log of coefficient modulus is the data for the regression
    A = np.row_stack([X, np.ones(np.shape(X))]).T  # matrix for the normal equations
    slope = np.linalg.lstsq(A, Y, rcond=None)[0][0]
    return np.exp(slope)


def manifold_map(coefficientMap, N, tauMap=lambda seq: 1):
    """Set the manifold evaluation map for a given Taylor coefficient map of order N and some choice of fixing tau.
    Input: coefficientMap - A function for returning Taylor or Chebyshev coefficients
            N - truncation dimension
            tauMap - A function to use for selecting the domain parameter."""

    def coefficient_map(x):
        """Initialize and define mapping onto Parameterization manifold"""
        unscaled_soln = coefficientMap(x, N)  # get non-rescaled coefficients
        tau = tauMap(unscaled_soln)  # find time rescaling for given method
        return tau, rescale(unscaled_soln, tau)

    return coefficient_map


def zero_map(tau, x_0, seq):
    """Zero finding map for Taylor/Chebyshev coefficients in the logistic example"""

    if seq.is_taylor():
        return right_shift_map(diff_map(seq) - tau * (seq ** 2 - seq)) + (seq(0) - x_0) * seq.id()
    else:
        pass


def zero_diff(tau, seq):
    """Return the derivative of the zero finding map for Taylor/Chebyshev coefficients in the logistic example."""

    if seq.is_taylor():
        sz = 2 * seq.N - 1  # dimension of projected jacobian
        jac = right_shift_matrix(sz) * (diff_map_matrix(sz) - 2 * tau * (
                seq ** 2).left_multiply_operator(sz, sz) + tau * np.eye(sz))
        jac[0, 0] += 1  # add I_0 operator
        return jac
    else:
        pass




def logistic_chebyshev_zero_map(tau, x_0, coef):
    """Evaluate the Chebyshev IVP operator map. This map returns zero if and only if u is a (truncated) coefficient sequence
    for the orbit of f through x_0 with time rescaling, tau."""

    u = Chebyshev(coef)
    fu = u.__pow__(2, u.N) - u
    F_0 = u.eval(-1) - x_0
    Fj = tau * center_shift_map(fu) - u
    Fj.coef[0] = 0
    Fu = F_0 * u.id() + Fj
    return Fu.coef


def diff_chebyshev_zero_map(tau, coef):
    """Evaluate the derivative of the Chebyshev IVP operator map for the logistic equation. Returns a Jacobian matrix
    of the same size as u."""
    u = Chebyshev(coef)
    DF_0 = np.array(ezcat(1, [2 * (-1) ** j for j in range(1, u.N)]))  # row 1 of the jacobian
    DF_j = tau * center_shift_map(2 * u - u.id()).left_multiply_operator() - np.eye(u.N)
    return np.row_stack([DF_0, DF_j[1:, :]])

if __name__ == "__main__":
    N = 10
    x0 = 0.5
    a = logistic_taylor_orbit(x0, N)
    tau0 = tau_from_last_coef(a)
    a0 = rescale(a, tau0)
    tau1 = tau_from_exp_reg(a)
    a1 = rescale(a, tau1)
    print(tau0, tau1)

    F = manifold_map(logistic_taylor_orbit, N, tau_from_last_coef)
    tau2, a2 = F(x0)

    y = zero_map(1, x0, a)
    print(y.norm())
    DF = zero_diff(1, a)
    print(np.linalg.matrix_rank(DF))
    # # Coefficient decay plots
    # X = np.nonzero(a.coef)[0]  # indices of nonzero coefficients are the predictors for the regression
    # Y = np.log(np.abs(a.coef[X]))  # log of coefficient modulus is the data for the regression
    # A = np.row_stack([X, np.ones(np.shape(X))]).T  # matrix for the normal equations
    # slope, intcpt = np.linalg.lstsq(A, Y, rcond=None)[0]
    #
    # plt.plot(X, Y, 'o')
    # plt.plot(X, slope * X + intcpt, 'r')
    # plt.show()

    # # test chebyshev zero finding map
    # tau = 2.2
    # x_0 = 0.5
    # u_init = ezcat(np.array([1, 1, 1]), np.zeros(17))
    # u0 = Chebyshev(u_init)




    # F = lambda u: logistic_chebyshev_zero_map(x_0, tau, u)
    # DF = lambda u: diff_chebyshev_zero_map(tau, u)
    # sol = find_root(F, DF, F(u_0.coef))
    # chebsol = Chebyshev(sol)
    #
    # phi = lambda t, x0: 1 / ((1 / x0 - 1) * np.exp(t) + 1)
    #
    # fig, ax = plt.subplots()
    # cheb_eval = np.linspace(-1, 1, 100)  # evaluate chebyshev series here
    # t_eval = np.linspace(0, 2 * tau, 100)  # evaluate true solution here
    # ax.plot(t_eval, chebsol.eval(cheb_eval))
    # ax.plot(t_eval, phi(t_eval, x_0), 'r:')
    # plt.show()
