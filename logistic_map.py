from parameterization_tools import *
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# EXAMPLE 1:
# The map for the first N Taylor coefficients of the flow for ODE x' = x^2 - x
# This is a map of the form, T: R ---> R^N. In this simple case the map can be computed with any discrete convolution
# since the nonlinearity is only quadratic. Pretending this isn't the case, the best algorithm for computing T would
# require approximately N^2 floating point multiplications.


def logistic_taylor_orbit(x_0, N, tau=1.0):
    """Taylor coefficient map for the flow of the logistic with domain parameter tau computed via recursive formula.
    Input:
        x_0 - A floating point initial condition.
        N - The number of Taylor coefficients to return (i.e. 1 + degree).
        tau - Integration time rescaling parameter.
    Output:
        A Taylor Sequence
    """

    def recursive_map(a):
        """The recursive Cauchy product terms for the logistic Taylor coefficient map"""
        return np.dot(a, np.flip(a)) - a[-1]

    taylorCoefficient = np.array([np.float32(x_0)])  # initialize Taylor coefficient vector
    for j in range(N - 1):
        tNext = (tau / (j + 1)) * recursive_map(taylorCoefficient)
        taylorCoefficient = np.append(taylorCoefficient, tNext)

    return Taylor(taylorCoefficient)


def logistic_chebyshev_orbit(x_0, N, tau=1.0):
    """Chebyshev coefficient map for the flow of the logistic equation with domain parameter tau computed via Newton iteration.
    Initial guess is obtained by RK45 integration at the Chebyshev nodes followed by interpolation

    a = logistic_chebyshev_orbit(x_0, N) returns the Chebyshev series of order N parameterizing the orbit of the logistic
        equation x' = x^2 - x with initial data x_0.

    a = logistic_chebyshev_orbit(x_0, N, tau=T) returns the Chebyshev series of order N parameterizing the orbit of the
        logistic equation x' = T * (x^2 - x) with initial data x_0 and domain parameter T > 0.

    Example:
        x_0 = 0.5
        N = 25
        tau = 3
        chebsol = logistic_chebyshev_orbit(x_0, N, tau=tau)  # Return a chebyshev series parameterizing the orbit for t in [0, 6]
        fig, ax = plt.subplots()  # Verify the parameterization by plotting against rk45
        cheb_eval = np.linspace(-1, 1, 100)  # uniform partition of [-1, 1] to evaluate chebyshev series
        t_eval = np.linspace(0, 2 * tau, 100)  # time nodes associated with evaluation of the true solution for t in [0, 6]
        ax.scatter(t_eval, phi(t_eval, x_0), 3)  # true solution plotted on [0, 6]
        ax.plot(t_eval, chebsol.eval(cheb_eval), 'r:')  # chebyshev solution plotted on [-1, 1]
        plt.show()"""

    # start with an initial guess obtained from numerical integration
    chebyshevNodes = npoly.chebyshev.chebpts2(N)
    tNodes = tau * (chebyshevNodes + 1)  # map from [-1, 1] into the interval [0, 2*tau]
    rk45_solution = solve_ivp(lambda t, y: y ** 2 - y, [0, 2 * tau], [x_0], t_eval=tNodes)
    chebyshev_guess = Chebyshev.from_data(np.ravel(rk45_solution.y))

    # iterate Newton operator for the characterization map to refine the guess
    F = lambda u: zero_map(tau, x_0, N, u, 'Chebyshev', size=N)
    DF = lambda u: zero_map_diff(tau, N, u, 'Chebyshev', size=N)

    # return Chebyshev(find_root(F, chebyshev_guess.coef, jac=DF, tol=1e-13))
    return Chebyshev(newton_root(F, DF, chebyshev_guess.coef, xtol=1e-13))


def logistic_characteristic_map(tau, x_0, seq, size=None):
    """Evaluate the Taylor or Chebyshev characteristic map for the logistic example.

    Examples:
        y = logistic_characteristic_map(tau, x_0, u, 5) where u = (a0, a1, a2, a3, a4) is a Taylor sequence yields a
        sequence in S_{5} which is zero iff (a0,...,a_4) are the first 5 Taylor coefficients for the orbit of
        x' = tau*(x^2 - x) with initial data x_0.

        y = logistic_characteristic_map(tau, x_0, u) where u = (a0, a1, a2, a3, a4) is a Taylor sequence yields a
        sequence in S_{10} with zeros to order 5 iff (a0,...,a_4) are the first 5 Taylor coefficients for the orbit of
        x' = tau*(x^2 - x) through x_0. The values (y_6,...,y_10) are nonzero "spillover" terms whose size depends on
        the domain parameter.

        y = logistic_characteristic_map(tau, x_0, u, 5) where u = (a0, a1, a2, a3, a4) is a Chebyshev sequence yields a
        sequence in S_{5} which is nonzero but "small" if (a0,...,a_4) are the first 5 Chebyshev coefficients for the
        orbit of x' = tau*(x^2 - x) with initial data x_0. Here y is not zero even if u is exact because Chebyshev
        coefficients depend on one another at all orders."""

    if size is None:
        size = 2 * seq.N  # default dimension of characterization map is the full infinite dimensional map which has finite
        # dimensional range when seq is a polynomial.

    if seq.is_taylor():
        return (right_shift_map(diff_map(seq) - tau * (seq ** 2 - seq)) + (seq[0] - x_0) * seq.id()).project(size)
    else:
        return (tau * center_shift_map(seq ** 2 - seq) - Chebyshev(ezcat(0, seq[1:])) + (
                seq.eval(-1) - x_0) * seq.id()).project(
            size)


def logistic_characteristic_diff(tau, seq, size=None):
    """Evaluate the Frechet derivative of the Taylor characteristic map for the logistic example. This returns a matrix
    representation of the derivative.

    Example:
        dy = logistic_characteristic_diff(tau, (a0, a1, a2, a3, a4), 5) yields a 5x5 matrix which is the Jacobian
        of the characterization map with projection onto C^5 i.e. D(Pi_N(F(a))).

        dy = logistic_characteristic_diff(tau, (a0, a1, a2, a3, a4)) yields a 10x10 matrix which is the Jacobian
        of the characterization map with projection onto its full image in C^{10} i.e. D(F(a))."""

    if size is None:
        size = 2 * seq.N  # default dimension of characterization map derivative is the full infinite dimensional map which
        # has finite rank when seq represents a polynomial.

    if seq.is_taylor():
        dy = right_shift_matrix(size) @ (
                diff_map_matrix(size) - 2 * tau * seq.left_multiply_operator(size, size) + tau * np.eye(size))
        dy[0, 0] += 1  # add I_0 operator
        return dy
    else:
        DF_0 = np.array(ezcat(1, [2 * (-1) ** j for j in range(1, size)]))  # row 1 of the jacobian
        DF_j = tau * center_shift_map_matrix(size) @ (2 * seq - seq.id()).project(size).left_multiply_operator() - np.eye(size)
        return np.row_stack([DF_0, DF_j[1:, :]])


def zero_map(tau, x_0, N, coefVector, basis, size=None):
    """Zero finding map for Newton iteration in the logistic example. This evaluates a characterization map but passes
    inputs and outputs as vectors in R^m instead of as Sequence objects hence this can be passed to a rootfinder.

    Inputs:
        tau - domain parameter
        x_0 - initial condition
        N - size of the truncation space where the projected solution lives
        basis - Either 'Taylor' or 'Chebyshev'
        size - size of the truncation to apply after evaluating F. This need not be the same as N and can be up to 2*N
            to allow evaluation of spillover terms.

    Taylor example:
        F = lambda u: zero_map(tau, x_0, N, u, 'Taylor', N) is the full logistic characterization map (in practice it maps from R^N into R^N
        DF = lambda u: zero_map_diff(tau, N, u, 'Taylor', N) is the derivative of the logistic characterization map
        u0 = logistic_taylor_orbit(x_0, 5, tau).project(N)  is the first 5 coefficients computed by recursion then
            padded with zeros.
        sol = find_root(F, u0.coef, jac=DF) returns the full Taylor expansion to order N via Newton iteration of the characterization map.
        print(logistic_characteristic_map(tau, x_0, Taylor(sol), N).norm()) verifies it has small residual

    Chebyshev example:
        N = 20
        F = lambda u: zero_map(tau, x_0, N, u, 'Chebyshev', size=N)
        DF = lambda u: zero_map_diff(tau, N, u, 'Chebyshev', size=N)
        u0 = Chebyshev(np.array([0.5, 1.2, -0.45]), N)
        sol = find_root(F, u0.coef, jac=DF)
        chebsol = Chebyshev(sol)

        # plot solution against ground truth
        phi = lambda t, x0: 1 / ((1 / x0 - 1) * np.exp(t) + 1)
        fig, ax = plt.subplots()
        cheb_eval = np.linspace(-1, 1, 100)  # evaluate chebyshev series here
        t_eval = np.linspace(0, 2 * tau, 100)  # evaluate true solution here
        ax.plot(t_eval, chebsol.eval(cheb_eval))
        ax.plot(t_eval, phi(t_eval, x_0), 'r:')
        plt.show()"""

    if basis == 'Taylor':
        return (logistic_characteristic_map(tau, x_0, Taylor(coefVector, N), size=size)).coef
    else:
        return (logistic_characteristic_map(tau, x_0, Chebyshev(coefVector, N), size=size)).coef


def zero_map_diff(tau, N, coefVector, basis, size=None):
    """Return the derivative of the zero finding map for the logistic example."""

    if basis == 'Taylor':
        return logistic_characteristic_diff(tau, Taylor(coefVector, N), size=size)
    else:
        return logistic_characteristic_diff(tau, Chebyshev(coefVector, N), size=size)


# define a zero finding problem for tau
def newton_residual(x_0, tau, u):
    """Evaluate the map |DF(u)^{-1} * F(u)| where u is a zero of the characterization map to order N"""

    y = logistic_characteristic_map(tau, x_0, u)
    dy = logistic_characteristic_diff(tau, u)
    return np.linalg.norm(np.linalg.solve(dy, y.coef))


# domain parameter definitions
def rescale(sequence, tau):
    """Generic function for rescaling a given parameterization sequence for an orbit of
    x' = f(x) into a parameterization of x' = tau*f(x)."""

    if tau == 1:
        return sequence

    elif sequence.is_taylor():  # rescaling is a simple formula which rescales a_j by tau^j
        powerVector = np.array([pow(tau, j) for j in range(sequence.N)])
        return Taylor(sequence.coef * powerVector)

    else:  # Chebyshev rescaling by Newton iteration.
        chebyshev_guess = Chebyshev(
            ezcat(sequence[0], tau * sequence.coef[1:]))  # initial guess is just rescaling coefficients by tau

        # iterate Newton operator for the characterization map to refine the guess
        F = lambda u: zero_map(tau, x_0, N, u, 'Chebyshev', size=N)
        DF = lambda u: zero_map_diff(tau, N, u, 'Chebyshev', size=N)
        # chebyshev_coefs = find_root(F, chebyshev_guess.coef, jac=DF, tol=1e-13)
        chebyshev_coefs = newton_root(F, DF, chebyshev_guess.coef, xtol=1e-13)
        return Chebyshev(chebyshev_coefs)


def tau_from_last_coef(sequence, mu=np.finfo(float).eps):
    """Set domain parameter by last coefficient decay. Input is a sequence computed with domain parameter equal to 1.
    Returns the domain parameter rescaling which forces the last coefficient norm to be equal to mu."""

    if not isinstance(sequence, Sequence):
        print("tau_from_last_coef should receive a Sequence as input, not an np.array or torch.tensor")
        raise KeyboardInterrupt

    maxIdx = np.max(np.nonzero(sequence.coef))  # index of the last nonzero coefficient
    tau = (mu / np.abs(sequence.coef[maxIdx])) ** (1 / maxIdx)
    return tau


def tau_from_exp_reg(sequence):
    """Set domain parameter by the slope of the best fit exponential regression. Input is a sequence computed with
    domain parameter equal to 1. Return the domain parameter as the exp(slope) of this best fit line.

    Example:
        a = logistic_taylor_orbit(0.5, 100, tau=1.0)
        X = np.nonzero(a.coef)[0]  #  indices of nonzero coefficients are the predictors for the regression
        Y = np.log(np.abs(a.coef[X]))  # log of coefficient modulus is the data for the regression
        A = np.row_stack([X, np.ones(np.shape(X))]).T  # matrix for the normal equations
        slope, intcpt = np.linalg.lstsq(A, Y, rcond=None)[0]  # least squares regression

        plt.plot(X, Y, 'o')  # plot coefficients in log scale with best fit line verifies the fit.
        plt.plot(X, slope * X + intcpt, 'r')"""

    X = np.nonzero(sequence.coef)  # indices of nonzero coefficients are the predictors for the regression
    Y = np.log(np.abs(sequence.coef[X]))  # log of coefficient modulus is the data for the regression
    A = np.row_stack([X, np.ones(np.shape(X))]).T  # matrix for the normal equations
    slope = np.linalg.lstsq(A, Y, rcond=None)[0][0]
    return np.exp(slope)


def tau_from_residual(x_0, sequence, mu, bracket=(0, 10.0)):
    """Set domain parameter by the residual of the finite rank Newton map. Input is a sequence computed with
    domain parameter equal to 1. Return the domain parameter satisfying |DF(u)^{-1} * F(u)| = mu."""

    # define a zero finding problem for tau
    def F(tau):
        """Evaluate the zero finding problem for tau_3"""

        # call with *tau to unpack numpy array. If this continues to be a problem I have to vectorize the characteristic map calls
        test_seq = rescale(sequence, tau)
        return newton_residual(x_0, tau, test_seq)

    return lambda tau: F(tau) - mu

    # soln = optimize.root_scalar(lambda tau: F(tau) - mu, bracket=bracket)
    # if soln.converged:
    #     return soln.root
    # else:
    #     print('Domain parameter failed to find a root in the interval: {0}. Try a wider range'.format(bracket))
    #     raise KeyboardInterrupt


# define functions for training a neural network parameterization
def phi(t, x_0):
    """Ground truth evaluation for the flow of the logistic equation."""
    return 1 / ((1 / x_0 - 1) * np.exp(t) + 1)


def parameterization_manifold_map(coefficientMap, N, tauMap):
    """Set the manifold evaluation map for a given Taylor coefficient map of order N and some choice of domain parameter map.
    Input:
            coefficientMap - A function for returning Taylor or Chebyshev coefficients
            N - truncation dimension
            tauMap - A function to use for selecting the domain parameter.

    Example: F = manifold_map(logistic_taylor_orbit, 20, lambda seq:tau_from_last_coef(seq, 1e-13)) is a map from R --> R^20 such that
        F(x0) = (a0, a1,...,a_{19}) is the Taylor parameterization for x' = tau*(x^2 - x) where tau is chosen so that
        |a_{19}| = 1e-13."""

    def coefficient_map(x):
        """Initialize and define mapping onto Parameterization manifold"""
        unscaled_soln = coefficientMap(x, N)  # get non-rescaled coefficients
        tau = tauMap(unscaled_soln)  # compute the required domain parameter
        return tau, rescale(unscaled_soln, tau)

    return coefficient_map


if __name__ == '__main__':
    x_0 = 0.5
    N = 15
    tau = 1
    u = logistic_chebyshev_orbit(x_0, N, tau=tau)
    yN = logistic_characteristic_map(tau, x_0, u, size=N)
    yF = logistic_characteristic_map(tau, x_0, u)

    # plots of newton steps

    res = lambda seq, tau: newton_residual(x_0, tau, rescale(seq, tau))
    # res = lambda t: newton_residual(x_0, t, logistic_chebyshev_orbit(x_0, N, tau=t))

    a = logistic_taylor_orbit(x_0, N)
    c = logistic_chebyshev_orbit(x_0, N)
    tauNodes = np.linspace(1.0, 4, 100)
    taylorResidual = np.array([res(a, t) for t in tauNodes])
    chebyshevResidual = np.array([res(c, t) for t in tauNodes])

    plt.figure()
    plt.plot(tauNodes, taylorResidual)
    plt.title('Taylor')
    plt.show()

    plt.figure()
    plt.plot(2 * tauNodes, chebyshevResidual)
    plt.title('Chebyshev')
    plt.show()


    t = 1.3
    u = logistic_chebyshev_orbit(x_0, N, t)
    y = logistic_characteristic_map(t, x_0, u)
    dy = logistic_characteristic_diff(t, u)


    # plot solution
    plt.figure()
    plt.plot(np.linspace(-1, 1, 100), [u(t) for t in np.linspace(-1, 1, 100)])
    plt.show()


# # %% Taylor vs Chebyshev C^K vs analytic
# x_0 = 0.5
# N = 25
# tau = 3
# a = logistic_taylor_orbit(x_0, N, tau=1)
# c = logistic_chebyshev_orbit(x_0, N, 0.5 * tau)
#
# t_eval = np.linspace(0, tau, 100)
# tay_eval = np.linspace(0, 1, 100)
# cheb_eval = np.linspace(-1, 1, 100)
#
# fig, ax = plt.subplots()
# plt.figure()
# ax.plot(t_eval, c.eval(cheb_eval))
# ax.plot(t_eval, a.eval(tay_eval), 'r:')
#
# plt.figure()
# plt.plot(t_eval, a.eval(tay_eval), 'r:')
# # plt.show()
#
# plt.figure()
# plt.plot(t_eval, c.eval(cheb_eval))
# # plt.show()
