from parameterization_tools import *
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# EXAMPLE 1:
# The map for the first N Taylor coefficients of the flow for ODE x' = x^2 - x
# This is a map of the form, T: R ---> R^N. In this simple case the map can be computed with any discrete convolution
# since the nonlinearity is only quadratic. Pretending this isn't the case, the best algorithm for computing T would
# require approximately N^2 floating point multiplications.


def logistic_taylor_orbit(x_0, N, tau=1.0):
    """Taylor coefficient map for the flow of the logistic with domain parameter tau computed via recursive formula."""

    def recursive_map(a):
        """The recursive Cauchy product terms for the logistic Taylor coefficient map"""
        return np.dot(a, np.flip(a)) - a[-1]

    taylorCoefficient = np.array([float(x_0)])  # initialize Taylor coefficient vector
    for j in range(N - 1):
        tNext = (tau / (j + 1)) * recursive_map(taylorCoefficient)
        taylorCoefficient = np.append(taylorCoefficient, tNext)

    return Taylor(taylorCoefficient)


def logistic_chebyshev_orbit(x_0, N, tau=1.0):
    """Chebyshev coefficient map for the flow of the logistic equation with domain parameter tau computed via Newton iteration.
    Initial guess is obtained by RK45 integration at the Chebyshev nodes followed by interpolation."""

    # start with an initial guess obtained from numerical integration
    chebyshevNodes = npoly.chebyshev.chebpts2(N)
    tNodes = tau*(chebyshevNodes + 1)  # map from [-1, 1] into the interval [0, 2*tau]
    rk45_solution = solve_ivp(lambda t, y: y ** 2 - y, [0, 2 * tau], [x_0], t_eval=tNodes)
    chebyshev_guess = Chebyshev.from_data(np.ravel(rk45_solution.y))


    # taylor_guess = logistic_taylor_orbit(x_0, np.min([N, 20]), tau=tau).project(N)  # Compute no more than 20 degree Taylor polynomial
    # cheb_guess = taylor_guess.taylor2chebyshev()

    # iterate Newton operator for the characterization map to refine the guess
    F = lambda u: zero_map(tau, x_0, N, u, 'Chebyshev', size=N)
    DF = lambda u: zero_map_diff(tau, N, u, 'Chebyshev', size=N)
    rk45_solution = find_root(F, chebyshev_guess.coef, jac=DF, tol=1e-13)
    return Chebyshev(rk45_solution)


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
        return (right_shift_map(diff_map(seq) - tau * (seq ** 2 - seq)) + (seq(0) - x_0) * seq.id()).project(size)
    else:
        return (tau * center_shift_map(seq ** 2 - seq) - Chebyshev(ezcat(0, seq.coef[1:])) + (
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
        DF_j = tau * center_shift_map((2 * seq - seq.id()).project(size)).left_multiply_operator() - np.eye(size)
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
        pass


def tau_from_last_coef(sequence, mu=np.finfo(float).eps):
    """Set domain parameter by last coefficient decay. Return the rescaled Taylor coefficients which have last coefficient norm
    equal to mu, and the domain parameter which achieves this decay."""

    maxIdx = np.max(np.nonzero(sequence.coef))  # index of the last nonzero coefficient
    tau = (mu / np.abs(sequence(maxIdx))) ** (1 / maxIdx)
    return tau


def tau_from_exp_reg(sequence):
    """Set domain parameter by the slope of the best fit exponential regression. Return the rescaled Taylor/Chebyshev
    coefficients and the domain parameter.

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


# def logistic_chebyshev_zero_map(tau, x_0, coef):
#     """Evaluate the Chebyshev IVP operator map. This map returns zero if and only if u is a (truncated) coefficient sequence
#     for the orbit of f through x_0 with time rescaling, tau."""
#
#     u = Chebyshev(coef)
#     fu = u.__pow__(2, u.N) - u
#     F_0 = u.eval(-1) - x_0
#     Fj = tau * center_shift_map(fu) - u
#     Fj.coef[0] = 0
#     Fu = F_0 * u.id() + Fj
#     return Fu.coef


# def diff_chebyshev_zero_map(tau, coef):
#     """Evaluate the derivative of the Chebyshev IVP operator map for the logistic equation. Returns a Jacobian matrix
#     of the same size as u."""
#     u = Chebyshev(coef)
#     DF_0 = np.array(ezcat(1, [2 * (-1) ** j for j in range(1, u.N)]))  # row 1 of the jacobian
#     DF_j = tau * center_shift_map(2 * u - u.id()).left_multiply_operator() - np.eye(u.N)
#     return np.row_stack([DF_0, DF_j[1:, :]])


N = 25
x_0 = 0.5
tau = 10

chebNodes = npoly.chebyshev.chebpts2(N)
t_node = tau*(chebNodes + 1)
sol = solve_ivp(lambda t, y: y**2 - y, [0, 2 * tau], [x_0], t_eval=t_node)
chebsol = logistic_chebyshev_orbit(x_0, N, tau=tau)
reg_u = Chebyshev.from_data(np.ravel(sol.y))




# %%

fig, ax = plt.subplots()
cheb_eval = np.linspace(-1, 1, 100)  # uniform partition of [-1, 1] to evaluate chebyshev series
t_eval = np.linspace(0, 2 * tau, 100)  # evaluate true solution here
ax.scatter(t_eval, phi(t_eval, x_0), 3)
ax.plot(t_eval, reg_u.eval(cheb_eval))
# ax.plot(t_eval, chebsol.eval(cheb_eval), 'r:')
# ax.scatter(sol.t, sol.y)
plt.show()
