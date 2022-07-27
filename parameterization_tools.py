"""
A collection of helper functions for parameterizing orbits in Examples from the paper:
"A deep learning approach to efficient parameterization of invariant manifolds for continuous  dynamical systems"
   
    Author: Shane Kepley
    email: s.kepley@vu.nl
    Date: 7/5/22;
"""
import numpy as np
from numpy import polynomial as npoly
from scipy import optimize
from scipy.linalg import toeplitz


def ezcat(*coordinates):
    """A multiple dispatch concatenation function for numpy arrays. Accepts arbitrary inputs as int, float, tuple,
    list, or numpy array and concatenates into a vector returned as a numpy array. This is recursive so probably not
    very efficient for large scale use."""

    if len(coordinates) == 1:
        if isinstance(coordinates[0], list):
            return np.array(coordinates[0])
        elif isinstance(coordinates[0], np.ndarray):
            return coordinates[0]
        else:
            return np.array([coordinates[0]])

    try:
        return np.concatenate([coordinates[0], ezcat(*coordinates[1:])])
    except ValueError:
        return np.concatenate([np.array([coordinates[0]]), ezcat(*coordinates[1:])])


def find_root(f, initialGuess, **kwargs):
    """Default root finding method to use if one is not specified"""

    solution = optimize.root(f, initialGuess, **kwargs)  # set root finding algorithm
    return solution.x  # return only the solution vector


class Sequence:
    """Base class for holding real/complex sequences defining analytic functions"""

    def __init__(self, coefficients, *truncation):
        """Initialize a coefficient sequence in ascending order and pad with zeros as needed"""

        if isinstance(coefficients, list):
            coefficients = np.array(
                coefficients)  # conversion to numpy array is necessary for multiplication to act correctly

        if truncation:
            self.N = truncation[0]
        else:
            self.N = len(coefficients)
        self.coef = self.embed(coefficients)

    def __call__(self, idx):
        """Return coefficients of the sequence."""

        return self.coef[idx]

    def __repr__(self):
        return repr(type(self)) + '  Order:{0} \n'.format(self.N) + repr(self.coef)

    def __rmul__(self, left_factor):
        """Multiplication by any scalar on the left for an arbitrary sequence is defined pointwise. This is called
        if the left factor __mul__ method yields a not_implemented exception type."""

        return getattr(self, '__class__')(left_factor * self.coef)

    def __add__(self, right_summand):
        """Addition of any sequences is pointwise"""

        try:
            return getattr(self, '__class__')(self.coef + right_summand.coef)

        except ValueError:  # summands are not the same size so embed into the smallest common dimension
            N = max([self.N, right_summand.N])
            return self.project(N) + right_summand.project(N)

    def __neg__(self):
        """Overload unary minus operator for Sequences"""

        return -1 * self

    def __sub__(self, subtract_sequence):
        """Overload subtraction for Sequences"""

        return self + subtract_sequence.__neg__()

    def __pow__(self, exponent, *args):
        """Generic powers recursively called using the subclass multiplication"""

        if not isinstance(exponent, int):
            raise TypeError
        elif exponent < 0:
            raise ValueError

        if exponent == 0:
            return self.id()
        elif exponent == 1:
            return self
        else:
            return self.__mul__(self.__pow__(exponent - 1, *args),
                                *args)  # recursively multiply and pass through variable arguments

    def id(self):
        """Return the multiplicative identity sequence in the same Banach algebra as self"""

        return getattr(self, '__class__')(ezcat(1, np.zeros(self.N - 1)))

    def embed(self, coefficients):
        """Embed coefficients into the correct truncation space by padding or truncating coefficients as needed."""
        # print(self.N)
        # print(coefficients)
        if len(coefficients) >= self.N:
            embed_coefficients = coefficients[:self.N]  # truncate to order N
        else:
            embed_coefficients = ezcat(coefficients, np.zeros(self.N - len(coefficients)))  # pad with zeros to order N
        return embed_coefficients

    def copy(self):
        """Return a deep copy of a Sequence"""

        return getattr(self, '__class__')(self.coef)

    def project(self, truncation):
        """Project this sequence into a new truncation space by truncating or padding with zeros. Unless truncation is
        equal to self.N, this returns a new copy of the Sequence."""

        if self.N == truncation:
            return self
        else:
            newSequence = self.copy()
            newSequence.N = truncation  # adjust truncation to desired space
            newSequence.coef = newSequence.embed(newSequence.coef)
            return newSequence

    def append(self, newCoefs):
        """Append some new coefficients to the end of a Sequence"""

        self.coef = ezcat(self.coef, newCoefs)
        self.N = max(self.N, len(self.coef))  # update the truncation space if necessary

    def norm(self, nu=1):
        """Return the ell^1_nu norm of a Sequence"""
        if nu == 1:
            return np.sum(np.abs(self.coef))
        else:
            return np.sum([nu ** j * np.abs(self(j)) for j in range(self.N)])

    def is_taylor(self):
        """return true if self is a Taylor sequence and false assumes its a Chebyshev sequence"""

        return issubclass(type(self), Taylor)


class Chebyshev(Sequence):
    """A sequence of the form a = (a_0, a_1,...,a_{N-1}) representing the Chebyshev series function
     a_0 + 2 * sum(a_j * T_j(t)). NOTE: a is the one sided Fourier coefficients for this function, NOT the true Chebyshev
     coefficients which are related by c_j = 2*a_j for j > 0.  """

    def __init__(self, *args, **kwargs):
        """Initialize a Chebyshev instance"""

        super().__init__(*args, **kwargs)  # initialize Sequence instance and pass through *args
        self.numpy = self.to_numpy()

    def __mul__(self, rightFactor, *truncate):
        """Product of Chebyshev sequences is given by discrete convolution. We do this by hijacking the numpy Chebyshev
        series class to ensure the product is correct and fast. Then broadcast back into our coefficient class and truncation."""

        try:  # rightFactor is another Chebyshev instance
            product = self.numpy * rightFactor.numpy  # hijack numpy Chebyshev class to perform multiplication.
            product_coefs = ezcat(product.coef[0], 0.5 * product.coef[
                                                         1:])  # rescale Chebyshev coefficients to conform to Chebyshev class format
            return Chebyshev(product_coefs,
                             *truncate)  # recast as Chebyshev sequence and pass through optional truncation args

        except AttributeError:  # rightFactor is not a Chebyshev instance, pass it to numpy multiplication and see what happens. This is
            # a lazy but very dangerous way to implement right scalar multiplication. Be careful!
            return Chebyshev(self.coef * rightFactor)

    def eval(self, t):
        """Evaluate this Chebyshev series by hijacking the numpy Chebyshev evaluation."""

        return self.numpy(t)

    def to_numpy(self):
        """Return a representation of this sequence in the numpy Chebyshev class. In the numpy implementation the factor
        of 2 is absorbed into the higher order coefficients. This class also ignores higher order zero coefficients which makes
        it annoying to use directly. However, the numpy instance is hijacked for some class methods to test for
        correctness or ensure efficiency."""

        return npoly.chebyshev.Chebyshev(ezcat(self.coef[0], 2 * self.coef[1:]))

    def left_multiply_operator(self):
        """Return a matrix representation for the linear operator on S which acts by multiplication with self composed
        with projection into the truncation space.

        Example: If u = self has length N and h is a length N truncated sequence, then this function returns a matrix
        representation for the linear operator L : R^N -> R^N satisfying L(h) = pi_N(u*h)."""

        A = np.array(list(map(lambda j: ezcat(0, self.coef[(1 + j):], np.zeros(j)), range(self.N))))
        B = toeplitz(self.coef, self.coef)
        return A + B

    @staticmethod
    def randint(N, maxValue=10):
        """Return a random Chebyshev sequence of length N with integer coefficients up to maxValue"""
        return Chebyshev(np.random.randint(maxValue, size=N))

    @staticmethod
    def from_numpy(npChebyshev, *truncate):
        """Convert an array of coefficients or a numpy.polynomial.chebyshev.Chebyshev instance into a Chebyshev instance."""

        if issubclass(type(npoly.chebyshev.Chebyshev), np.polynomial.chebyshev.Chebyshev):
            return Chebyshev.from_numpy(npChebyshev.coef, *truncate)
        else:
            return Chebyshev(ezcat(npChebyshev[0], 0.5 * npChebyshev[1:]),
                             *truncate)  # Rescale Chebyshev coefficients to one sided Fourier

    @staticmethod
    def from_data(data):
        """Return a Chebyshev interpolant for data evaluated at the Chebyshev nodes"""

        N = len(data)  # get the order of the interpolant (this is the degree + 1)
        chebNodes = npoly.chebyshev.chebpts2(N)  # grid of chebyshev nodes where the data is assumed to be defined
        numpy_cheb_coefs = npoly.chebyshev.chebfit(chebNodes, data, N - 1)  # chebyshev coefficients in numpy format
        return Chebyshev.from_numpy(numpy_cheb_coefs)


class Taylor(Sequence):
    """Taylor coefficient sequence of the form (a_0, a_1,...,a_{N-1}) representing the polynomial function
    sum a_j * t^j."""

    def __mul__(self, rightFactor, *truncate):
        """Return a Cauchy product"""

        try:  # rightFactor is another Taylor instance
            product_coefs = np.convolve(self.coef, rightFactor.coef)
            return Taylor(product_coefs,
                          *truncate)  # recast as Taylor sequence and pass through optional truncation args

        except AttributeError:  # rightFactor is not a Taylor instance, pass it to numpy multiplication and see what happens. This is
            # a lazy but very dangerous way to implement right scalar multiplication. Be careful!
            return Taylor(self.coef * rightFactor, *truncate)

    def left_multiply_operator(self, *dims):
        """Return a matrix representation for the linear operator on S which acts by multiplication with self composed
        with projection into the truncation space.
        Examples:

            L = u.left_multiply_operator() is a N-by-N matrix representation for the linear operator satisfying
        L(h) = pi_N(u*h) for any h in R^N.

            L = u.left_multiply_operator(m, n) is a n-by-m matrix representation for the linear operator satisfying
        L(h) = pi_m(u*h) for any h in R^n

        """
        if not dims:  # default is R^N for both domain and codomain
            dims = [self.N, self.N]

        col = self.project(dims[1])
        row = ezcat(self(0), np.zeros(dims[0] - 1))
        return toeplitz(col.coef, row)

    def eval(self, t):
        """Evaluate this Taylor polynomial"""

        return np.polyval(np.flip(self.coef), t)

    def taylor2chebyshev(self):
        """Change of basis from Taylor to Chebyshev."""

        # use numpy to convert from Taylor to Chebyshev basis. Note that numpy throws away leading zeros.
        numpy_chebyshev_coefs = npoly.chebyshev.poly2cheb(self.coef)
        # specify the size in order to replace zeros
        return Chebyshev(ezcat(numpy_chebyshev_coefs[0], 0.5 * numpy_chebyshev_coefs[1:]), self.N)

    @staticmethod
    def randint(N, maxValue=10):
        """Simple function to return a random Taylor sequence of length N with integer coefficients up to maxValue"""
        return Taylor(np.random.randint(maxValue, size=N))


def right_shift_map(seq, *truncate):
    """Evaluate the right shift map appearing in IVP for Taylor series"""

    return Taylor(ezcat(0, seq.coef), *truncate)


def right_shift_matrix(N):
    """Return the maxtrix representation of the right shift map acting on the first N coordinates as an N-by-N matrix"""

    return np.diag(np.array(np.ones(N - 1)), -1)


def diff_map(seq, *truncate):
    """Evaluate the differentiation map for solving IVP with Taylor"""

    return Taylor(np.array([(j + 1) * seq(j + 1) for j in range(seq.N - 1)]), *truncate)


def diff_map_matrix(N):
    """Return the matrix representation of the differentiation map actng on the first N coordinates as an
    N-by-N matrix."""

    return np.diag(np.arange(1, N), 1)


def center_shift_map(seq):
    """Evaluate the centered shift map appearing in IVP/BVP for Chebyshev series."""

    mid_sequence = np.array([(seq(j - 1) - seq(j + 1)) / (2 * j) for j in range(1, seq.N - 1)])
    return Chebyshev(ezcat(0, mid_sequence, seq(seq.N - 1) / (2 * (seq.N - 1))))


# def I_0(seq):
#     """Evaluate the map I_0(h) = (0, h_1, h_2,...,h_{N-1}) appearing in IVP/BVP for Chebyshev series"""
#
#     return Chebyshev(ezcat(0, seq.coef[1:]))

if __name__ == "__main__":
    a = Taylor.randint(5)
    h = Taylor.randint(5)
    c = npoly.chebyshev.poly2cheb(a.coef)
    A = Chebyshev(c)
