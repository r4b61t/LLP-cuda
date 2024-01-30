import numpy as np
from numpy import kron, ones


class Exact():
    """

    Parameters

    ----------

    Qs : np.ndarray,
        A 2D numpy array containing Q-values.

    As : np.ndarray,
        A 2D numpy array containing association matrix.

    e : float, default=1/20
        The weight between the bank's revenue and the cash flow
        among bank's customers in the cost function.

    Attributes

    ----------

    N : int,
        The number of assets.

    M : int,
        The number of actions.

    h : np.ndarray,
        A 2D numpy array containing effective onsite "magnetic" fields.

    J : np.ndarray,
        A 2D numpy array containing "spin-spin "interactions between assets.

    costs : np.ndarray,
        Cost function during optimisation.

    cost_min : float,
        The minimum cost function.

    x : string,
        A string containing the best bitstring.

    """

    def __init__(self, Qs, As, e):

        assert Qs.ndim == 2
        assert As.ndim == 2
        assert As.shape[0] == As.shape[1]
        assert As.shape[0] == Qs.shape[0]
        assert isinstance(e, float) and e >= 0

        self.N = Qs.shape[0]
        self.M = Qs.shape[1]
        self.h = (1 - e) * np.copy(Qs)
        self.h[:, 0] += -e * np.sum(As, axis=1)
        self.J = e * np.copy(As)
        self.e = e
        self.c0 = -e * np.sum(As) / 2
        self.costs = None
        self.cost_min = None
        self.x = None

    def optimized(self):

        self._set_exact_energies()

        self.cost_min = min(self.costs)

        self.x = self._index2bitstring(np.argmin(self.costs))

    def _index2bitstring(self, idx):

        x = np.zeros(self.N)

        num = np.array([int(i) for i in np.base_repr(idx, base=self.M)])

        x[(self.N - len(num)):] = num

        x = [str(i) for i in x.astype('int')]

        new = ""
        for i in x:
            new += i

        return new

    def _set_exact_energies(self):

        if self.costs is None:

            self.costs = np.zeros(self.M ** self.N) + self.c0

            coupling_node = np.zeros(self.M)
            coupling_node[0] = 1

            for i in range(self.N):

                self.costs += self._onsite_op(-self.h[i, :], i)

                for ii in range(i):

                    if self.J[i, ii] != 0:
                        self.costs += -self.J[i, ii] * self._twosite_op(coupling_node, coupling_node, ii, i)

    def _onsite_op(self, P, i):

        return kron(ones(self.M ** i), kron(P, ones(self.M ** (self.N - i - 1))))

    def _twosite_op(self, P, Q, i, j):

        assert i < j

        return kron(ones(self.M ** i),
                    kron(P, kron(ones(self.M ** (j - i - 1)), kron(Q, ones(self.M ** (self.N - j - 1))))))
