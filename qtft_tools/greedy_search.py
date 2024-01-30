import numpy as np
from tqdm import tqdm


class GreedySearch():
    """

    Parameters

    ----------

    x0 : string,
        The initial bitstring.

    Qs : np.ndarray,
        A 2D numpy array containing Q-values.

    As : np.ndarray,
        A 2D numpy array containing association matrix.

    Ls : np.ndarray
        A 2D numpy array containing provisions.

    e : float, default=1/20
        The weight between the bank's revenue and the cash flow
        among bank's customers in the cost function.


    Attributes

    ----------

    xs : np.ndarray,
        A numpy array containing optimized bitstrings.

    cs : np.ndarray,
        A numpy array containing cost functions associated
        with xs.

    ls : np.ndarray,
        A numpy array containing provisions associated
        with xs.

    x_best : string,
        The best bitstring.

    c_best : float,
        The minimum cost function.

    l_best : float,
        The minimum combined provisios.

    """

    def __init__(self, x0, Qs, As, Ls, e=1 / 20):

        assert isinstance(x0, str)
        assert len(x0) == Qs.shape[0]
        for char in x0:
            assert int(char) < Qs.shape[1]
        assert Qs.ndim == 2
        assert As.ndim == 2
        assert As.shape[0] == As.shape[1]
        assert As.shape[0] == Qs.shape[0]
        assert isinstance(e, float) and e >= 0
        assert Ls.shape[0] == Qs.shape[0]
        assert Ls.shape[1] == Qs.shape[1]

        self.x0 = x0
        self.Qs = np.copy(Qs)
        self.h = (1 - e) * np.copy(Qs)
        self.h[:, 0] += -e * np.sum(As, axis=1)
        self.J = e * np.copy(As)
        self.y0 = e * np.sum(As) / 2
        self.e = e
        self.Ls = Ls

        self.N = Qs.shape[0]
        self.M = Qs.shape[1]
        self.x_best = np.zeros((self.N, self.M), dtype=int)
        for i in range(self.N):
            self.x_best[i][int(x0[i])] = 1
        self.y_best = self.get_objective_function(self.x_best)
        self.l_best = self.get_provision(self.x_best)
        self.q_best = None
        self.ys = [self.y_best]
        self.ls = [self.l_best]

    def optimized(self, maxiter=300):

        count = 0
        for _ in tqdm(range(maxiter)):
            rs = np.zeros((self.N, self.M))
            for i in range(self.N):
                for j in range(self.M):
                    new_x = self.update_string(self.x_best, i, j)
                    y = self.get_objective_function(new_x)
                    l = self.get_provision(new_x)
                    reward = self.l_best - l
                    penalty = self.y_best - y

                    if reward < 0:
                        rs[i, j] = -float('inf')
                    elif reward == 0 and penalty > 0:
                        rs[i, j] = -float('inf')
                    elif reward >= 0 and penalty <= 0:
                        rs[i, j] = reward
                    elif reward > 0 and penalty > 0:
                        rs[i, j] = -penalty / reward

            i, j = np.unravel_index(rs.argmax(), rs.shape)
            self.x_best = self.update_string(self.x_best, i, j)
            self.y_best = self.get_objective_function(self.x_best)
            self.l_best = self.get_provision(self.x_best)

            self.ys += [self.y_best]
            self.ls += [self.l_best]

        self.q_best = np.sum(np.multiply(self.Qs, self.x_best))

    def update_string(self, bitstring, asset_i, action_j):
        x = np.copy(bitstring)
        x[asset_i] = 0
        x[asset_i][action_j] = 1
        return x

    def get_objective_function(self, state):

        y = np.sum(np.multiply(self.h, state))
        ind = np.where(self.x_best[:, 0] == 1)[0]
        y += 0.5 * np.sum(self.J[ind][:, ind])
        y += self.y0

        return y

    def get_provision(self, state):

        l = np.sum(np.multiply(self.Ls, state))

        return l
