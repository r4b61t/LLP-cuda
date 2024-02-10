# Note: to edit
# QAOA , combine_groups
#
#  Quantum algorithms for loan loss collection optimization
#  with a provision constraint.
#
#  Written by QTFT team. 7th Oct 2021.
#
import json

from qtft_tools.exact import Exact
from qtft_tools.greedy_search import GreedySearch
from qtft_tools.group import Group
from qtft_tools.qaoa import QAOA
import logging
from typing import List, Optional

import numpy as np
from tqdm import tqdm
import networkx as nx
import community

from new_basis_llp_qaoa.statevector_llp_v2 import StateVectorLLPV2, get_str


class DC_QAOA():
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

    p : int, default=2
        The number of driving cycles in QAOA.

    n_candidates : int, default=10
        The maximum number of bitstrings that represent the solutions
        of a given problem instance.

    max_community_size : int, default=7


    Attributes

    -------

    N : int,
        The number of assets.

    M : int,
        The number of actions.

    groups : list,
        A list containing Group objects that represent asset communities.

    res : Group,
        The solution for all assets.

    x_best : string,
        The best found bitstring.

    """

    def __init__(self, Qs, As, e=1 / 20, p=2, n_candidates=10, max_community_size=7):

        assert Qs.ndim == 2
        assert As.ndim == 2
        assert As.shape[0] == As.shape[1]
        assert As.shape[0] == Qs.shape[0]
        assert isinstance(e, float) and e >= 0
        assert isinstance(p, int) and p >= 1
        assert isinstance(n_candidates, int) and n_candidates > 0
        assert isinstance(max_community_size, int) and max_community_size > 0

        self.Qs = Qs
        self.As = As
        self.e = e
        self.p = p
        self.n_candidates = n_candidates
        self.max_community_size = max_community_size

        self.N = Qs.shape[0]
        self.M = Qs.shape[1]
        self.groups: List[Group] = []
        self.res: Optional[Group] = None

    def to_dict(self):
        return dict(
            e=self.e,
            p=self.p,
            n_candidates=self.n_candidates,
            max_community_size=self.max_community_size,
            N=self.N,
            M=self.M,
            groups=[group.to_dict() for group in self.groups],
            res=self.res.to_dict() if self.res else None
        )

    def to_json(self, file_name: str):
        with open(file_name, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_json(cls, file_name: str):
        with open(file_name) as json_file:
            info_dict = json.load(json_file)
        res = DC_QAOA(Qs=np.identity(2), As=np.identity(2))
        res.e = info_dict['e']
        res.p = info_dict['p']
        res.n_candidates = info_dict['n_candidates']
        res.max_community_size = info_dict['max_community_size']
        res.N = info_dict['N']
        res.M = info_dict['M']
        res.groups = [Group.from_dict(group) for group in info_dict['groups']]
        res.res = info_dict['res']
        return res

    def optimized(self, maxiter=20, method='COBYLA', use_cache=True):

        # Step 1: Community Detection
        if not self.groups:
            self.set_communities()

        for i in tqdm(range(len(self.groups)), desc="Performing QAOA for subgroups"):
            logging.info(f"Group {i}")
            group = self.groups[i]

            # Perform QAOA
            Q = self.Qs[group.nodes]
            A = self.As[group.nodes][:, group.nodes]
            logging.info(f"Initialising StateVectorLLP for group {i} with group size = {group.size}.")
            m = StateVectorLLPV2(Q, A, self.e, self.p)

            logging.info(f"Performing QAOA for group {i}.")
            # to edit -------------------------------------------------------
            m.optimized(maxiter=maxiter, method=method, use_cache=use_cache)
            logging.info(f"QAOA for group {i} is done. Saving results.")
            delattr(m, 'ub_coarse_cache')
            delattr(m, 'hb_taylor_terms')
            delattr(m, 'hb')
            delattr(m, 'ha')

            # Output probabilities
            ps = m.probabilities().copy()
            group.c_qaoa = m.get_energy()

            # Save Candidates
            inds = np.argsort(ps)[::-1][:self.n_candidates]
            group.ps = ps[inds]

            # "Save selected bitstrings
            for ind in inds:
                group.xs = np.append(group.xs, m.get_str_from_index(ind))

            # "Save cost functions for selected bitstrings
            for bitstring in group.xs:
                group.cs = np.append(group.cs, m.get_cost_from_str(bitstring))

            # Save exact ground state (not necessary)
            m_exact = Exact(Q, A, self.e)
            m_exact.optimized()

            group.x_exact = m_exact.x
            group.c_exact = m_exact.cost_min

            q_number = m.qubits_number
            action_bit_len = m.action_bit_len

            logging.info("Begin state reconstruction.")
            if i == 0:
                self.res = self.groups[0]
            else:
                self.res = self._combine_groups(self.res, self.groups[i], ps, q_number, action_bit_len)
            logging.info(f"Done with group {i}.")

        # Sorting
        ind = np.argsort(self.res.nodes)
        self.res.id = "final result"
        self.res.nodes = self.res.nodes[ind]
        for i in range(len(self.res.xs)):
            self.res.xs[i] = "".join(list(np.array(list(self.res.xs[i]))[ind]))

        # Best solution
        self.x_best = self.res.xs[np.argmin(self.res.cs)]

    def set_communities(self):

        # Perform Greedy Modularity community finding.
        As_nx = nx.convert_matrix.from_numpy_array(self.As)
        G = nx.algorithms.community.modularity_max.greedy_modularity_communities(As_nx)

        for i in range(len(G)):
            # Perform iterative Louvain community finding with shared nodes.
            group = Group(str(i), np.array(list(G[i])))

            # Update groups
            self.groups += self._get_louvian_communities_with_shared_nodes(group)

        # Sort descendingly according to the size of the group.
        n_groups = np.array([group.size for group in self.groups])
        ind = np.argsort(n_groups)[::-1]
        self.groups = np.array(self.groups)[ind]

    def _combine_groups(self, L, R, ps, q_number, action_bit_len):

        # Find the indices of the shared nodes
        _, ind_L, ind_R = np.intersect1d(L.nodes, R.nodes, return_indices=True)

        # Create a new combined group
        LR = Group(name=L.id + "|" + R.id,
                   nodes=np.concatenate((L.nodes, np.delete(R.nodes, ind_R))))

        # Loop over all candidates
        for ii in range(len(L.xs)):
            for jj in range(len(R.xs)):
                LR = self._combine_bitstrings(L.xs[ii], R.xs[jj], ind_L, ind_R, LR)

        # Generate more candidates if no valid reconstruction
        count = 0
        idx = np.argsort(ps)[::-1]
        while len(LR.xs) == 0:
            R_x = get_str(idx[self.n_candidates - 1 + count], q_number, action_bit_len)
            for ii in range(len(L.xs)):
                LR = self._combine_bitstrings(L.xs[ii], R_x, ind_L, ind_R, LR)
            count += 1

        # Evaluate bitstring. (QAOA is not performed.)
        m2 = StateVectorLLPV2(self.Qs[LR.nodes], self.As[LR.nodes][:, LR.nodes], self.e, self.p)
        for x in LR.xs:
            LR.cs = np.append(LR.cs, m2.get_cost_from_str(x))

        # Truncate
        ind = np.argsort(LR.cs)[::-1][:self.n_candidates]
        LR.xs = LR.xs[ind]
        LR.cs = LR.cs[ind]

        return LR

    def _combine_bitstrings(self, x_L, x_R, ind_L, ind_R, LR):

        # Check equality for the shared bits
        valid = [x_L[ind_L[k]] == x_R[ind_R[k]] for k in range(len(ind_L))]

        # Append x_R to x_L
        if np.all(valid):
            x_R_new = ""
            for k in range(len(x_R)):
                if not k in ind_R:
                    x_R_new = x_R_new + x_R[k]
            x_new = x_L + x_R_new
            LR.xs = np.append(LR.xs, x_new)

        return LR

    def _get_louvian_communities_with_shared_nodes(self, group):

        assert isinstance(group, Group)

        # NetworkX graph for the group
        G = nx.convert_matrix.from_numpy_array(self.As[group.nodes][:, group.nodes])

        # Find community groups
        mapping = community.best_partition(G)

        # Group labels
        labels = np.array(list(mapping.values()))

        # Unique labels
        unique_labels = np.unique(labels)

        # Subgroup including shared nodes
        groups = []

        for i in range(len(unique_labels)):

            # New subgroup
            new_group = dict()

            # Indices of the assets in the subgroup i
            ind = np.where(labels == unique_labels[i])[0]

            # NetworkX subgraph
            g = G.subgraph(ind)

            # A set of shared nodes for the subgroup i
            shared_nodes_i = []

            # Check shared nodes
            for node in g.nodes:

                # Insert shared nodes
                if len(G.edges(node)) != len(g.edges(node)):
                    shared_nodes_i = np.array(
                        [int(v) for (u, v) in list(G.edges(node)) if (u, v) not in list(g.edges(node))])
                    ind = np.append(ind, shared_nodes_i)

            ind = np.unique(ind).astype('int')

            # Update group
            name = str(i) if group.id == "-1" else group.id + "_" + str(i)
            new_group = Group(name, group.nodes[ind])

            # Recursive reduction
            if new_group.size > self.max_community_size:

                smaller_groups = self._get_louvian_communities_with_shared_nodes(new_group)

                groups += smaller_groups

            else:
                groups += [new_group]

        return groups
