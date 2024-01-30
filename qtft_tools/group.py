import numpy as np


# Groups — attributes are candidate solutions with probs & bitstrings
class Group:
    """

    Parameters

    ----------

    name : string,
        Group name.

    nodes : np.ndarray,
        A numpy array containing assets in the group.


    Attributes

    -------

    size : int,
        The number of assets.

    xs : np.ndarray,
        A numpy array containing candidate bitstrings.

    ps : np.ndarray,
        A numpy array containing probabilities associated with xs.

    cs : np.ndarray,
        A numpy array containing cost functions associated with cs.

    x_exact : string,
        The exact best bitstring.

    c_exact : float,
        The exact minimum cost function.

    """

    def __init__(self, name, nodes):

        assert isinstance(name, str)
        assert isinstance(nodes, np.ndarray)

        self.id = name
        self.nodes = nodes

        self.size = len(nodes)
        self.xs = np.array([])
        self.ps = np.array([])
        self.cs = np.array([])
        self.x_exact = None
        self.c_exact = None
        self.c_qaoa = None

    def approximation_ratio(self):
        return self.c_qaoa / self.c_exact

    def show(self):

        print("id:" + self.id + ", nodes:" + str(self.nodes) + ", size:" + str(self.size))

        if self.xs:
            print("x_exact:" + self.x_exact + ", c_exact:%.5f" % self.c_exact)
            for i in range(len(self.xs)):
                print("\tx:" + self.xs[i] + ", p:%.5f" % self.ps[i] + ", E:%.5f" % self.cs[i])

    def to_dict(self):
        res = self.__dict__.copy()
        res['nodes'] = self.nodes.tolist()
        res['xs'] = self.xs.copy().tolist()
        res['ps'] = self.ps.copy().tolist()
        res['cs'] = self.cs.copy().tolist()
        return res

    @classmethod
    def from_dict(cls, input_dict):
        name = input_dict['id']
        nodes = input_dict['nodes']
        res = Group(name=name, nodes=np.array(nodes))
        res.xs = np.array(input_dict['xs'])
        res.ps = np.array(input_dict['ps'])
        res.cs = np.array(input_dict['cs'])
        res.x_exact = input_dict['x_exact']
        res.c_exact = input_dict['c_exact']
        res.c_qaoa = input_dict['c_qaoa']

        return res


