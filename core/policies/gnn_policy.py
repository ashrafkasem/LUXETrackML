import tensorflow as tf
import numpy as np

################################################################################################### Define Edge Network
class EdgeNet(tf.keras.layers.Layer):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """
    def __init__(self, name="EdgeNet", hid_dim=10):
        super(EdgeNet, self).__init__(name=name)

        self.layer = tf.keras.Sequential(
            [
                tf.keras.Input(
                    shape=(hid_dim + 3) * 2,
                ),
                tf.keras.layers.Dense(hid_dim, activation="tanh"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

    def call(self, X, Ri, Ro):
        bo = tf.sparse.sparse_dense_matmul(Ro, X, adjoint_a=True)
        bi = tf.sparse.sparse_dense_matmul(Ri, X, adjoint_a=True)

        # Shape of B = N_edges x 6 (2x (3 coordinates))
        # each row consists of two node that are possibly connected.
        B = tf.concat([bo, bi], axis=1)  # n_edges x 6, 3-> r,phi,z

        return self.layer(B)


# Define Node Network
class NodeNet(tf.keras.layers.Layer):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """
    def __init__(self, name="NodeNet", hid_dim=10):
        super(NodeNet, self).__init__(name=name)

        self.layer = tf.keras.Sequential(
            [
                tf.keras.Input(
                    shape=(hid_dim + 3) * 3,
                ),
                tf.keras.layers.Dense(hid_dim, activation="tanh"),
                tf.keras.layers.Dense(hid_dim, activation="tanh"),
            ]
        )

    def call(self, X, e, Ri, Ro):
        bo = tf.sparse.sparse_dense_matmul(Ro, X, adjoint_a=True)
        bi = tf.sparse.sparse_dense_matmul(Ri, X, adjoint_a=True)
        Rwo = Ro * e[:, 0]
        Rwi = Ri * e[:, 0]

        mi = tf.sparse.sparse_dense_matmul(Rwi, bo)
        mo = tf.sparse.sparse_dense_matmul(Rwo, bi)
        # Shape of M = N_nodes x 9 (3x (3 coordinates))
        # each row consists of a node and its 2 possible neigbours
        M = tf.concat([mi, mo, X], axis=1)

        return self.layer(M)


##################################################################################################
class GNN(tf.keras.Model):
    """
    A message-passing graph neural network model which performs
    binary classification of nodes.
    """
    def __init__(self):
        # Network definitions here
        super(GNN, self).__init__(name="GNN")
        self.InputNet = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    GNN.config["hid_dim"], input_shape=(3,), activation="tanh"
                )
            ],
            name="InputNet",
        )
        self.EdgeNet = EdgeNet(name="EdgeNet", hid_dim=GNN.config["hid_dim"])
        self.NodeNet = NodeNet(name="NodeNet", hid_dim=GNN.config["hid_dim"])
        self.n_iters = GNN.config["n_iters"]

    def call(self, graph_array):
        X, Ri, Ro = graph_array  # decompose the graph array
        H = self.InputNet(X)  # execute InputNet to produce hidden dimensions
        H = tf.concat([H, X], axis=1)  # add new dimensions to original X matrix
        for i in range(self.n_iters):  # recurrent iteration of the network
            e = self.EdgeNet(H, Ri, Ro)  # execute EdgeNet
            H = self.NodeNet(H, e, Ri, Ro)  # execute NodeNet using the output of EdgeNet
            H = tf.concat([H, X], axis=1)  # update H with the output of NodeNet
        e = self.EdgeNet(H, Ri, Ro)  # execute EdgeNet one more time to obtain edge predictions
        return e  # return edge prediction array