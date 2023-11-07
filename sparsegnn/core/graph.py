# graph.py

import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix


class Graph:
    """Graph data structure, supporting undirected or directed graphs.

    Attributes:
        nodes (set): A set of nodes in the graph.
        edges (dict): A dictionary with keys as tuples representing edges (node1, node2)
                      and values as weights of the edges.
        is_directed (bool): Whether the graph is directed.
    """

    def __init__(self, directed=False):
        """Initialize the graph.

        Args:
            directed (bool): If True, the graph will be treated as directed.
        """
        self.nodes = set()
        self.edges = {}  # key: (node1, node2), value: weight
        self.is_directed = directed

    def add_node(self, node):
        """Add a node to the graph.

        Args:
            node: The node to add.
        """
        self.nodes.add(node)

    def add_edge(self, node1, node2, weight=1.0):
        """Add an edge to the graph.

        Args:
            node1: The starting node of the edge.
            node2: The ending node of the edge.
            weight (float): The weight of the edge.
        """
        self.nodes.add(node1)
        self.nodes.add(node2)
        self.edges[(node1, node2)] = weight
        if not self.is_directed:
            self.edges[(node2, node1)] = weight

    def remove_node(self, node):
        """Remove a node and its associated edges from the graph.

        Args:
            node: The node to remove.
        """
        self.nodes.discard(node)
        self.edges = {
            edge: weight for edge, weight in self.edges.items() if node not in edge
        }

    def remove_edge(self, node1, node2):
        """Remove an edge from the graph.

        Args:
            node1: The starting node of the edge.
            node2: The ending node of the edge.
        """
        self.edges.pop((node1, node2), None)
        if not self.is_directed:
            self.edges.pop((node2, node1), None)

    def get_edge_list(self):
        """Get a list of edges in the graph.

        Returns:
            list: A list of tuples (node1, node2) representing the edges.
        """
        return list(self.edges.keys())

    def get_adjacency_matrix(self):
        """Get the adjacency matrix of the graph as a COO matrix.

        Returns:
            scipy.sparse.coo_matrix: The adjacency matrix in COO format.
        """
        if not self.nodes:
            return None

        # Create a mapping from node to a unique index
        node_to_index = {node: idx for idx, node in enumerate(sorted(self.nodes))}
        rows, cols, data = [], [], []

        for (node1, node2), weight in self.edges.items():
            row = node_to_index[node1]
            col = node_to_index[node2]
            rows.append(row)
            cols.append(col)
            data.append(weight)

        # Create a sparse matrix in COOrdinate format
        num_nodes = len(self.nodes)
        coo = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        return coo

    def adjacency_matrix_to_tensor(self):
        """Convert the adjacency matrix to a TensorFlow SparseTensor.

        Returns:
            tensorflow.SparseTensor: The adjacency matrix as a SparseTensor.
        """
        coo = self.get_adjacency_matrix()
        if coo is None:
            return None

        indices = np.array([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)
