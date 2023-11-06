# test_graph.py

import pytest
from sparsegnn.core import (
    Graph,
)


def test_add_and_remove_node():
    graph = Graph()
    assert len(graph.nodes) == 0

    graph.add_node(1)
    assert 1 in graph.nodes
    assert len(graph.nodes) == 1

    graph.remove_node(1)
    assert 1 not in graph.nodes
    assert len(graph.nodes) == 0


def test_add_and_remove_edge():
    graph = Graph()
    graph.add_node(1)
    graph.add_node(2)

    graph.add_edge(1, 2)
    assert (1, 2) in graph.edges
    assert graph.edges[(1, 2)] == 1.0

    graph.remove_edge(1, 2)
    assert (1, 2) not in graph.edges


def test_edge_weight():
    graph = Graph()
    graph.add_node(1)
    graph.add_node(2)

    graph.add_edge(1, 2, weight=0.5)
    assert graph.edges[(1, 2)] == 0.5


def test_adjacency_matrix():
    graph = Graph()
    graph.add_node(1)
    graph.add_node(2)
    graph.add_edge(1, 2)

    adjacency_matrix = graph.get_adjacency_matrix()
    assert adjacency_matrix is not None

    dense_matrix = adjacency_matrix.todense()
    assert dense_matrix[0, 1] == 1


def test_adjacency_matrix_to_tensor():
    graph = Graph()
    graph.add_node(1)
    graph.add_node(2)
    graph.add_edge(1, 2)

    tensor = graph.adjacency_matrix_to_tensor()
    assert tensor is not None
    assert tensor.shape.as_list() == [2, 2]


def test_directed_graph_edge():
    graph = Graph(directed=True)
    graph.add_node(1)
    graph.add_node(2)

    graph.add_edge(1, 2)
    assert (1, 2) in graph.edges
    assert (2, 1) not in graph.edges


if __name__ == "__main__":
    pytest.main()
