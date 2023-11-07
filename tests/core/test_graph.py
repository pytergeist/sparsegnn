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


def test_add_edge_directed_graph():
    graph = Graph(directed=True)
    graph.add_node("A")
    graph.add_node("B")

    graph.add_edge("A", "B")
    assert ("A", "B") in graph.edges
    assert ("B", "A") not in graph.edges  # This will cover line 78


def test_remove_edge_directed_graph():
    graph = Graph(directed=True)
    graph.add_node("A")
    graph.add_node("B")

    graph.add_edge("A", "B")
    graph.remove_edge("B", "A")
    assert ("A", "B") in graph.edges
    graph.remove_edge("A", "B")
    assert ("A", "B") not in graph.edges


def test_adjacency_matrix_to_tensor_with_no_edges():
    graph = Graph()
    tensor = graph.adjacency_matrix_to_tensor()
    assert tensor is None


def test_get_edge_list_undirected():
    graph = Graph()
    graph.add_node(1)
    graph.add_node(2)
    graph.add_edge(1, 2)

    edge_list = graph.get_edge_list()
    assert (
        len(edge_list) == 2
    )  # Since the graph is undirected, (1, 2) and (2, 1) should both be present
    assert (1, 2) in edge_list
    assert (2, 1) in edge_list


def test_get_edge_list_directed():
    directed_graph = Graph(directed=True)
    directed_graph.add_node(1)
    directed_graph.add_node(2)
    directed_graph.add_edge(1, 2)

    directed_edge_list = directed_graph.get_edge_list()
    assert len(directed_edge_list) == 1  # Only one directed edge should be present
    assert (1, 2) in directed_edge_list
    assert (
        2,
        1,
    ) not in directed_edge_list  # This checks that the reverse edge is not present


if __name__ == "__main__":
    pytest.main()
