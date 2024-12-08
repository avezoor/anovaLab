import pytest

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as spgraph

from numpy.testing import assert_equal

try:
    import sparse
except Exception:
    sparse = None

pytestmark = pytest.mark.skipif(sparse is None,
                                reason="pydata/sparse not installed")


msg = "pydata/sparse (0.15.1) does not implement necessary operations"


sparse_params = (pytest.param("COO"),
                 pytest.param("DOK", marks=[pytest.mark.xfail(reason=msg)]))


@pytest.fixture(params=sparse_params)
def sparse_cls(request):
    return getattr(sparse, request.param)


@pytest.fixture
def graphs(sparse_cls):
    graph = [
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]
    A_dense = np.array(graph)
    A_sparse = sparse_cls(A_dense)
    return A_dense, A_sparse


@pytest.mark.parametrize(
    "func",
    [
        spgraph.shortest_path,
        spgraph.dijkstra,
        spgraph.floyd_warshall,
        spgraph.bellman_ford,
        spgraph.johnson,
        spgraph.reverse_cuthill_mckee,
        spgraph.maximum_bipartite_matching,
        spgraph.structural_rank,
    ]
)
def test_csgraph_equiv(func, graphs):
    A_dense, A_sparse = graphs
    actual = func(A_sparse)
    desired = func(sp.csc_matrix(A_dense))
    assert_equal(actual, desired)


def test_connected_components(graphs):
    A_dense, A_sparse = graphs
    func = spgraph.connected_components

    actual_comp, actual_labels = func(A_sparse)
    desired_comp, desired_labels, = func(sp.csc_matrix(A_dense))

    assert actual_comp == desired_comp
    assert_equal(actual_labels, desired_labels)


def test_laplacian(graphs):
    A_dense, A_sparse = graphs
    sparse_cls = type(A_sparse)
    func = spgraph.laplacian

    actual = func(A_sparse)
    desired = func(sp.csc_matrix(A_dense))

    assert isinstance(actual, sparse_cls)

    assert_equal(actual.todense(), desired.todense())


@pytest.mark.parametrize(
    "func", [spgraph.breadth_first_order, spgraph.depth_first_order]
)
def test_order_search(graphs, func):
    A_dense, A_sparse = graphs

    actual = func(A_sparse, 0)
    desired = func(sp.csc_matrix(A_dense), 0)

    assert_equal(actual, desired)


@pytest.mark.parametrize(
    "func", [spgraph.breadth_first_tree, spgraph.depth_first_tree]
)
def test_tree_search(graphs, func):
    A_dense, A_sparse = graphs
    sparse_cls = type(A_sparse)

    actual = func(A_sparse, 0)
    desired = func(sp.csc_matrix(A_dense), 0)

    assert isinstance(actual, sparse_cls)

    assert_equal(actual.todense(), desired.todense())


def test_minimum_spanning_tree(graphs):
    A_dense, A_sparse = graphs
    sparse_cls = type(A_sparse)
    func = spgraph.minimum_spanning_tree

    actual = func(A_sparse)
    desired = func(sp.csc_matrix(A_dense))

    assert isinstance(actual, sparse_cls)

    assert_equal(actual.todense(), desired.todense())


def test_maximum_flow(graphs):
    A_dense, A_sparse = graphs
    sparse_cls = type(A_sparse)
    func = spgraph.maximum_flow

    actual = func(A_sparse, 0, 2)
    desired = func(sp.csr_matrix(A_dense), 0, 2)

    assert actual.flow_value == desired.flow_value
    assert isinstance(actual.flow, sparse_cls)

    assert_equal(actual.flow.todense(), desired.flow.todense())


def test_min_weight_full_bipartite_matching(graphs):
    A_dense, A_sparse = graphs
    func = spgraph.min_weight_full_bipartite_matching

    actual = func(A_sparse[0:2, 1:3])
    desired = func(sp.csc_matrix(A_dense)[0:2, 1:3])

    assert_equal(actual, desired)
