import random

import networkx as nx
import numpy as np

def sdcdp_model(degrees, probabilities, directed=False, simple=False):
    """
    Generates a network based on the Social Distance Configuration model with Degree Preservation (SDC-DP).

    See docs/sdcdp.md for documentation and full mathematical details.

    Parameters
    ----------
    degrees : array-like of int
        If `directed` is False, a degree sequence with shape (N,).
        If `directed` is True, an out- and in-degree sequence with shape (2, N).

    probabilities : array-like of float
        Edge formation probabilities with shape (N, N).
        If `directed` is False, must be symmetric.
        If `directed` is True, may be asymmetric.

    directed : bool, optional
        If False, assumes the network is undirected.
        If True, assumes the network is directed.
        Default is False.

    simple : bool, optional
        If False, may return self-loops or multiedges.
        If True, does not return self-loops or multiedges.
        Default is False.

    Returns
    -------
    G : nx.MultiGraph or nx.MultiDiGraph
        Generated network matching input constraints.
    """

    if not isinstance(degrees, (list, np.ndarray)):
        raise TypeError("Invalid Input: 'degrees' must be a list, list-of-lists, or NumPy array.")
    if not isinstance(probabilities, (list, np.ndarray)):
        raise TypeError("Invalid Input: 'probabilities' must be a list-of-lists or NumPy array.")
    
    degrees = np.array(degrees)
    P = np.array(probabilities)

    if not np.issubdtype(degrees.dtype, np.number) or np.any((degrees < 0) | (degrees % 1 != 0)):
        raise ValueError("Invalid Input: each entry in 'degrees' must be numeric and a non-negative integer.")
    if not np.issubdtype(P.dtype, np.number) or np.any((P < 0) | (P > 1)):
        raise ValueError("Invalid Input: each entry in 'probabilities' must be numeric and bounded between 0 and 1.")

    if directed and (degrees.ndim != 2 or degrees.shape[0] != 2):
        raise ValueError("Invalid Input: 'degrees' must have shape (2, N) for directed graphs.")
    if not directed and degrees.ndim != 1:
        raise ValueError("Invalid Input: 'degrees' must have shape (N,) for undirected graphs.")
    if directed and np.sum(degrees[0]) != np.sum(degrees[1]):
        raise ValueError("Invalid Input: sum of out-'degrees' must equal sum of in-'degrees' for directed graphs.")
    if not directed and np.sum(degrees) % 2 != 0:
        raise ValueError("Invalid Input: sum of 'degrees' must be even for undirected graphs.")
    
    number_of_edges = np.sum(degrees[0]) if directed else np.sum(degrees) // 2
    number_of_nodes = len(degrees[0]) if directed else len(degrees)
    nodes = np.arange(number_of_nodes)

    if P.shape != (number_of_nodes, number_of_nodes):
        raise ValueError("Invalid Input: 'probabilities' must have shape (N, N).")
    if not directed and not np.array_equal(P, P.T):
        raise ValueError("Invalid Input: 'probabilities' must be symmetric for undirected graphs.")
    
    degrees_o = np.array(degrees[0]) if directed else np.array(degrees)
    degrees_i = np.array(degrees[1]) if directed else degrees_o

    masked_nodes_o = degrees_o > 0
    masked_nodes_i = degrees_i > 0 if directed else masked_nodes_o

    P[~masked_nodes_o, :] = 0
    P[:, ~masked_nodes_i] = 0
    np.fill_diagonal(P, 0)

    candidates_o = np.count_nonzero(P, axis=1)
    candidates_i = np.count_nonzero(P, axis=0) if directed else candidates_o

    masked_nodes_o = candidates_o > 0
    masked_nodes_i = candidates_i > 0 if directed else masked_nodes_o
    has_valid_mask = np.any(masked_nodes_o) and np.any(masked_nodes_i) if directed else np.any(masked_nodes_o)

    G = nx.MultiDiGraph() if directed else nx.MultiGraph()
    G.add_nodes_from(nodes)

    while number_of_edges > 0:
        if has_valid_mask:
            degrees_c = degrees_o + degrees_i if directed else degrees_o
            node1 = random.choices(population=nodes[masked_nodes_o], cum_weights=np.cumsum(degrees_c[masked_nodes_o]), k=1)[0]
            degrees_o[node1] -= 1
            node2 = random.choices(population=nodes[masked_nodes_i], cum_weights=np.cumsum(P[node1, masked_nodes_i]), k=1)[0]
            degrees_i[node2] -= 1
            
            candidates_o[node1] -= 1
            candidates_i[node2] -= 1

            P[node1, node2] = 0
            if not directed:
                P[node2, node1] = 0
            
            if degrees_o[node1] == 0:
                candidates_o[node1] = 0
                candidates_i[P[node1, :] > 0] -= 1

                P[node1, :] = 0
                if not directed:
                    P[:, node1] = 0
            
            if degrees_i[node2] == 0:
                candidates_i[node2] = 0
                candidates_o[P[:, node2] > 0] -= 1

                P[:, node2] = 0
                if not directed:
                    P[node2, :] = 0
            
            masked_nodes_o = candidates_o > 0
            masked_nodes_i = candidates_i > 0 if directed else masked_nodes_o
            has_valid_mask = np.any(masked_nodes_o) and np.any(masked_nodes_i) if directed else np.any(masked_nodes_o)
            
            G.add_edge(node1, node2)
            number_of_edges -= 1

        elif not simple:
            if directed:
                stubs_remaining_o = []
                stubs_remaining_i = []
                for node, degree in enumerate(degrees_o):
                    stubs_remaining_o.extend([node] * degree)
                for node, degree in enumerate(degrees_i):
                    stubs_remaining_i.extend([node] * degree)
                random.shuffle(stubs_remaining_o)
                random.shuffle(stubs_remaining_i)
                
            else:
                stubs_remaining_c = []
                for node, degree in enumerate(degrees_o):
                    stubs_remaining_c.extend([node] * degree)
                random.shuffle(stubs_remaining_c)
                half = len(stubs_remaining_c) // 2
                stubs_remaining_o = stubs_remaining_c[:half]
                stubs_remaining_i = stubs_remaining_c[half:]
            
            G.add_edges_from(zip(stubs_remaining_o, stubs_remaining_i))
            number_of_edges = 0

        else:
            break

    return G