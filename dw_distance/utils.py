# dw_distance/utils.py

import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx

def construct_graph(structure, cutoff=3.5):
    """
    Construct a graph from an atomic structure based on a cutoff distance.

    Parameters:
    - structure (ase.Atoms): Atomic structure.
    - cutoff (float): Cutoff distance for connecting nodes.

    Returns:
    - G (networkx.Graph): Constructed graph.
    """
    positions = structure.get_positions()
    num_atoms = len(structure)
    G = nx.Graph()
    G.add_nodes_from(range(num_atoms))
    # Compute pairwise distances
    distance_matrix = squareform(pdist(positions))
    # Add edges based on cutoff
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if distance_matrix[i, j] <= cutoff:
                G.add_edge(i, j)
    return G

def normalize_features(features):
    """
    Normalize feature vectors to have zero mean and unit variance.

    Parameters:
    - features (np.ndarray): Feature array.

    Returns:
    - normalized_features (np.ndarray): Normalized feature array.
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    normalized_features = (features - mean) / (std + 1e-8)
    return normalized_features

def get_atomic_positions(structure, symbols=None):
    """
    Get positions of atoms, optionally filtering by atomic symbols.

    Parameters:
    - structure (ase.Atoms): Atomic structure.
    - symbols (list or str): Atomic symbols to include (e.g., 'Mo', ['Mo', 'S']).

    Returns:
    - positions (np.ndarray): Array of atomic positions.
    - indices (list): List of atom indices corresponding to the positions.
    """
    if symbols is None:
        positions = structure.get_positions()
        indices = [atom.index for atom in structure]
    else:
        if isinstance(symbols, str):
            symbols = [symbols]
        indices = [atom.index for atom in structure if atom.symbol in symbols]
        positions = structure.get_positions()[indices]
    return positions, indices

# Example usage
if __name__ == "__main__":
    from ase.build import bulk

    # Create MoS2 monolayer
    mos2 = bulk('MoS2', 'hexagonal', a=3.160, c=12.3)

    # Construct graph
    G = construct_graph(mos2, cutoff=3.5)
    print("Graph Nodes:", G.number_of_nodes())
    print("Graph Edges:", G.number_of_edges())