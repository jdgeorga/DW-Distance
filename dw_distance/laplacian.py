# dw_distance/laplacian.py

import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csgraph
from .config import DEFAULT_LAPLACIAN_METHOD, DEFAULT_CUTOFF_DISTANCE

def compute_laplacian(structure, method='graph', center_atom_types=[0], **kwargs):
    """
    Compute the Laplacian matrix for a given structure, considering only center atom types.

    Parameters:
    - structure (ase.Atoms): Atomic structure.
    - method (str): Method to compute the Laplacian ('graph', 'distance').
    - center_atom_types (list): Types of atoms to be considered as centers. Default is [0].
    - **kwargs: Additional parameters (e.g., 'cutoff' for graph method, 'sigma' for distance method).

    Returns:
    - laplacian (np.ndarray): Laplacian matrix.
    """

    if method is None:
        method = DEFAULT_LAPLACIAN_METHOD
    
    positions = structure.get_positions()
    atom_types = structure.arrays['atom_types']
    
    # Filter positions and indices based on center_atom_types
    center_indices = [i for i, at in enumerate(atom_types) if at in center_atom_types]
    center_positions = positions[center_indices]
    num_centers = len(center_indices)

    if method == 'graph':
        cutoff = kwargs.get('cutoff', DEFAULT_CUTOFF_DISTANCE)
        G = nx.Graph()
        G.add_nodes_from(range(num_centers))
        
        # Compute pairwise distances for center atoms only, considering periodic boundary conditions
        distance_matrix = structure.get_all_distances(mic=True)[center_indices][:, center_indices]
        
        # Add edges based on cutoff
        for i in range(num_centers):
            for j in range(i + 1, num_centers):
                if distance_matrix[i, j] <= cutoff:
                    G.add_edge(i, j)
        
        # Get adjacency matrix
        adjacency_matrix = nx.adjacency_matrix(G).astype(float)
        
        # Compute Laplacian
        laplacian = csgraph.laplacian(adjacency_matrix, normed=True)
        
        return laplacian.toarray()

    elif method == 'distance':
        # Alternative method based on distance weighting
        sigma = kwargs.get('sigma', 1.0)
        
        # Compute weight matrix for center atoms only, considering periodic boundary conditions
        distance_matrix = structure.get_all_distances(mic=True)[center_indices][:, center_indices]
        weight_matrix = np.exp(-distance_matrix ** 2 / (2 * sigma ** 2))
        np.fill_diagonal(weight_matrix, 0)
        
        # Compute Laplacian
        laplacian = csgraph.laplacian(weight_matrix, normed=True)
        return laplacian

    else:
        raise ValueError(f"Method '{method}' not supported for Laplacian computation.")

# Example usage
if __name__ == "__main__":
    from ase.build import bulk
    from dw_distance.structure_utils import create_bilayer

    # Create MoS2/WSe2 bilayer
    bilayer = create_bilayer('Mo', 'S', 'W', 'Se')

    # Compute Laplacian for the bilayer, considering only Mo and W atoms (types 0 and 3)
    laplacian = compute_laplacian(bilayer, method='graph', center_atom_types=[0, 3], cutoff=3.0)
    print("Laplacian Matrix Shape:", laplacian.shape)