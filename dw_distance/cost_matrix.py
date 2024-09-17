# dw_distance/cost_matrix.py

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import expm
from .config import DEFAULT_TAU

def compute_cost_matrix(features_A, features_B, metric='euclidean', p=2, kernel=None,
                        use_laplacian=False, laplacian_A=None, laplacian_B=None, tau=1.0):
    """
    Compute the cost matrix between two sets of features.

    Parameters:
    - features_A (np.ndarray): Feature array for structure A.
    - features_B (np.ndarray): Feature array for structure B.
    - metric (str): Distance metric to use (default 'euclidean').
    - p (int): Power to raise the distance metric (e.g., p=2 for squared distance).
    - kernel (callable): Kernel function to apply to the distances.
    - use_laplacian (bool): Whether to apply exponential Laplacian scaling.
    - laplacian_A (np.ndarray): Laplacian matrix for structure A.
    - laplacian_B (np.ndarray): Laplacian matrix for structure B.
    - tau (float): Diffusion time parameter for Laplacian scaling.

    Returns:
    - cost_matrix (np.ndarray): Computed cost matrix.
    """

    # Use default tau if not provided
    if tau is None:
        tau = DEFAULT_TAU

    # Compute pairwise distance matrix
    distance_matrix = cdist(features_A, features_B, metric=metric) ** p

    # Apply kernel function if provided
    if kernel is not None:
        distance_matrix = kernel(distance_matrix)

    # Apply exponential Laplacian scaling if requested
    if use_laplacian:
        if laplacian_A is None or laplacian_B is None:
            raise ValueError("Laplacian matrices must be provided when use_laplacian is True.")

        # Compute diffusion operators
        diffusion_operator_A = expm(-tau * laplacian_A)
        diffusion_operator_B = expm(-tau * laplacian_B)

        print("Diffusion Operator A Shape:", diffusion_operator_A.shape)
        print("Diffusion Operator B Shape:", diffusion_operator_B.shape)

        # Apply diffusion to features
        diffused_features_A = diffusion_operator_A @ features_A
        diffused_features_B = diffusion_operator_B @ features_B

        # Recompute distance matrix with diffused features
        distance_matrix = cdist(diffused_features_A, diffused_features_B, metric=metric) ** p

        # Apply kernel function again if provided
        if kernel is not None:
            distance_matrix = kernel(distance_matrix)

    return distance_matrix

# Example kernel function
def gaussian_kernel(distance_matrix, sigma=1.0):
    return np.exp(-distance_matrix ** 2 / (2 * sigma ** 2))

if __name__ == "__main__":
    from featurization import generate_features
    from laplacian import compute_laplacian
    from ase.build import bulk
    from ase.build import make_supercell

    # Create MoS2 monolayer
    mos2 = bulk('MoS2', 'hexagonal', a=3.160, c=12.3)  # Lattice parameters for MoS2

    # Create WSe2 monolayer
    wse2 = bulk('WSe2', 'hexagonal', a=3.282, c=12.9)  # Lattice parameters for WSe2

    # Average lattice parameters
    a_avg = (mos2.get_cell()[0, 0] + wse2.get_cell()[0, 0]) / 2

    # Adjust both monolayers to have the average lattice constant
    mos2.set_cell([a_avg, a_avg, mos2.get_cell()[2, 2]], scale_atoms=True)
    wse2.set_cell([a_avg, a_avg, wse2.get_cell()[2, 2]], scale_atoms=True)

    # Stack the two monolayers to create a bilayer
    wse2.translate([0, 0, mos2.get_cell()[2, 2] / 2 + wse2.get_cell()[2, 2] / 2 + 3.3])  # Adjust interlayer spacing

    # Combine the two layers
    bilayer = mos2 + wse2

    # Generate features using SOAP
    features = generate_features(bilayer, method='SOAP')
    print("SOAP Features Shape:", features.shape)

    # Split features into two parts for demonstration (e.g., top and bottom layers)
    num_atoms = len(bilayer)
    features_A = features[:num_atoms//2]
    features_B = features[num_atoms//2:]

    # Compute Laplacians for both parts
    laplacian_A = compute_laplacian(bilayer[:num_atoms//2])
    laplacian_B = compute_laplacian(bilayer[num_atoms//2:])

    # Compute cost matrix with Laplacian scaling
    cost_matrix = compute_cost_matrix(features_A, features_B, use_laplacian=True,
                                      laplacian_A=laplacian_A, laplacian_B=laplacian_B, tau=1.0)

    print("Cost Matrix Shape:", cost_matrix.shape)