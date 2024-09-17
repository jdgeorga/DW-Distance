# dw_distance/optimal_transport.py

import numpy as np
import ot
from .config import DEFAULT_OT_METHOD, DEFAULT_REGULARIZATION, DEFAULT_GW_LOSS
from .cost_matrix import compute_cost_matrix

def compute_ot_distance(features_A, features_B, method=None, reg=None, metric='euclidean', p=2, kernel=None,
                        use_laplacian=False, laplacian_A=None, laplacian_B=None, tau=1.0, **kwargs):
    """
    Compute the optimal transport distance between two distributions.

    Parameters:
    - features_A (np.ndarray): Feature array for structure A.
    - features_B (np.ndarray): Feature array for structure B.
    - method (str): Optimal transport method ('sinkhorn', 'emd', 'gromov_wasserstein').
    - reg (float): Regularization parameter for Sinkhorn algorithm.
    - metric (str): Distance metric to use (default 'euclidean').
    - p (int): Power to raise the distance metric (e.g., p=2 for squared distance).
    - kernel (callable): Kernel function to apply to the distances.
    - use_laplacian (bool): Whether to apply exponential Laplacian scaling.
    - laplacian_A (np.ndarray): Laplacian matrix for structure A.
    - laplacian_B (np.ndarray): Laplacian matrix for structure B.
    - tau (float): Diffusion time parameter for Laplacian scaling.
    - **kwargs: Additional parameters for the OT method.

    Returns:
    - distance (float): Computed optimal transport distance.
    """
    if method is None:
        method = DEFAULT_OT_METHOD
    if reg is None:
        reg = DEFAULT_REGULARIZATION

    n = features_A.shape[0]
    m = features_B.shape[0]
    a = np.ones(n) / n
    b = np.ones(m) / m

    if method.lower() in ['sinkhorn', 'emd']:
        # Compute cost matrix between A and B
        cost_matrix = compute_cost_matrix(features_A, features_B, metric=metric, p=p, kernel=kernel,
                                          use_laplacian=use_laplacian, laplacian_A=laplacian_A,
                                          laplacian_B=laplacian_B, tau=tau)

        assert np.all(np.isfinite(cost_matrix)), "Cost matrix contains non-finite values."
        assert np.all(cost_matrix >= 0), "Cost matrix contains negative values."

        if method.lower() == 'sinkhorn':
            distance = ot.sinkhorn2(a, b, cost_matrix, reg, **kwargs)
        else:  # method == 'emd'
            distance = ot.emd2(a, b, cost_matrix, **kwargs)
    
    elif method.lower() == 'gromov_wasserstein':
        # Compute separate cost matrices for A and B
        cost_matrix_A = compute_cost_matrix(features_A, features_A, metric=metric, p=p, kernel=kernel,
                                            use_laplacian=use_laplacian, laplacian_A=laplacian_A,
                                            laplacian_B=laplacian_A, tau=tau)
        cost_matrix_B = compute_cost_matrix(features_B, features_B, metric=metric, p=p, kernel=kernel,
                                            use_laplacian=use_laplacian, laplacian_A=laplacian_B,
                                            laplacian_B=laplacian_B, tau=tau)

        assert np.all(np.isfinite(cost_matrix_A)), "Cost matrix A contains non-finite values."
        assert np.all(cost_matrix_A >= 0), "Cost matrix A contains negative values."
        assert np.all(np.isfinite(cost_matrix_B)), "Cost matrix B contains non-finite values."
        assert np.all(cost_matrix_B >= 0), "Cost matrix B contains negative values."

        loss_fun = kwargs.get('loss_fun', DEFAULT_GW_LOSS)
        distance = ot.gromov_wasserstein2(cost_matrix_A, cost_matrix_B, a, b, loss_fun=loss_fun, **kwargs)
    
    else:
        raise ValueError(f"Optimal transport method '{method}' is not supported.")

    return distance

# Example usage
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

    # Split features into two parts
    num_atoms = len(bilayer)
    features_A = features[:num_atoms//2]
    features_B = features[num_atoms//2:]

    # Compute Laplacians for both parts
    laplacian_A = compute_laplacian(bilayer[:num_atoms//2])
    laplacian_B = compute_laplacian(bilayer[num_atoms//2:])

    # Compute OT distance
    distance = compute_ot_distance(features_A, features_B, method='sinkhorn', reg=0.01,
                                   use_laplacian=True, laplacian_A=laplacian_A, laplacian_B=laplacian_B)
    print("Sinkhorn Distance:", distance)