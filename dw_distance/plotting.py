# dw_distance/plotting.py

import numpy as np
import matplotlib.pyplot as plt

def plot_cost_matrix(cost_matrix, title='Cost Matrix'):
    """
    Plot the cost matrix as a heatmap.

    Parameters:
    - cost_matrix (np.ndarray): Cost matrix to plot.
    - title (str): Title of the plot.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(cost_matrix, interpolation='nearest', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Features B')
    plt.ylabel('Features A')
    plt.tight_layout()
    plt.show()

def plot_feature_distributions(features_A, features_B, title='Feature Distributions'):
    """
    Plot distributions of features using histograms.

    Parameters:
    - features_A (np.ndarray): Feature array for structure A.
    - features_B (np.ndarray): Feature array for structure B.
    - title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(features_A.flatten(), bins=50, alpha=0.5, label='Structure A')
    plt.hist(features_B.flatten(), bins=50, alpha=0.5, label='Structure B')
    plt.title(title)
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_coupling_matrix(coupling_matrix, title='Coupling Matrix'):
    """
    Plot the coupling matrix as a heatmap.

    Parameters:
    - coupling_matrix (np.ndarray): Coupling matrix from optimal transport.
    - title (str): Title of the plot.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(coupling_matrix, interpolation='nearest', cmap='plasma')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Target Distribution')
    plt.ylabel('Source Distribution')
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Import necessary modules
    from cost_matrix import compute_cost_matrix
    from featurization import generate_features
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

    # Split features
    num_atoms = len(bilayer)
    features_A = features[:num_atoms//2]
    features_B = features[num_atoms//2:]

    # Compute cost matrix
    cost_matrix = compute_cost_matrix(features_A, features_B)

    # Plot cost matrix
    plot_cost_matrix(cost_matrix, title='Cost Matrix between Layers')