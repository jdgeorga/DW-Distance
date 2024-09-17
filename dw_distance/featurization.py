# dw_distance/featurization.py

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP
from .config import DEFAULT_DESCRIPTOR_METHOD, DEFAULT_SOAP_PARAMS
from sklearn.preprocessing import normalize

# Import other descriptors as needed, e.g., CoulombMatrix, ACSF

def generate_features(structure, method=None, positions=None, center_atom_types=[0], **kwargs):
    """
    Generate high-dimensional feature vectors for a given structure.

    Parameters:
    - structure (ase.Atoms): Atomic structure.
    - method (str): Descriptor method ('SOAP', 'CoulombMatrix', etc.).
    - positions (array-like): Positions at which to calculate descriptors.
    - center_atom_types (list): Types of atoms to be considered as centers for SOAP. Default is [0].
    - **kwargs: Additional parameters for the descriptor.

    Returns:
    - features (np.ndarray): Array of feature vectors.
    """

    if method is None:
        method = DEFAULT_DESCRIPTOR_METHOD

    if method.upper() == 'SOAP':
        # Set default parameters from config.py, override with kwargs if provided
        soap_params = DEFAULT_SOAP_PARAMS.copy()
        soap_params.update(kwargs)

        # Create a copy of the structure to modify
        modified_structure = structure.copy()

        # Modify the atomic numbers to be atom_types + 1
        modified_structure.set_atomic_numbers(structure.arrays['atom_types'] + 1)

        # Collapse the z-axis for all atoms
        atom_positions = modified_structure.get_positions()
        atom_positions[:, 2] = 0
        modified_structure.set_positions(atom_positions)

        # Ensure 'species' is set based on the modified atomic numbers
        if 'species' not in soap_params:
            soap_params['species'] = list(set(modified_structure.get_atomic_numbers()))
        else:
            soap_params['species'] = [s + 1 for s in soap_params['species']]

        # Create SOAP descriptor with parameters
        soap = SOAP(**soap_params)
        # Positions at which to compute the descriptors
        if positions is None:
            # Select only Mo atoms (atom_types 0) for feature generation
            positions = [atom.index for atom in structure if structure.arrays['atom_types'][atom.index] in center_atom_types]
        # Compute the SOAP features
        features = soap.create(modified_structure, centers=positions)
        
        # # Print the first 10 columns of each row of the features
        # print("First 10 columns of SOAP Features:")
        # for row in features:
        #     print(row[:10])
        
        # Normalize the features
        features = normalize(features, axis=1)

        return features

    elif method.upper() == 'COULOMBMATRIX':
        # Placeholder for Coulomb Matrix descriptor
        from dscribe.descriptors import CoulombMatrix
        n_atoms_max = kwargs.get('n_atoms_max', len(structure))
        # Create Coulomb Matrix descriptor
        cm = CoulombMatrix(n_atoms_max=n_atoms_max, permutation='sorted_l2')
        features = cm.create(structure)

        # Normalize the features
        features = normalize(features, axis=1)

        return features

    else:
        raise ValueError(f"Descriptor method '{method}' is not supported.")

# Example usage within the module (can be removed if not needed)
if __name__ == "__main__":
    # Import necessary modules
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