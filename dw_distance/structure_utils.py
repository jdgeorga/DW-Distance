# dw_distance/structure_utils.py

import numpy as np
from ase.build import mx2
from ase.build import rotate  # Try importing from ase.build

def create_mx2_monolayer(metal, chalcogen, phase='2H', a=None, thickness=None):
    """
    Create a monolayer of a transition metal dichalcogenide (TMDC) using the mx2 function.

    Parameters:
    - metal (str): Symbol of the metal atom (e.g., 'Mo', 'W').
    - chalcogen (str): Symbol of the chalcogen atom (e.g., 'S', 'Se').
    - phase (str): Phase of the material ('2H' or '1T'). Default is '2H'.
    - a (float): Lattice constant 'a' in angstroms. If None, default values are used.
    - thickness (float): Thickness of the layer in angstroms. If None, default values are used.

    Returns:
    - structure (ase.Atoms): The monolayer structure.
    """
    # Default lattice parameters for common TMDCs
    default_params = {
        ('Mo', 'S'): {'a': 3.160, 'thickness': 3.127},
        ('W', 'Se'): {'a': 3.282, 'thickness': 3.340},
        # Add other materials as needed
    }
    if a is None or thickness is None:
        params = default_params.get((metal, chalcogen))
        if params is None:
            raise ValueError(f"Default parameters for {metal}{chalcogen} are not defined. Please specify 'a' and 'thickness'.")
        if a is None:
            a = params['a']
        if thickness is None:
            thickness = params['thickness']

    # Corrected function call to mx2
    structure = mx2(formula = f"{metal}{chalcogen}2", kind=phase, a=a, thickness=thickness)
    
    # Add atom_types array
    atom_types = np.zeros(len(structure), dtype=int)
    atom_types[structure.symbols == metal] = 0
    atom_types[structure.symbols == chalcogen] = 1
    atom_types[structure.positions[:, 2] < structure.cell[2, 2] / 2] = 2
    structure.arrays['atom_types'] = atom_types
    
    return structure

def create_bilayer(metal1, chalcogen1, metal2, chalcogen2, phase='2H', interlayer_distance=6.6, size=(1, 1, 1)):
    # Create the first monolayer
    layer1 = create_mx2_monolayer(metal1, chalcogen1, phase=phase)
    
    # Create the second monolayer
    layer2 = create_mx2_monolayer(metal2, chalcogen2, phase=phase)
    
    # Find the mean cell 
    new_cell = (layer1.cell + layer2.cell) / 2
    c = 30  # Set the z dimension to 30 Angstroms
    
    # Set the new cell parameters for both layers, scaling the atoms only in x and y
    layer1.set_cell(new_cell, scale_atoms=True)
    layer2.set_cell(new_cell, scale_atoms=True)
    
    # Move the second layer to the top
    layer2.translate([0, 0, (layer1.cell[2, 2] + layer2.cell[2, 2]) / 2 + interlayer_distance])
    
    # Combine the layers
    bilayer = layer1 + layer2
    bilayer.cell[2, 2] = c
    
    # Update atom_types for the bilayer before creating supercell
    atom_types = np.zeros(len(bilayer), dtype=int)
    atom_types[:len(layer1)] = layer1.arrays['atom_types']
    atom_types[len(layer1):] = layer2.arrays['atom_types'] + 3
    bilayer.arrays['atom_types'] = atom_types

    # Create a supercell based on the size parameter
    from ase.build import make_supercell
    bilayer = make_supercell(bilayer, [[size[0], 0, 0], [0, size[1], 0], [0, 0, 1]])
    
    # The atom_types array is automatically repeated when creating the supercell,
    # so we don't need to update it again

    return bilayer
