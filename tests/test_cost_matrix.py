# tests/test_cost_matrix.py

import unittest
import numpy as np
from ase.build import bulk
from ase import Atoms
from dw_distance import generate_features, compute_cost_matrix, compute_laplacian
from dw_distance.structure_utils import create_bilayer

class TestCostMatrix(unittest.TestCase):
    def setUp(self):
        # Create the original 3x3 MoS2/WSe2 bilayer structure
        self.bilayer1 = create_bilayer('Mo', 'S', 'W', 'Se', size=(3, 3, 1))
        
        # Create a second 3x3 bilayer with the top layer displaced
        self.bilayer2 = self.bilayer1.copy()
        
        # Identify atoms in the top layer by their atom types (3, 4, 5)
        top_layer_mask = np.isin(self.bilayer2.arrays['atom_types'], [3, 4, 5])
        
        # Displace the top layer by (1/3, 2/3) in fractional coordinates
        cell = self.bilayer2.get_cell()
        displacement = cell[0]/3 * (1/3) + cell[1]/3 * (2/3)
        
        # Apply the displacement only to atoms in the top layer
        self.bilayer2.positions[top_layer_mask] += displacement
        
        self.center_atom_types = [0]  # Only Mo atoms in the bottom layer
        
        # Generate features for both bilayers, considering only the bottom Mo atoms
        self.features1 = generate_features(self.bilayer1, center_atom_types=self.center_atom_types)
        self.features2 = generate_features(self.bilayer2, center_atom_types=self.center_atom_types)
        

    def test_compute_cost_matrix_default(self):
        # Test computing the cost matrix with default parameters
        
        cost_matrix = compute_cost_matrix(self.features1, self.features2)
        self.assertEqual(cost_matrix.shape, (9, 9))  # 9 Mo atoms in each 3x3 bilayer
        self.assertTrue(np.all(cost_matrix >= 0))
        print("Cost matrix (default):\n", np.round(cost_matrix, 3))

    def test_compute_cost_matrix_with_kernel(self):
        # Test computing the cost matrix with a Gaussian kernel
        from dw_distance import gaussian_kernel
        cost_matrix = compute_cost_matrix(self.features1, self.features2, kernel=gaussian_kernel)
        self.assertEqual(cost_matrix.shape, (9, 9))  # 9 Mo atoms in each 3x3 bilayer
        print("Cost matrix (with Gaussian kernel):\n", np.round(cost_matrix, 3))

    def test_compute_cost_matrix_with_laplacian(self):
        # Compute Laplacian matrices for each bilayer
        laplacian1 = compute_laplacian(self.bilayer1, center_atom_types=self.center_atom_types)
        laplacian2 = compute_laplacian(self.bilayer2, center_atom_types=self.center_atom_types)
        # Test computing the cost matrix with Laplacian scaling
        cost_matrix = compute_cost_matrix(
            self.features1, self.features2, use_laplacian=True,
            laplacian_A=laplacian1, laplacian_B=laplacian2
        )
        self.assertEqual(cost_matrix.shape, (9, 9))  # 9 Mo atoms in each 3x3 bilayer
        print("Cost matrix (with Laplacian):\n", np.round(cost_matrix, 3))

    def test_compute_cost_matrix_missing_laplacian(self):
        # Test that missing Laplacian matrices raises a ValueError
        with self.assertRaises(ValueError):
            compute_cost_matrix(self.features1, self.features2, use_laplacian=True)

if __name__ == '__main__':
    unittest.main()