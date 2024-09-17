# tests/test_laplacian.py

import unittest
import numpy as np
from ase.build import bulk
from dw_distance import compute_laplacian
from dw_distance.structure_utils import create_bilayer


class TestLaplacian(unittest.TestCase):
    def setUp(self):
        # Create a 3x3 MoS2/WSe2 bilayer structure
        self.bilayer = create_bilayer('Mo', 'S', 'W', 'Se', size=(3, 3, 1))
        self.center_atom_types = [0]  # Only Mo atoms in the bottom layer

    def test_compute_laplacian_default(self):
        # Test computing the Laplacian matrix with default parameters
        laplacian = compute_laplacian(self.bilayer, center_atom_types=self.center_atom_types)
        # print laplacian matrix
        np.set_printoptions(precision=3, suppress=True)
        print("laplacian matrix:\n", np.array2string(laplacian, separator=', ', threshold=np.inf))
        # Check if the Laplacian matrix is symmetric
        self.assertTrue(np.allclose(laplacian, laplacian.T))
        # Check if the Laplacian matrix is positive semi-definite
        eigenvalues = np.linalg.eigvalsh(laplacian)
        self.assertTrue(np.all(np.isclose(eigenvalues, 0, atol=1e-15) | (eigenvalues > 0)))
        # Check if the Laplacian matrix has the correct shape
        self.assertEqual(laplacian.shape, (9, 9))  # 9 Mo atoms in the 3x3 bilayer

    def test_compute_laplacian_invalid_method(self):
        # Test that an invalid method raises a ValueError
        with self.assertRaises(ValueError):
            compute_laplacian(self.bilayer, method='INVALID', center_atom_types=self.center_atom_types)

if __name__ == '__main__':
    unittest.main()