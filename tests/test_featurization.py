# tests/test_featurization.py

import unittest
import numpy as np
from dw_distance import generate_features
from dw_distance.structure_utils import create_bilayer
from ase.io import write

class TestFeaturization(unittest.TestCase):
    def setUp(self):
        # Create 3x3 MoS2/WSe2 bilayer using the utility function
        self.bilayer = create_bilayer('Mo', 'S', 'W', 'Se', size=(3, 3, 1))

        self.center_atom_types = [0]  # Mo atoms are center atoms

    def test_generate_features_default(self):
        # Test generating features with default parameters
        features = generate_features(self.bilayer, center_atom_types=self.center_atom_types)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], 9)  # 9 Mo atoms in the 3x3 bilayer
        self.assertGreater(features.shape[1], 0)

    def test_generate_features_with_positions(self):
        # Test generating features at specified positions (e.g., Mo atoms only)
        positions = [atom.index for atom in self.bilayer if self.bilayer.arrays['atom_types'][atom.index] in self.center_atom_types]
        print(positions)
        features = generate_features(self.bilayer, positions=positions)
        self.assertEqual(features.shape[0], 9)  # 9 Mo atoms in the 3x3 bilayer

    def test_generate_features_invalid_method(self):
        # Test that an invalid method raises a ValueError
        with self.assertRaises(ValueError):
            generate_features(self.bilayer, method='INVALID')

    def test_mo_atoms_identical_features(self):
        # Test that all Mo atoms have identical SOAP feature vectors
        features = generate_features(self.bilayer, center_atom_types=self.center_atom_types)        
        # Check that all rows in the features array are identical
        self.assertTrue(np.allclose(features, features[0]))

if __name__ == '__main__':
    unittest.main()