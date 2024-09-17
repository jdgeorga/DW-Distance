import unittest
import numpy as np
from dw_distance import generate_features, compute_ot_distance, compute_laplacian
from dw_distance.structure_utils import create_bilayer

class TestOptimalTransport(unittest.TestCase):
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
        
        # Compute Laplacians for both bilayers
        self.laplacian1 = compute_laplacian(self.bilayer1, center_atom_types=self.center_atom_types)
        self.laplacian2 = compute_laplacian(self.bilayer2, center_atom_types=self.center_atom_types)

    def test_compute_ot_distance_default(self):
        # Test computing the OT distance with default parameters
        distance = compute_ot_distance(self.features1, self.features2)
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, -1e-10)  # Allow for small negative values due to machine precision
        print("OT distance (default):", distance)

    def test_compute_ot_distance_with_regularization(self):
        # Test computing the OT distance with custom regularization
        distance = compute_ot_distance(self.features1, self.features2, method='sinkhorn', reg=0.05)
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, -1e-10)  # Allow for small negative values due to machine precision
        print("OT distance (sinkhorn, reg=0.05):", distance)

    def test_compute_ot_distance_with_laplacian(self):
        # Test computing the OT distance with Laplacian scaling
        distance = compute_ot_distance(self.features1, self.features2, method='sinkhorn', reg=0.01,
                                       use_laplacian=True, laplacian_A=self.laplacian1, laplacian_B=self.laplacian2)
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, -1e-10)  # Allow for small negative values due to machine precision
        print("OT distance (with Laplacian):", distance)

    def test_compute_ot_distance_emd(self):
        # Test computing the OT distance using EMD method
        distance = compute_ot_distance(self.features1, self.features2, method='emd')
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, -1e-10)  # Allow for small negative values due to machine precision
        print("OT distance (EMD):", distance)

    def test_compute_ot_distance_gromov_wasserstein(self):
        # Test computing the OT distance using Gromov-Wasserstein method
        distance = compute_ot_distance(self.features1, self.features2, method='gromov_wasserstein')
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, -1e-10)  # Allow for small negative values due to machine precision
        print("OT distance (Gromov-Wasserstein):", distance)

    def test_compute_ot_distance_gromov_wasserstein_with_laplacian(self):
        # Test computing the OT distance using Gromov-Wasserstein method with Laplacian scaling
        distance = compute_ot_distance(self.features1, self.features2, method='gromov_wasserstein',
                                       use_laplacian=True, laplacian_A=self.laplacian1, laplacian_B=self.laplacian2)
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, -1e-10)  # Allow for small negative values due to machine precision
        print("OT distance (Gromov-Wasserstein with Laplacian):", distance)

if __name__ == '__main__':
    unittest.main()