# tests/test_plotting.py

import unittest
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for testing
from dw_distance import generate_features, compute_cost_matrix
from dw_distance import plot_cost_matrix, plot_feature_distributions, plot_coupling_matrix
from dw_distance.structure_utils import create_bilayer

class TestPlotting(unittest.TestCase):
    def setUp(self):
        # Create a minimal MoS2/WSe2 bilayer structure
        self.bilayer = create_bilayer('Mo', 'S', 'W', 'Se')

        # Generate features for the bilayer
        self.features = generate_features(self.bilayer)

        # Split features into two parts representing the two layers
        num_atoms = len(self.bilayer)
        self.features_A = self.features[:num_atoms // 2]
        self.features_B = self.features[num_atoms // 2:]

        # Compute cost matrix
        self.cost_matrix = compute_cost_matrix(self.features_A, self.features_B)

    def test_plot_cost_matrix(self):
        # Test plotting the cost matrix
        try:
            plot_cost_matrix(self.cost_matrix, title='Test Cost Matrix')
        except Exception as e:
            self.fail(f"plot_cost_matrix raised an exception: {e}")

    def test_plot_feature_distributions(self):
        # Test plotting feature distributions
        try:
            plot_feature_distributions(self.features_A, self.features_B, title='Test Feature Distributions')
        except Exception as e:
            self.fail(f"plot_feature_distributions raised an exception: {e}")

    def test_plot_coupling_matrix(self):
        # Since we don't have a real coupling matrix, we can simulate one
        import numpy as np
        coupling_matrix = np.random.rand(len(self.features_A), len(self.features_B))
        try:
            plot_coupling_matrix(coupling_matrix, title='Test Coupling Matrix')
        except Exception as e:
            self.fail(f"plot_coupling_matrix raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()