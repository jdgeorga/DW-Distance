# dw_distance/__init__.py

"""
DW-Distance Package

This package provides tools for comparing atomic structures using the Diffusion Wasserstein Distance.
"""

# Import key functions and classes from submodules
from .featurization import generate_features
from .cost_matrix import compute_cost_matrix, gaussian_kernel
from .laplacian import compute_laplacian
from .optimal_transport import compute_ot_distance
from .plotting import plot_cost_matrix, plot_feature_distributions, plot_coupling_matrix
from .utils import construct_graph, normalize_features, get_atomic_positions
from .structure_utils import create_mx2_monolayer, create_bilayer
from .config import (
    DEFAULT_DESCRIPTOR_METHOD,
    DEFAULT_SOAP_PARAMS,
    DEFAULT_LAPLACIAN_METHOD,
    DEFAULT_CUTOFF_DISTANCE,
    DEFAULT_OT_METHOD,
    DEFAULT_REGULARIZATION,
    DEFAULT_TAU,
)

# Define package version
__version__ = '0.1.0'

# Define all available functions and classes
__all__ = [
    'generate_features',
    'compute_cost_matrix',
    'compute_laplacian',
    'compute_ot_distance',
    'plot_cost_matrix',
    'plot_feature_distributions',
    'plot_coupling_matrix',
    'construct_graph',
    'normalize_features',
    'get_atomic_positions',
    'gaussian_kernel',
    'create_mx2_monolayer',
    'create_bilayer',
    # Configuration constants
    'DEFAULT_DESCRIPTOR_METHOD',
    'DEFAULT_SOAP_PARAMS',
    'DEFAULT_LAPLACIAN_METHOD',
    'DEFAULT_CUTOFF_DISTANCE',
    'DEFAULT_OT_METHOD',
    'DEFAULT_REGULARIZATION',
    'DEFAULT_TAU',
]