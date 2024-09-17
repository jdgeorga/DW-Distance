# dw_distance/config.py

# Default parameters for descriptors
DEFAULT_DESCRIPTOR_METHOD = 'SOAP'
DEFAULT_SOAP_PARAMS = {
    'r_cut': 5.0,
    'n_max': 8,
    'l_max': 6,
    'sigma': 0.1,
    'periodic': True,
    'sparse': False
    # 'species' can be set dynamically based on the structure
}

# Default parameters for Laplacian computation
DEFAULT_LAPLACIAN_METHOD = 'graph'
DEFAULT_CUTOFF_DISTANCE = 3.5  # in angstroms

# Default parameters for optimal transport
DEFAULT_OT_METHOD = 'sinkhorn'
DEFAULT_REGULARIZATION = 1e-2

# Default diffusion time
DEFAULT_TAU = 1.0

# Default parameters for Gromov-Wasserstein
DEFAULT_GW_LOSS = 'square_loss'

# Default metric for cost matrix computation
DEFAULT_METRIC = 'euclidean'

# Default power for distance metric
DEFAULT_P = 2