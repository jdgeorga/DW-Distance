# Usage Guide

This guide provides detailed instructions on how to use the DW-Distance tool for comparing atomic structures.

## Prerequisites

- Python 3.6 or higher
- Required packages installed (see )

## Installation

Install the package by running:

```bash
pip install -e .
```

## Featurization

Use the  module to generate feature vectors for your structures. The default method is SOAP, but you can specify other descriptors.

```python
from dw_distance.featurization import generate_features

features = generate_features(structure, method='SOAP', **kwargs)
```

## Computing the Cost Matrix

Compute the cost matrix between two sets of features using the  module.

```python
from dw_distance.cost_matrix import compute_cost_matrix

cost_matrix = compute_cost_matrix(features_A, features_B, metric='euclidean', use_laplacian=True, laplacian=laplacian)
```

## Calculating the Laplacian

Use the  module to compute the Laplacian matrix of a structure.

```python
from dw_distance.laplacian import compute_laplacian

laplacian = compute_laplacian(structure)
```

## Optimal Transport Calculation

Perform optimal transport calculations using the  module.

```python
from dw_distance.optimal_transport import compute_sinkhorn_distance

distance = compute_sinkhorn_distance(cost_matrix, reg=0.01)
```

## Generating Plots

Visualize your results using the  module.

```python
from dw_distance.plotting import plot_cost_matrix

plot_cost_matrix(cost_matrix)
```

## Examples

Refer to the scripts in the  directory for complete examples on how to use the tool in different scenarios.

## Support

For questions or support, please contact johnathangeorgaras@gmail.com

