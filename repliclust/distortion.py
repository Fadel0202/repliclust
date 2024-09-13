"""
Provides functions for nonlinearly distorting datasets, so that clusters
become non-convex and take on more irregular shapes beyond ellipsoids.

**Functions**:
    :py:func:`distort`
        Distort a dataset.
    :py:func:`project_to_sphere`
        Apply stereographic projection to make the data directional.
"""

import numpy as np
import torch
from torch import nn
from scipy.stats import ortho_group


def construct_near_ortho_matrix(hidden_dim, scaling_factor=0.1):
    axes = ortho_group.rvs(hidden_dim)
    logs = np.random.normal(loc=0, scale=scaling_factor, size=hidden_dim)
    scalings = np.exp(logs - np.mean(logs))
    sign = np.random.choice(a=[-1,1], p=[0.5,0.5])
    assert np.allclose(np.prod(scalings), 1.0)
    return sign * torch.tensor(np.transpose(axes) @ np.diag(scalings) @ axes, dtype=torch.float32)


class NeuralNetwork(nn.Module):
    def __init__(self, hidden_dim=64, dim=2, n_layers=50):
        super().__init__()

        embedding = nn.Linear(dim, hidden_dim)
        projection = nn.Linear(hidden_dim, dim)
        # Uncomment line below if you want to tie the projection and embeddings weights:
        projection.weight = nn.Parameter(embedding.weight.T)

        middle_layers = []
        for i in range(n_layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            middle_layers.append(layer)
            middle_layers.append(nn.LayerNorm(hidden_dim))
            middle_layers.append(nn.Tanh())

        self.middle_stack = nn.Sequential(
            embedding,
            *(middle_layers),
            projection
        )

    def forward(self, x):
        x_transformed = self.middle_stack(x)
        return x_transformed
    

def distort(X, hidden_dim=128, n_layers=16, device="cuda", set_seed=None):
    """ Distort dataset with random neural network to make clusters take on irregular shapes. """
    if not torch.cuda.is_available():
        device = "cpu"
        print("Switched to CPU because CUDA is not available.")

    if set_seed is not None:
        torch.manual_seed(set_seed)

    dim = X.shape[1]
    random_nn = NeuralNetwork(hidden_dim=hidden_dim, dim=dim, n_layers=n_layers).to(device)

    max_length = np.sqrt(np.max(np.sum(X**2,axis=1)))
    X_norm = X/max_length

    with torch.no_grad():
        X_tensor = torch.tensor(X_norm.astype('float32')).to(device)
        X_tf = random_nn(X_tensor).cpu()
    
    return X_tf


def wrap_around_sphere(X):
    """ Apply stereographic projection to make the data directional. """
    lengths = np.sqrt(np.sum(X**2,axis=1))
    l_max = np.max(lengths)

    # normalize data so that the maximum length is 1
    X_tf = X/l_max

    # carry out the stereographic projection to yield x_full on the sphere
    partial_squared_norms = np.sum(X_tf**2, axis=1)
    x_p = (1 - partial_squared_norms) / (1 + partial_squared_norms)
    x_rest = X_tf * (1 + x_p[:, np.newaxis])
    x_full = np.concatenate([x_rest, x_p[:, np.newaxis]], axis=1)
    assert np.allclose(np.sum(x_full**2, axis=1), 1.0)

    return x_full