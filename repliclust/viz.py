"""
Provides the built-in visualization features of `repliclust`.

**Functions**:
    :py:func:`plot`
        Plot a dataset.
"""

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap


def plot(X, y=None, dimensionality_reduction="tsne", dim_red_params={}, **plot_params):
    """ Plot a dataset with clusters. """
    plt.figure()

    if X.shape[1] == 2:
        T = X

    elif X.shape[1] > 2:
        if dimensionality_reduction=="tsne":
            tsne_model = TSNE(n_components=2, perplexity=30, **dim_red_params)
            T = tsne_model.fit_transform(X)
        elif dimensionality_reduction=="umap":
            umap_model = umap.UMAP(n_neighbors=30, n_components=2, **dim_red_params)
            T = umap_model.fit_transform(X)
        else:
            raise ValueError(
                "dimensionality_reduction should be one of 'tsne' or 'umap'" 
                + f" (found '{dimensionality_reduction}')"
            )

    elif X.shape[1] < 2:
        raise ValueError(f"dimensionality must be >=2 (found '{X.shape[1]}')")

    plt.scatter(T[:,0], T[:,1], c=y, **plot_params)
    
    if X.shape[1] == 2:
        plt.xlabel("X1")
        plt.ylabel("X2")
    elif dimensionality_reduction=="tsne":
        plt.xlabel("TSNE1")
        plt.ylabel("TSNE2")
    elif dimensionality_reduction=="umap":
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
    
    plt.show()