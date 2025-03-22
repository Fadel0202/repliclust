import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
from functools import lru_cache
from scipy.stats import norm
from tqdm import tqdm

from .base import Archetype
from .maxmin.archetype import MaxMinArchetype
from .overlap._gradients import update_centers, compute_quantiles
from .utils import assemble_covariance_matrix


class ArchetypeOptimized(MaxMinArchetype):
    """Version optimisée de la classe Archetype avec parallélisation."""
    
    def __init__(self, n_clusters=5, dim=2, n_samples=1000,
                 aspect_ref=3.0, radius_maxmin=2.0,
                 max_overlap=0.001, min_overlap=0.0001,
                 use_cache=True, n_workers=None, overlap_mode='lda',
                 learning_rate=0.1, linear_penalty_weight=0.5,
                 max_epoch=50):
        super().__init__(n_clusters=n_clusters, 
                        dim=dim,
                        n_samples=n_samples,
                        aspect_ref=aspect_ref,
                        radius_maxmin=radius_maxmin,
                        max_overlap=max_overlap,
                        min_overlap=min_overlap)
                    
        self.use_cache = use_cache
        self.n_workers = n_workers or min(os.cpu_count(), 4)  # Limitons le nombre de workers
        self.overlap_mode = overlap_mode
        self.learning_rate = learning_rate
        self.linear_penalty_weight = linear_penalty_weight
        self.max_epoch = max_epoch
        
        # Convertir les valeurs de chevauchement en quantiles
        self.quantile_bounds = {
            'min': norm.ppf(1 - max_overlap/2),
            'max': norm.ppf(1 - min_overlap/2)
        }
    
    def _create_covariance_matrices(self):
        """Crée les matrices de covariance à partir des axes et longueurs."""
        if not hasattr(self, '_axes') or not hasattr(self, '_lengths'):
            self._axes, self._lengths = self.covariance_sampler.sample_covariances(self)
            
        cov_list = []
        for i in range(self.n_clusters):
            cov_matrix = assemble_covariance_matrix(
                self._axes[i], 
                self._lengths[i], 
                inverse=False
            )
            cov_list.append(cov_matrix)
            
        return cov_list
    
    def _create_average_cov_inv_list(self, cov_list):
        """Crée la liste des matrices de covariance moyennes inversées."""
        if self.overlap_mode != 'lda':
            return None
            
        ave_cov_inv_list = []
        for i in range(self.n_clusters):
            for j in range(i+1, self.n_clusters):
                try:
                    ave_cov = (cov_list[i] + cov_list[j]) / 2
                    ave_cov_inv = np.linalg.inv(ave_cov)
                    ave_cov_inv_list.append(ave_cov_inv)
                except np.linalg.LinAlgError:
                    # Fallback si l'inversion échoue
                    print(f"Avertissement: Échec d'inversion pour clusters {i} et {j}, utilisation d'une matrice identité")
                    ave_cov_inv_list.append(np.eye(self.dim))
                    
        return ave_cov_inv_list
    
    def _initialize_centers(self):
        """Initialise les centres de cluster aléatoirement."""
        centers = np.random.randn(self.n_clusters, self.dim)
        
        # Mise à l'échelle pour une meilleure séparation initiale
        scale_factor = np.sqrt(self.dim) * self.scale
        centers *= scale_factor
        
        return centers
    
    def optimize_centers(self, quiet=False):
        """Optimise les centres de cluster pour respecter les contraintes de chevauchement."""
        # Création des matrices de covariance
        cov_list = self._create_covariance_matrices()
        
        # Création des matrices de covariance moyennes inversées si nécessaire
        ave_cov_inv_list = self._create_average_cov_inv_list(cov_list)
        
        # Initialisation des centres
        centers = self._initialize_centers()
        
        # Optimisation des centres
        if self.n_clusters > 1:
            # Version simple et robuste pour tous les cas
            for epoch in range(self.max_epoch):
                for i in range(self.n_clusters):
                    try:
                        # Obtenir les autres indices de cluster
                        other_cluster_idx = [j for j in range(self.n_clusters) if j != i]
                        
                        # Appel correct à update_centers
                        grad, loss = update_centers(
                            ref_cluster_idx=i,
                            centers=centers,
                            cov_list=cov_list,
                            ave_cov_inv_list=ave_cov_inv_list,
                            axis_deriv_t_list=ave_cov_inv_list,  # Même liste pour simplifier
                            learning_rate=self.learning_rate,
                            linear_penalty_weight=self.linear_penalty_weight,
                            quantile_bounds=self.quantile_bounds,
                            mode=self.overlap_mode
                        )
                    except Exception as e:
                        if not quiet:
                            print(f"Erreur lors de l'optimisation du centre {i}: {str(e)}")
                
        return centers
        
    def sample_mixture_model(self, quiet=False):
        """
        Implémentation optimisée pour échantillonner un modèle de mélange probabiliste.
        """
        # Échantillonner les covariances (axes et longueurs d'axes)
        self._axes, self._lengths = self.covariance_sampler.sample_covariances(self)
        
        # Construire la liste des matrices de covariance
        cov_list = self._create_covariance_matrices()
        
        # Échantillonner les centres de clusters
        if self.n_clusters == 1:
            self._centers = np.zeros((1, self.dim))
        else:
            # Optimiser les centres
            self._centers = self.optimize_centers(quiet=quiet)
        
        # Assigner les distributions
        self._distributions = self.distribution_mix.assign_distributions(self.n_clusters)
        
        # Construire le modèle de mélange
        from repliclust.base import MixtureModel
        mixture_model = MixtureModel(
            self._centers, 
            self._axes, self._lengths, 
            self._distributions
        )
        
        return mixture_model
    
    def synthesize(self, n_samples=None, quiet=False):
        """
        Méthode améliorée pour synthétiser des données directement.
        """
        # Mise à jour du nombre d'échantillons si spécifié
        if n_samples is not None:
            self.n_samples = n_samples
            
        # Génération du modèle de mélange
        mixture_model = self.sample_mixture_model(quiet=quiet)
        
        # Échantillonnage des tailles de groupe
        group_sizes = self.groupsize_sampler.sample_group_sizes(self, self.n_samples)
        
        # Génération des données
        X, y = mixture_model.sample_data(group_sizes)
        
        return X, y