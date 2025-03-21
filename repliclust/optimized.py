import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
from functools import lru_cache
from .base import Archetype
from .maxmin.archetype import MaxMinArchetype
from .overlap._gradients import compute_quantiles_cached, update_centers

class ArchetypeOptimized(Archetype):
    """Version optimisée de la classe Archetype"""
    
    def __init__(self, n_clusters=5, dim=2, n_samples=1000,
                 aspect_ref=3.0, radius_maxmin=2.0,
                 max_overlap=0.001, min_overlap=0.0001,
                 use_cache=True, n_workers=None):
        super().__init__(n_clusters=n_clusters, 
                        dim=dim,
                        n_samples=n_samples,
                        aspect_ref=aspect_ref,
                        radius_maxmin=radius_maxmin,
                        max_overlap=max_overlap,
                        min_overlap=min_overlap)
                        
        self.use_cache = use_cache
        self.n_workers = n_workers or min(os.cpu_count(), self.n_clusters)
        
        # Initialisation des matrices de covariance et des centres
        self._initialize_covariance_matrices()
        self._initialize_centers()
    
    def _initialize_covariance_matrices(self):
        """Initialise les matrices de covariance"""
        if not hasattr(self, 'cov_list') or self.cov_list is None:
            # Création des matrices de covariance par défaut (identité)
            self.cov_list = [np.eye(self.dim) for _ in range(self.n_clusters)]
            
        # Vérification et conversion des matrices existantes
        for i, cov in enumerate(self.cov_list):
            if not isinstance(cov, np.ndarray):
                self.cov_list[i] = np.array(cov)
            if self.cov_list[i].ndim != 2:
                dim = int(np.sqrt(len(self.cov_list[i])))
                self.cov_list[i] = self.cov_list[i].reshape(dim, dim)
            if self.cov_list[i].shape != (self.dim, self.dim):
                raise ValueError(f"Dimension incorrecte pour la matrice de covariance {i}")
    
    def _initialize_centers(self):
        """Initialise les centres des clusters"""
        if not hasattr(self, 'initial_centers') or self.initial_centers is None:
            # Génération aléatoire des centres initiaux
            self.initial_centers = np.random.randn(self.n_clusters, self.dim)
            # Mise à l'échelle des centres pour une meilleure séparation initiale
            self.initial_centers *= np.sqrt(self.dim)
    
    @lru_cache(maxsize=128)
    def _compute_quantiles_cached(self, ref_idx, centers_tuple):
        """Version optimisée avec cache du calcul des quantiles"""
        # Vérification et préparation des centres
        centers = np.array(centers_tuple)
        if centers.ndim != 2:
            raise ValueError(f"centers doit être 2D, trouvé {centers.ndim}D")
            
        other_idx = [i for i in range(self.n_clusters) if i != ref_idx]
        
        # Conversion des matrices de covariance
        cov_list_tuple = []
        for cov in self.cov_list:
            # Vérification de la dimensionnalité
            if cov.ndim != 2:
                raise ValueError(f"Matrice de covariance invalide: dimension {cov.ndim}")
            # Conversion en tuple
            cov_list_tuple.append(tuple(map(tuple, cov)))
        
        try:
            return compute_quantiles_cached(
                ref_idx,
                tuple(other_idx),
                centers_tuple,
                tuple(cov_list_tuple),
                mode='lda'
            )
        except Exception as e:
            print(f"Erreur dans le calcul des quantiles: {str(e)}")
            # Fallback sur le mode centre-à-centre en cas d'erreur
            return compute_quantiles_cached(
                ref_idx,
                tuple(other_idx),
                centers_tuple,
                tuple(cov_list_tuple),
                mode='c2c'
            )
    
    def _optimize_centers_parallel(self, initial_centers):
        """Optimisation parallèle des centres"""
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for i in range(self.n_clusters):
                futures.append(
                    executor.submit(
                        self._optimize_single_center,
                        i, initial_centers
                    )
                )
            
            centers = np.empty_like(initial_centers)
            for i, future in enumerate(futures):
                centers[i] = future.result()
                
        return centers
    
    def _optimize_single_center(self, idx, centers):
        """Optimisation d'un centre unique"""
        centers = centers.copy()
        
        # Vérification des dimensions
        if centers.ndim != 2:
            raise ValueError(f"centers doit être 2D, trouvé {centers.ndim}D")
        
        # Conversion en tuple pour le cache
        centers_tuple = tuple(map(tuple, centers))
        
        try:
            quantiles = self._compute_quantiles_cached(idx, centers_tuple)
            
            # Mise à jour du centre avec les quantiles calculées 
            update_centers(
                ref_cluster_idx=idx,
                centers=centers,
                cov_list=self.cov_list,
                quantiles=quantiles,
                axis_deriv_t_list=None,  # Ajout des arguments manquants
                linear_penalty_weight=0.1,
                quantile_bounds={'min': self.min_overlap, 'max': self.max_overlap},
                learning_rate=0.1,
                mode='lda'
            )
            
        except Exception as e:
            print(f"Erreur lors de l'optimisation du centre {idx}: {str(e)}")
            return centers[idx]
        
        return centers[idx]

    def _compute_cluster_sizes(self, n_samples):
        """Calcul de la taille des clusters"""
        # Distribution uniforme par défaut
        base_size = n_samples // self.n_clusters
        sizes = np.full(self.n_clusters, base_size)
        
        # Ajustement pour atteindre exactement n_samples
        remainder = n_samples - (base_size * self.n_clusters)
        if remainder > 0:
            sizes[:remainder] += 1
            
        return sizes

    def synthesize(self, n_samples=None):
        """Génération optimisée des données"""
        if n_samples is not None:
            self.n_samples = n_samples
            
        # Vérification des dimensions et initialisation si nécessaire
        self._check_dimensions()
        self._check_centers()
        
        # Optimisation parallèle des centres
        centers = self._optimize_centers_parallel(self.initial_centers)
        
        # Pré-allocation des tableaux
        X = np.empty((self.n_samples, self.dim))
        y = np.zeros(self.n_samples, dtype=int)
        
        # Génération vectorisée des échantillons
        chunk_size = self.n_samples // self.n_workers
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            
            for i in range(self.n_workers):
                start = i * chunk_size
                end = start + chunk_size if i < self.n_workers - 1 else self.n_samples
                futures.append(
                    executor.submit(
                        self._generate_samples_chunk,
                        start, end, centers
                    )
                )
            
            for i, future in enumerate(futures):
                start = i * chunk_size
                end = start + chunk_size if i < self.n_workers - 1 else self.n_samples
                X[start:end], y[start:end] = future.result()
        
        return X, y
    
    def _generate_samples_chunk(self, start, end, centers):
        """Génération vectorisée d'un sous-ensemble d'échantillons"""
        n_samples = end - start
        X = np.empty((n_samples, self.dim))
        y = np.empty(n_samples, dtype=int)
        
        # Distribution des échantillons entre les clusters
        cluster_sizes = self._compute_cluster_sizes(n_samples)
        current_idx = 0
        
        for cluster_idx, size in enumerate(cluster_sizes):
            if size > 0:  # Vérification pour éviter les erreurs
                X[current_idx:current_idx + size] = (
                    np.random.multivariate_normal(
                        centers[cluster_idx],
                        self.cov_list[cluster_idx],
                        size
                    )
                )
                y[current_idx:current_idx + size] = cluster_idx
                current_idx += size
            
        return X, y
    
    def _check_dimensions(self):
        """Vérifie la cohérence des dimensions"""
        if not hasattr(self, 'cov_list') or not self.cov_list:
            self._initialize_covariance_matrices()
        
        for i, cov in enumerate(self.cov_list):
            if not isinstance(cov, np.ndarray):
                self.cov_list[i] = np.array(cov)
            if self.cov_list[i].ndim != 2:
                dim = int(np.sqrt(len(self.cov_list[i])))
                self.cov_list[i] = self.cov_list[i].reshape(dim, dim)
            if self.cov_list[i].shape != (self.dim, self.dim):
                raise ValueError(f"Dimension incorrecte pour la matrice de covariance {i}")
    
    def _check_centers(self):
        """Vérifie l'existence et la validité des centres"""
        if not hasattr(self, 'initial_centers') or self.initial_centers is None:
            self._initialize_centers()
        
        if not isinstance(self.initial_centers, np.ndarray):
            self.initial_centers = np.array(self.initial_centers)
        
        if self.initial_centers.ndim != 2:
            raise ValueError(f"initial_centers doit être 2D, trouvé {self.initial_centers.ndim}D")
        
        if self.initial_centers.shape != (self.n_clusters, self.dim):
            raise ValueError(f"Dimensions incorrectes pour initial_centers: attendu {(self.n_clusters, self.dim)}, "
                            f"trouvé {self.initial_centers.shape}")