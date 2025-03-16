import numpy as np
import time
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os
from functools import lru_cache
import repliclust as rpl

def test_original_vs_optimized():
    """Test comparatif des performances"""
    # Configuration
    n_clusters = [3, 5, 10]
    dimensions = [2, 10, 50]
    n_samples = 1000
    n_runs = 3
    
    results = {
        'original': [],
        'optimized': [],
        'speedup': []
    }
    
    # Tests avec la version originale
    print("Testing original version...")
    for n_clust in n_clusters:
        for dim in dimensions:
            start = time.time()
            for _ in range(n_runs):
                archetype = rpl.Archetype(
                    n_clusters=n_clust,
                    dim=dim,
                    n_samples=n_samples,
                    aspect_ref=3.0,
                    radius_maxmin=2.0,
                    max_overlap=0.001,
                    min_overlap=0.0001
                )
                X, y = archetype.synthesize()
            time_orig = (time.time() - start) / n_runs
            results['original'].append(time_orig)
            
    # Sauvegarde des résultats
    plot_results(results, "benchmark_results.png")
    print_summary(results)

def plot_results(results, filename):
    """Génère les graphiques de performance"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['original'], label='Original')
    if 'optimized' in results and results['optimized']:
        plt.plot(results['optimized'], label='Optimisé')
    plt.xlabel('Configuration (n_clusters, dim)')
    plt.ylabel('Temps (s)')
    plt.title('Temps d\'exécution')
    plt.legend()
    plt.grid(True)
    
    if 'speedup' in results and results['speedup']:
        plt.subplot(1, 2, 2)
        plt.plot(results['speedup'])
        plt.xlabel('Configuration')
        plt.ylabel('Accélération')
        plt.title('Facteur d\'accélération')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    
def print_summary(results):
    """Affiche le résumé des performances"""
    print("\nRésumé des performances:")
    print("-" * 40)
    print(f"Temps moyen version originale: {np.mean(results['original']):.3f}s")
    if 'optimized' in results and results['optimized']:
        print(f"Temps moyen version optimisée: {np.mean(results['optimized']):.3f}s")
        print(f"Accélération moyenne: {np.mean(results['speedup']):.2f}x")

if __name__ == "__main__":
    test_original_vs_optimized()