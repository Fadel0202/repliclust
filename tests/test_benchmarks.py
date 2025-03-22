import numpy as np
import time
import matplotlib.pyplot as plt
import os
import repliclust as rpl
from repliclust.maxmin.archetype import MaxMinArchetype  # Import direct de la classe
from repliclust.optimized import ArchetypeOptimized
from tqdm import tqdm
import psutil
import warnings
import seaborn as sns

# Ignorer l'erreur d'initialisation de l'API OpenAI
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

def benchmark_clustering(n_clusters, dimensions, n_samples_list, n_runs=3):
    """Test comparatif des performances entre versions originale et optimisée"""
    results = {
        'original': {'times': [], 'memory': [], 'configs': []},
        'optimized': {'times': [], 'memory': [], 'configs': []},
        'speedup': [],
        'memory_ratio': []
    }
    
    total_configs = len(n_clusters) * len(dimensions) * len(n_samples_list)
    with tqdm(total=total_configs, desc="Configurations testées") as pbar:
        for n_clust in n_clusters:
            for dim in dimensions:
                for n_samples in n_samples_list:
                    config = f"C={n_clust}, D={dim}, N={n_samples}"
                    print(f"\nTesting configuration: {config}")
                    
                    # Test version originale
                    times_orig = []
                    memory_orig = []
                    for run in tqdm(range(n_runs), desc="Original"):
                        process = psutil.Process(os.getpid())
                        start_mem = process.memory_info().rss / 1024 / 1024
                        start = time.time()
                        
                        try:
                            # Utiliser directement MaxMinArchetype (importé plus haut)
                            archetype = MaxMinArchetype(
                                n_clusters=n_clust,
                                dim=dim,
                                n_samples=n_samples,
                                aspect_ref=3.0,
                                radius_maxmin=2.0,
                                max_overlap=0.001,
                                min_overlap=0.0001
                            )
                            X, y = archetype.synthesize(quiet=True)
                            
                            times_orig.append(time.time() - start)
                            memory_orig.append(process.memory_info().rss / 1024 / 1024 - start_mem)
                        except Exception as e:
                            print(f"Erreur dans la version originale: {str(e)}")
                            # Ne pas ajouter cette exécution aux statistiques
                    
                    # Test version optimisée
                    times_opt = []
                    memory_opt = []
                    for run in tqdm(range(n_runs), desc="Optimized"):
                        process = psutil.Process(os.getpid())
                        start_mem = process.memory_info().rss / 1024 / 1024
                        start = time.time()
                        
                        try:
                            archetype = ArchetypeOptimized(
                                n_clusters=n_clust,
                                dim=dim,
                                n_samples=n_samples,
                                aspect_ref=3.0,
                                radius_maxmin=2.0,
                                max_overlap=0.001,
                                min_overlap=0.0001,
                                max_epoch=5  # Moins d'époques pour accélérer
                            )
                            X, y = archetype.synthesize(quiet=True)
                            
                            times_opt.append(time.time() - start)
                            memory_opt.append(process.memory_info().rss / 1024 / 1024 - start_mem)
                        except Exception as e:
                            print(f"Erreur dans la version optimisée: {str(e)}")
                            # Ne pas ajouter cette exécution aux statistiques
                    
                    # Vérifier s'il y a des données valides
                    if len(times_orig) == 0 or len(times_opt) == 0:
                        print(f"Pas assez de données valides pour {config}, ignoré.")
                        pbar.update(1)
                        continue
                        
                    # Calcul des statistiques
                    time_orig_mean = np.mean(times_orig)
                    time_orig_std = np.std(times_orig)
                    memory_orig_mean = np.mean(memory_orig)
                    
                    time_opt_mean = np.mean(times_opt)
                    time_opt_std = np.std(times_opt)
                    memory_opt_mean = np.mean(memory_opt)
                    
                    # Stockage des résultats
                    results['original']['times'].append((time_orig_mean, time_orig_std))
                    results['original']['memory'].append(memory_orig_mean)
                    results['original']['configs'].append(config)
                    
                    results['optimized']['times'].append((time_opt_mean, time_opt_std))
                    results['optimized']['memory'].append(memory_opt_mean)
                    results['optimized']['configs'].append(config)
                    
                    # Protection contre la division par zéro
                    if time_opt_mean > 0:
                        results['speedup'].append(time_orig_mean / time_opt_mean)
                    else:
                        results['speedup'].append(1.0)
                        
                    if memory_opt_mean > 0:
                        results['memory_ratio'].append(memory_orig_mean / memory_opt_mean)
                    else:
                        results['memory_ratio'].append(1.0)
                    
                    print(f"\nRésultats pour {config}:")
                    print(f"Original  : {time_orig_mean:.3f}s ± {time_orig_std:.3f}s ({memory_orig_mean:.1f} MB)")
                    print(f"Optimisé  : {time_opt_mean:.3f}s ± {time_opt_std:.3f}s ({memory_opt_mean:.1f} MB)")
                    print(f"Speedup   : {results['speedup'][-1]:.2f}x")
                    print(f"Ratio mém.: {results['memory_ratio'][-1]:.2f}x")
                    
                    pbar.update(1)
    
    return results

def plot_detailed_results(results, filename_prefix="benchmark"):
    """Génération de graphiques détaillés avec seaborn"""
    if not results['original']['configs']:
        print("Aucun résultat valide à visualiser.")
        return
        
    try:
        sns.set_style("whitegrid")
        
        # Figure pour les temps d'exécution et la mémoire
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        x = np.arange(len(results['original']['configs']))
        width = 0.35
        
        # Temps d'exécution
        ax1.bar(x - width/2, 
                [t[0] for t in results['original']['times']], 
                width,
                yerr=[t[1] for t in results['original']['times']],
                label='Original', 
                capsize=5)
        ax1.bar(x + width/2,
                [t[0] for t in results['optimized']['times']],
                width, 
                yerr=[t[1] for t in results['optimized']['times']],
                label='Optimisé',
                capsize=5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(results['original']['configs'], rotation=45, ha='right')
        ax1.set_ylabel('Temps (s)')
        ax1.set_title('Temps d\'exécution par configuration')
        ax1.legend()
        
        # Utilisation mémoire
        ax2.plot(results['original']['memory'], 'o-', label='Original')
        ax2.plot(results['optimized']['memory'], 's-', label='Optimisé')
        ax2.set_xticks(x)
        ax2.set_xticklabels(results['original']['configs'], rotation=45, ha='right')
        ax2.set_ylabel('Mémoire (MB)')
        ax2.set_title('Utilisation mémoire')
        ax2.legend()
        
        # Speedup
        ax3.plot(results['speedup'], 'o-')
        ax3.axhline(y=1.0, color='r', linestyle='--')
        ax3.set_xticks(x)
        ax3.set_xticklabels(results['original']['configs'], rotation=45, ha='right')
        ax3.set_ylabel('Accélération')
        ax3.set_title('Facteur d\'accélération')
        
        # Distribution des speedups
        sns.boxplot(data=results['speedup'], ax=ax4)
        ax4.set_ylabel('Distribution des accélérations')
        ax4.set_title('Distribution des gains de performance')
        
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_detailed.png", dpi=300, bbox_inches='tight')
        
        # Sauvegarder aussi les versions séparées
        for ax, name in zip([ax1, ax2, ax3, ax4], 
                        ['temps', 'memoire', 'speedup', 'distribution']):
            fig_single = plt.figure(figsize=(8, 6))
            ax_new = fig_single.add_subplot(111)
            ax_new.set_position(ax.get_position())
            fig_single = ax.get_figure()
            fig_single.savefig(f"{filename_prefix}_{name}.png", dpi=300, bbox_inches='tight')
            plt.close(fig_single)
    except Exception as e:
        print(f"Erreur lors de la génération des graphiques: {str(e)}")

def save_results(results, filename):
    """Sauvegarde détaillée des résultats"""
    if not results['original']['configs']:
        print("Aucun résultat valide à sauvegarder.")
        return
        
    try:
        with open(filename, 'w') as f:
            f.write("Résultats détaillés des benchmarks\n")
            f.write("=" * 50 + "\n\n")
            
            for i, config in enumerate(results['original']['configs']):
                f.write(f"\nConfiguration: {config}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Original: {results['original']['times'][i][0]:.3f}s ± "
                    f"{results['original']['times'][i][1]:.3f}s\n")
                f.write(f"Mémoire: {results['original']['memory'][i]:.1f} MB\n")
                f.write(f"Optimisé: {results['optimized']['times'][i][0]:.3f}s ± "
                    f"{results['optimized']['times'][i][1]:.3f}s\n")
                f.write(f"Mémoire: {results['optimized']['memory'][i]:.1f} MB\n")
                f.write(f"Accélération: {results['speedup'][i]:.2f}x\n")
                f.write(f"Ratio mémoire: {results['memory_ratio'][i]:.2f}x\n")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des résultats: {str(e)}")

def run_benchmarks(n_runs=2):
    """Exécution complète des benchmarks"""
    # Configuration très simplifiée pour éviter les erreurs
    n_clusters = [2, 3]  # Très peu de clusters
    dimensions = [2, 3]  # Très petites dimensions
    n_samples_list = [100, 200]  # Très peu d'échantillons
    
    print("Démarrage des benchmarks...")
    try:
        results = benchmark_clustering(n_clusters, dimensions, n_samples_list, n_runs)
        
        # Vérifier si nous avons des résultats valides
        if not results['original']['configs']:
            print("Aucun benchmark n'a pu être exécuté avec succès. Abandon.")
            return
            
        # Génération des visualisations et rapports
        plot_detailed_results(results)
        save_results(results, "benchmark_results.txt")
        
        # Affichage du résumé final
        print("\nRésumé final des performances:")
        print("=" * 50)
        
        # Statistiques version originale
        orig_times = [t[0] for t in results['original']['times']]
        print("Version originale:")
        print(f"- Temps moyen : {np.mean(orig_times):.3f}s ± {np.std(orig_times):.3f}s")
        print(f"- Mémoire moy.: {np.mean(results['original']['memory']):.1f} MB")
        
        # Statistiques version optimisée
        opt_times = [t[0] for t in results['optimized']['times']]
        print("\nVersion optimisée:")
        print(f"- Temps moyen : {np.mean(opt_times):.3f}s ± {np.std(opt_times):.3f}s")
        print(f"- Mémoire moy.: {np.mean(results['optimized']['memory']):.1f} MB")
        
        # Gains de performance
        print("\nGains de performance:")
        print(f"- Accélération moyenne : {np.mean(results['speedup']):.2f}x ± {np.std(results['speedup']):.2f}x")
        print(f"- Ratio mémoire moyen : {np.mean(results['memory_ratio']):.2f}x ± {np.std(results['memory_ratio']):.2f}x")
    except Exception as e:
        print(f"Erreur lors de l'exécution des benchmarks: {str(e)}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    run_benchmarks(n_runs=2)  # Réduire encore plus le nombre d'exécutions