import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import repliclust as rpl

# Tester avec différentes dimensions
dimensions = [2, 10, 50, 100, 200]
silhouette_scores = []
ari_scores = []
nmi_scores = []

for dim in dimensions:
    # Créer un archétype avec une dimension spécifique
    archetype = rpl.Archetype(
        n_clusters=5,
        dim=dim,
        aspect_ref=2.0,
        aspect_maxmin=3.0,
        radius_maxmin=2.5,
        max_overlap=0.05,
        min_overlap=0.01
    )
    
    # Générer des données
    X, y = archetype.synthesize()
    
    # Appliquer un algorithme de clustering
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    y_pred = kmeans.labels_
    
    # Calculer le Silhouette Score
    sil_score = silhouette_score(X, y_pred)
    silhouette_scores.append(sil_score)
    
    # Calculer l'Adjusted Rand Index (ARI)
    ari = adjusted_rand_score(y, y_pred)
    ari_scores.append(ari)
    
    # Calculer le Normalized Mutual Information (NMI)
    nmi = normalized_mutual_info_score(y, y_pred)
    nmi_scores.append(nmi)
    
    print(f"Dimension {dim}: Silhouette={sil_score:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}")

# Visualiser l'impact de la dimension sur les performances
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(dimensions, silhouette_scores, marker='o', color='blue')
plt.ylabel('Silhouette Score')
plt.title('Impact de la dimension sur les performances de clustering')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(dimensions, ari_scores, marker='s', color='green')
plt.ylabel('ARI')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(dimensions, nmi_scores, marker='^', color='red')
plt.xlabel('Dimension')
plt.ylabel('NMI')
plt.grid(True)

plt.tight_layout()
plt.show()