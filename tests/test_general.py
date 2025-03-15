import repliclust as rpl

# Créer un archétype manuellement (sans API)
archetype = rpl.Archetype(
    n_clusters=3,
    dim=10,
    aspect_ref=3.0,  # Clusters oblongs
    radius_maxmin=2.0,
    max_overlap=0.001,  # Bien séparés
    min_overlap=0.0001
)

# Générer des données
X, y = archetype.synthesize()

# Visualiser les données
rpl.plot(X, y)