from sklearn.cluster import AffinityPropagation

def affinity(matrix):
    affprop = AffinityPropagation(affinity="precomputed", damping=0.9)
    affprop.fit(matrix)
    return affprop.cluster_centers_indices_, affprop.labels_