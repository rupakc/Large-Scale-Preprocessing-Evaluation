from sklearn.cluster import AffinityPropagation, DBSCAN, KMeans
from sklearn.cluster import OPTICS, AgglomerativeClustering, MeanShift, Birch
from constants import model_constants


def get_clustered_data(data_matrix, clustering_algorithm=model_constants.KMEANS, distance_metric='euclidean', num_clusters=3):
    if clustering_algorithm.lower() == model_constants.AFFINITY_PROP:
        aff_prop = AffinityPropagation(affinity=distance_metric)
        aff_prop.fit(data_matrix)
        return aff_prop.labels_, aff_prop
    elif clustering_algorithm.lower() == model_constants.DBSCAN:
        dbscan = DBSCAN(metric=distance_metric)
        dbscan.fit(data_matrix)
        return dbscan.labels_, dbscan
    elif clustering_algorithm.lower() == model_constants.OPTICS:
        optics = OPTICS(metric=distance_metric)
        optics.fit(data_matrix)
        return optics.labels_, optics
    elif clustering_algorithm.lower() == model_constants.MEANSHIFT:
        mean_shift = MeanShift()
        mean_shift.fit(data_matrix)
        return mean_shift.labels_, mean_shift
    elif clustering_algorithm.lower() == model_constants.BIRCH:
        birch = Birch(n_clusters=num_clusters)
        birch.fit(data_matrix)
        return birch.labels_, birch
    elif clustering_algorithm.lower() == model_constants.AGGLOMERATIVE:
        agglomerative = AgglomerativeClustering(n_clusters=num_clusters, affinity=distance_metric)
        agglomerative.fit(data_matrix)
        return agglomerative.labels_, agglomerative
    else:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(data_matrix)
        return kmeans.labels_, kmeans
