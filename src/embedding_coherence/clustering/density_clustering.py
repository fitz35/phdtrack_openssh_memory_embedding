

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

from commons.data_loading.data_types import SamplesAndLabels

# $ python main_value_node.py -p ds_density_clustering -otr testing -b undersampling -d load_data_structure_dataset

def density_clustering_pipeline(
        samples_and_sample_str_train: SamplesAndLabels,
) -> None:
    """
    Density clustering pipeline.
    """
    # Split data into training and test sets
    samples_train, _ = samples_and_sample_str_train.sample, samples_and_sample_str_train.labels
    #samples_test, _ = samples_and_sample_str_test # not working, need to split data into training if no testing data is provided

    # Track best silhouette score, best eps and the corresponding labels
    best_score = -1
    best_eps = None
    best_n_clusters = None
    best_labels = None

    # Define the range of eps values we want to try
    eps_values = np.linspace(0.1, 5, num=2)  # customize as necessary

    for eps in eps_values:
        # density clustering
        dbscan = DBSCAN(eps=eps, min_samples=5)  # customize min_samples as necessary
        dbscan.fit(samples_train)

        # Get labels for training set
        labels = dbscan.labels_

        # Number of clusters, ignoring noise if present
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Calculate silhouette score if there's more than one cluster
        if n_clusters > 1:
            score = silhouette_score(samples_train, labels)
            print(f"eps: {eps}, number of clusters: {n_clusters}, silhouette score: {score}")

            if score > best_score:
                best_score = score
                best_eps = eps
                best_n_clusters = n_clusters
                best_labels = labels
        else:
            print(f"WARN: n_clusters <= 1 !!! eps: {eps}, number of clusters: {n_clusters}")

    # check that we found a good eps value
    if best_eps is None:
        raise Exception("No good eps value found")

    n_noise = np.sum(best_labels == -1)
    print(f"Best eps: {best_eps}, number of clusters: {best_n_clusters}, silhouette score: {best_score}, noise points: {n_noise}")

