

from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from sklearn.utils import shuffle
import timeout_decorator
from timeout_decorator import TimeoutError
import numpy as np
import pandas as pd
from research_base.utils.results_utils import time_measure_result
from research_base.utils.data_utils import dict_to_csv_value, count_labels

from commons.data_loading.data_types import SamplesAndLabels

from embedding_coherence.data.hyperparams import CLUSTERIZATION_ALGORITHM, CLUSTERIZATION_METHOD, CLUSTERIZATION_METRIC, NB_VALUES_TO_TESTS_EPSILON
from params.common_params import CommonProgramParams

# $ python main_value_node.py -p ds_density_clustering -otr testing -b undersampling -d load_data_structure_dataset

def density_clustering_pipeline(
        params : CommonProgramParams,
        samples_and_sample_str_train: SamplesAndLabels,
):
    """
    Density clustering pipeline.
    """
    # Split data into training and test sets
    samples_train, labels_train = samples_and_sample_str_train.sample, samples_and_sample_str_train.labels
    #samples_test, _ = samples_and_sample_str_test # not working, need to split data into training if no testing data is provided

    # Track best silhouette score, best eps and the corresponding labels
    best_score = -1
    best_eps : int | None = None
    best_n_clusters : int | None = None
    best_labels : np.ndarray | None = None

    # Scale data (required for DBSCAN)
    with time_measure_result(
            f'scaling_duration', 
            params.RESULTS_LOGGER,
        ):
        # But not for cosine similarity
        #scaler = StandardScaler()
        #df_scaled = pd.DataFrame(scaler.fit_transform(samples_train), columns=samples_train.columns).astype('float32')
        df_scaled = samples_train.astype('float32')

    params.RESULTS_LOGGER.info(f"Number of samples before rebalancing and limiting rows: {dict_to_csv_value(count_labels(labels_train))}")

    # rebalance classes (economise memory and time)
    df_scaled, labels_train = balance_classes(params, df_scaled, labels_train)
    # limit the number of rows (economise memory and time)
    df_scaled, labels_train = limit_rows(params, df_scaled, labels_train)

    params.RESULTS_LOGGER.info(f"Number of samples after rebalancing and limiting rows: {dict_to_csv_value(count_labels(labels_train))}")

    # precompute cosine similarity matrix
    # reduce memory usage and save time (avoid to compute the same cosine similarity multiple times)
    # too much memory usage for large datasets : 84to
    #with time_measure_result(
    #        f'distance_matrix_computation', 
    #        params.RESULTS_LOGGER, 
    #        params.get_results_writer(),
    #        "distance_matrix_computation_time"
    #    ):
    #    distance_matrix = pairwise_distances(df_scaled, metric="cosine")

    # Define the range of eps values we want to try
    eps_values = np.linspace(0.01, 0.05, num=NB_VALUES_TO_TESTS_EPSILON)  # customize as necessary
    #eps_values = [0.6]

    # define the minimum number of samples we want in a cluster
    # We get the labels for the training set, and get half the number of samble of the labels with less samples
    # because we have 2 structure with keynodes in each files
    min_samples = int(min(count_labels(labels_train).values())/2)
    params.RESULTS_LOGGER.info(f"min_samples: {min_samples}")

    for eps in eps_values:
        # density clustering
        #dbscan = DBSCAN(
        #    eps=eps, 
        #    min_samples=50, 
        #    metric='cosine', 
        #    algorithm='ball_tree', 
        #    n_jobs=params.MAX_ML_WORKERS,
        #)  # customize min_samples as necessary
        #dbscan.fit(df_scaled)

        optics = OPTICS(
            min_samples=min_samples, 
            metric=CLUSTERIZATION_METRIC, 
            n_jobs=params.MAX_ML_WORKERS, 
            algorithm=CLUSTERIZATION_ALGORITHM, 
            cluster_method=CLUSTERIZATION_METHOD,
            xi=eps
        )

        @timeout_decorator.timeout(seconds=params.TIMEOUT_DURATION, timeout_exception=TimeoutError)
        def train_model(model :  OPTICS, input_data : pd.DataFrame, params :CommonProgramParams):
            with np.errstate(divide='ignore'): # ignore divide by zero warning
                model.fit_predict(input_data)

            return model

        with time_measure_result(
            f'clustering_duration_for_{eps}', 
            params.RESULTS_LOGGER
        ):
            optics = train_model(optics, df_scaled, params)
            

        # Get labels for training set
        labels = optics.labels_

        # Number of clusters, ignoring noise if present
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Calculate silhouette score if there's more than one cluster
        if n_clusters > 1:
            score = silhouette_score(df_scaled, labels)

            if score > best_score:
                best_score = score
                best_eps = eps
                best_n_clusters = n_clusters
                best_labels = labels
            params.RESULTS_LOGGER.info(f"eps: {eps}, number of clusters: {n_clusters}, silhouette score: {score}, noise points: {np.sum(labels == -1)}")
        else:
            params.RESULTS_LOGGER.warn(f"WARN: n_clusters <= 1 !!! eps: {eps}, number of clusters: {n_clusters}")
        
        

    # check that we found a good eps value
    if best_eps is None or best_n_clusters is None or best_labels is None:
        raise Exception("No good eps value found")

    n_noise = np.sum(best_labels == -1)
    
    params.RESULTS_LOGGER.info(f"Best eps: {best_eps}, number of clusters: {best_n_clusters}, silhouette score: {best_score}, noise points: {n_noise}")
    return pd.Series(best_labels, name='Cluster')




def balance_classes(params : CommonProgramParams, df: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Balances the classes in the given DataFrame and Series by undersampling the majority class.
    
    Parameters:
    - df: pd.DataFrame
        The input DataFrame containing the features.
    - labels: pd.Series
        The input Series containing the labels.
    - random_state: int, optional (default=42)
        Controls the randomness of the sampling and shuffling.
        
    Returns:
    - tuple[pd.DataFrame, pd.Series]
        The balanced and shuffled DataFrame and Series.
    """
    # Separate the majority and minority classes
    df_minority = df[labels != 0]
    labels_minority = labels[labels != 0]
    
    df_majority = df[labels == 0]
    labels_majority = labels[labels == 0]
    
    # Check if minority classes are actually in minority
    if len(df_minority) >= len(df_majority):
        return df, labels
    
    # Sample from the majority class to balance the classes
    df_majority_sampled = df_majority.sample(n=len(df_minority), random_state=params.RANDOM_SEED)
    labels_majority_sampled = labels_majority.loc[df_majority_sampled.index]
    
    # Concatenate the majority and minority samples
    df_sampled = pd.concat([df_minority, df_majority_sampled])
    labels_sampled = pd.concat([labels_minority, labels_majority_sampled])
    
    # Shuffle the data
    shufled_data : Tuple[pd.DataFrame, pd.Series] = shuffle(df_sampled, labels_sampled, random_state=params.RANDOM_SEED) # type: ignore
    df_sampled, labels_sampled = shufled_data
    
    return df_sampled, labels_sampled



def limit_rows(params : CommonProgramParams, df: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Limits the number of rows in the DataFrame and Series while maintaining class ratios.
    
    Parameters:
    - df: pd.DataFrame
        The input DataFrame containing the features.
    - labels: pd.Series
        The input Series containing the labels.
    - max_rows: int, optional (default=80000)
        The maximum number of rows in the returned data.
    - random_state: int, optional (default=42)
        Controls the randomness of the sampling.
        
    Returns:
    - tuple[pd.DataFrame, pd.Series]
        The DataFrame and Series with the number of rows limited to max_rows.
    """
    max_rows = params.MAX_NUMBERS_OF_SAMPLES_TO_USE_AFTER_REBALANCING
    if len(df) <= max_rows:
        return df, labels
    
    unique_labels = labels.unique()
    dfs = []
    label_series = []

    for label in unique_labels:
        df_label = df[labels == label]
        labels_label = labels[labels == label]
        num_rows = int((len(df_label) / len(df)) * max_rows)
        
        df_sampled = df_label.sample(n=num_rows, random_state=params.RANDOM_SEED)
        labels_sampled = labels_label.loc[df_sampled.index]
        
        dfs.append(df_sampled)
        label_series.append(labels_sampled)
    
    df_final = pd.concat(dfs)
    labels_final = pd.concat(label_series)
    
    return shuffle(df_final, labels_final, random_state=params.RANDOM_SEED) # type: ignore