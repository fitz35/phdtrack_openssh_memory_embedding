


import time
from typing import Tuple
from timeout_decorator import TimeoutError
from commons.data_loading.data_loading import load
import pandas as pd
from research_base.utils.results_utils import time_measure_result
from commons.feature_engineering.correlation_feature_engineering import feature_engineering_correlation_measurement
from commons.data_loading.data_types import SamplesAndLabels, split_dataset_if_needed, split_preprocessed_data_by_origin
from embedding_quality.data_balancing.data_balancing import apply_balancing
from embedding_quality.classification.ml_random_forest import ml_random_forest_pipeline
from commons.data_loading.data_cleaning import clean_all
from commons.data_loading.data_origin import DataOriginEnum
from params.common_params import INFO_COLUMNS, CommonProgramParams
from embedding_coherence.clustering.density_clustering import density_clustering_pipeline


def pipeline(params : CommonProgramParams, already_loaded_data : Tuple[SamplesAndLabels, SamplesAndLabels] | None = None):
    

    # check that params.DATA_ORIGINS_TRAINING is not empty
    if params.data_origins_training is None or len(params.data_origins_training) == 0:
        params.RESULTS_LOGGER.warning(f"No training data origins (params.DATA_ORIGINS_TRAINING: {params.data_origins_training})")
        exit(1)
    
    # check that params.DATA_ORIGINS_TRAINING and params.DATA_ORIGINS_TESTING are disjoint
    if params.data_origins_testing is not None and len(params.data_origins_testing) > 0:
        if len(params.data_origins_training.intersection(params.data_origins_testing)) > 0:
            params.RESULTS_LOGGER.warning(f"Training and testing data origins are not disjoint (params.DATA_ORIGINS_TRAINING: {params.data_origins_training}, params.DATA_ORIGINS_TESTING: {params.data_origins_testing})")
            exit(1)
    params.RESULTS_LOGGER.info(f"///---!!!! Launching testing pipeline on dataset {params.dataset_path} !!!!----///")
    params.RESULTS_LOGGER.info(f"Data origins training : {params.data_origins_training}")
    params.RESULTS_LOGGER.info(f"Data origins testing : {params.data_origins_testing}")

    start_time = time.time()
    params.RESULTS_LOGGER.info(f"Start time : {start_time}")

    if already_loaded_data is not None:
        params.RESULTS_LOGGER.info("Using already loaded data")
        origin_to_samples_and_labels = {
            DataOriginEnum.Training : already_loaded_data[0],
            DataOriginEnum.Validation : already_loaded_data[1]
        }
    else:
        # load data
        with time_measure_result(
                f'load_samples_and_labels_from_all_csv_files', 
                params.RESULTS_LOGGER
            ):
            origin_to_samples_and_labels = (
                load(
                    params.dataset_path, 
                    params.COMMON_LOGGER, 
                    params.RESULTS_LOGGER, 
                    params.data_origins_training.union(params.data_origins_testing if params.data_origins_testing is not None else set()), 
                    params.MAX_ML_WORKERS
                    )
                )
        
        # clean data
        origin_to_samples_and_labels = clean_all(params, origin_to_samples_and_labels, INFO_COLUMNS)

    # feature engineering
    with time_measure_result(
            f'feature_engineering', 
            params.RESULTS_LOGGER
        ):
        column_to_keep = feature_engineering_correlation_measurement(
            origin_to_samples_and_labels,
            params.FEATURE_CORRELATION_MATRICES_RESULTS_DIR_PATH,
            params.RESULTS_LOGGER,
            params.get_results_writer(),
            )

    # keep only the columns that are usefull
    for origin in origin_to_samples_and_labels:
        origin_to_samples_and_labels[origin].keep_columns( column_to_keep, params.RESULTS_LOGGER)

    # cut the data to training and testing
    training_samples_and_labels, maybe_testing_samples_and_labels = split_preprocessed_data_by_origin(params.data_origins_training, params.data_origins_testing, origin_to_samples_and_labels)
    training_samples_and_labels, testing_samples_and_labels = split_dataset_if_needed(training_samples_and_labels, maybe_testing_samples_and_labels)

    # rebalancing
    training_samples_and_labels = apply_balancing(params, training_samples_and_labels.sample, training_samples_and_labels.labels)
    

    # train and evaluate the model
    with time_measure_result(
            f'random forest : ', 
            params.RESULTS_LOGGER,
        ):
        try:
            ml_random_forest_pipeline(params, training_samples_and_labels.copy(), testing_samples_and_labels.copy())
        except MemoryError:
            params.RESULTS_LOGGER.error("Memory error on quality pipeline, skipping")
        except Exception as e:
            params.RESULTS_LOGGER.error(f"Error on quality pipeline: {e}")

    with time_measure_result(
            f'clustering', 
            params.RESULTS_LOGGER,
        ):
        try:
            clusters = density_clustering_pipeline(params, training_samples_and_labels.copy())
            # zipping clusters and labels
            associated_clusters_and_labels = zip_cluster_and_label(clusters, training_samples_and_labels.labels)
            # Associate cluster to labels
            params.RESULTS_LOGGER.info(f"Associating clusters to labels : \n {associated_clusters_and_labels}")
        except MemoryError:
            params.RESULTS_LOGGER.error("Memory error while clustering")
        except TimeoutError:
            params.RESULTS_LOGGER.error("Timeout error while clustering")
        except Exception as e:
            params.RESULTS_LOGGER.error(f"Error while clustering : {e}")

    

    end_time = time.time()
    params.RESULTS_LOGGER.info(f"End time : {end_time}")
    params.RESULTS_LOGGER.info(f"Total duration: {end_time - start_time}")



def zip_cluster_and_label(clusters : pd.Series, labels : pd.Series):
    """
    Zip clusters and labels into a dictionary.
    """

    def transform_dict(input_dict):
        """
        Transforms a dictionary with tuple keys into a nested dictionary.
        
        Args:
        - input_dict (dict): A dictionary with tuple keys (int, str) and int values.
        
        Returns:
        - dict: A nested dictionary where the first element of the tuple becomes the outer key
                and the second element becomes the inner key.
        """
        
        # Initialize the output dictionary
        output_dict = {}
        
        # Iterate over each key, value pair in the input dictionary
        for (num, letter), value in input_dict.items():
            
            # If the number (num) is not already a key in the output dictionary, add it
            if num not in output_dict:
                output_dict[num] = {}
            
            # Add the letter as a key inside the inner dictionary and set its value
            output_dict[num][letter] = value
            
        return output_dict

    df = pd.DataFrame({'Cluster': clusters, 'Label': labels})
    counts = df.groupby(['Cluster', 'Label']).size()
    counts_dict = transform_dict(counts.to_dict())
    return counts_dict