


from embedding_quality.params.params import ProgramParams
from embedding_quality.data_loading.data_loading import load
from commons.utils.results_utils import time_measure_result
from embedding_quality.feature_engineering.correlation_feature_engineering import feature_engineering_correlation_measurement
from embedding_quality.data_loading.data_types import split_dataset_if_needed, split_preprocessed_data_by_origin
from embedding_quality.data_balancing.data_balancing import apply_balancing
from embedding_quality.classification.ml_random_forest import ml_random_forest_pipeline


def pipeline(params : ProgramParams):
    # check that params.DATA_ORIGINS_TRAINING is not empty
    if params.data_origins_training is None or len(params.data_origins_training) == 0:
        params.COMMON_LOGGER.warning(f"No training data origins (params.DATA_ORIGINS_TRAINING: {params.data_origins_training})")
        exit(1)
    
    # check that params.DATA_ORIGINS_TRAINING and params.DATA_ORIGINS_TESTING are disjoint
    if params.data_origins_testing is not None and len(params.data_origins_testing) > 0:
        if len(params.data_origins_training.intersection(params.data_origins_testing)) > 0:
            params.COMMON_LOGGER.warning(f"Training and testing data origins are not disjoint (params.DATA_ORIGINS_TRAINING: {params.data_origins_training}, params.DATA_ORIGINS_TESTING: {params.data_origins_testing})")
            exit(1)



    # load data
    with time_measure_result(
            f'load_samples_and_labels_from_all_csv_files', 
            params.RESULTS_LOGGER, 
            params.results_writer,
            "data_loading_duration"
        ):
        origin_to_samples_and_labels = load(params, params.data_origins_training.union(params.data_origins_testing if params.data_origins_testing is not None else set()))

    # feature engineering
    with time_measure_result(
            f'feature_engineering', 
            params.RESULTS_LOGGER, 
            params.results_writer,
            "feature_engineering_duration"
        ):
        column_to_keep = feature_engineering_correlation_measurement(params, origin_to_samples_and_labels)

    # keep only the columns that are usefull
    for origin in origin_to_samples_and_labels:
        origin_to_samples_and_labels[origin].keep_columns(params, column_to_keep)

    # cut the data to training and testing
    training_samples_and_labels, maybe_testing_samples_and_labels = split_preprocessed_data_by_origin(params, origin_to_samples_and_labels)
    training_samples_and_labels, testing_samples_and_labels = split_dataset_if_needed(training_samples_and_labels, maybe_testing_samples_and_labels)

    # rebalancing
    training_samples_and_labels = apply_balancing(params, training_samples_and_labels.sample, training_samples_and_labels.labels)

    # train and evaluate the model
    with time_measure_result(
            f'random forest : ', 
            params.RESULTS_LOGGER, 
            params.results_writer,
            "classification_duration"
        ):
        ml_random_forest_pipeline(params, training_samples_and_labels, testing_samples_and_labels)

    # save results
    params.results_writer.save_results()

    