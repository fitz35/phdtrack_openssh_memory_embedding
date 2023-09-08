


from embedding_quality.params.params import ProgramParams
from embedding_quality.data_loading.data_loading import load
from commons.utils.results_utils import time_measure_result
from embedding_quality.feature_engineering.correlation_feature_engineering import feature_engineering_correlation_measurement


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