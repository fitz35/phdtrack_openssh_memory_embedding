


import time
from embedding_quality.params.params import ProgramParams
from commons.data_loading.data_loading import load
from research_base.utils.results_utils import time_measure_result
from commons.feature_engineering.correlation_feature_engineering import feature_engineering_correlation_measurement
from commons.data_loading.data_types import split_dataset_if_needed, split_preprocessed_data_by_origin
from embedding_quality.data_balancing.data_balancing import apply_balancing
from embedding_quality.classification.ml_random_forest import ml_random_forest_pipeline
from embedding_quality.data.data_cleaning import clean


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

    start_time = time.time()
    params.set_result_for("start_time", str(start_time))

    # load data
    with time_measure_result(
            f'load_samples_and_labels_from_all_csv_files', 
            params.RESULTS_LOGGER, 
            params.get_results_writer(),
            "data_loading_duration"
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
    for origin in origin_to_samples_and_labels:
        origin_to_samples_and_labels[origin] = clean(params, origin_to_samples_and_labels[origin])


    end_time = time.time()
    params.set_result_for("end_time", str(end_time))
    params.set_result_for("duration", str(end_time - start_time))

    # save results
    params.get_results_writer().save_results()

    