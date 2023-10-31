


import time
from commons.data_loading.data_loading import load
from research_base.utils.results_utils import time_measure_result
from commons.data_loading.data_types import split_dataset_if_needed, split_preprocessed_data_by_origin
from embedding_generation.embedding.word2vec import word2vec_pipeline
from params.common_params import CommonProgramParams
from embedding_generation.embedding.transformers import transformers_pipeline
from embedding_generation.embedding.deeplearning_pipeline import DeeplearningPipelines



DEEPLEARNING_PIPELINES_NAME_TO_PIPELINE = {
    #DeeplearningPipelines.Transformers : transformers_pipeline,
    DeeplearningPipelines.Word2vec : word2vec_pipeline,
    
}

def pipeline(params : CommonProgramParams):
    

    # check that params.DATA_ORIGINS_TRAINING is not empty
    if params.data_origins_training is None or len(params.data_origins_training) == 0:
        params.RESULTS_LOGGER.warning(f"No training data origins (params.DATA_ORIGINS_TRAINING: {params.data_origins_training})")
        exit(1)
    
    # check that params.DATA_ORIGINS_TRAINING and params.DATA_ORIGINS_TESTING are disjoint
    if params.data_origins_testing is not None and len(params.data_origins_testing) > 0:
        if len(params.data_origins_training.intersection(params.data_origins_testing)) > 0:
            params.RESULTS_LOGGER.warning(f"Training and testing data origins are not disjoint (params.DATA_ORIGINS_TRAINING: {params.data_origins_training}, params.DATA_ORIGINS_TESTING: {params.data_origins_testing})")
            exit(1)
    
    params.RESULTS_LOGGER.info(f"///---!!!! Launching embedding pipeline on dataset {params.dataset_path} !!!!----///")
    params.RESULTS_LOGGER.info(f"Data origins training : {params.data_origins_training}")
    params.RESULTS_LOGGER.info(f"Data origins testing : {params.data_origins_testing}")

    start_time = time.time()
    params.RESULTS_LOGGER.info(f"Pipeline start time : {start_time} seconds")

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

    # cut the data to training and testing
    training_samples_and_labels, maybe_testing_samples_and_labels = split_preprocessed_data_by_origin(params.data_origins_training, params.data_origins_testing, origin_to_samples_and_labels)
    training_samples_and_labels, testing_samples_and_labels = split_dataset_if_needed(training_samples_and_labels, maybe_testing_samples_and_labels)

    for pipeline in DEEPLEARNING_PIPELINES_NAME_TO_PIPELINE.keys():

        DEEPLEARNING_PIPELINES_NAME_TO_PIPELINE[pipeline](params, training_samples_and_labels.copy(), testing_samples_and_labels.copy())
        

    end_time = time.time()
    params.RESULTS_LOGGER.info(f"Pipeline end time : {end_time} seconds")
    params.RESULTS_LOGGER.info(f"Pipeline duration : {end_time - start_time} seconds")
    