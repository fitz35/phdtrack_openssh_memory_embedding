


import time
from commons.data_loading.data_loading import load
from research_base.utils.results_utils import time_measure_result
from commons.data_loading.data_types import split_dataset_if_needed, split_preprocessed_data_by_origin
from word2vec.data.save_embedding import gen_and_save_embedding
from word2vec.embedding.word2vec import word2vec_pipeline
from word2vec.params.params import ProgramParams

from word2vec.data.word2vec_input import transform_hex_data


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
    
    # prepare data for word2vec
    for origin in origin_to_samples_and_labels:
        df = origin_to_samples_and_labels[origin].sample
        df = transform_hex_data(params, df)

    # cut the data to training and testing
    training_samples_and_labels, maybe_testing_samples_and_labels = split_preprocessed_data_by_origin(params.data_origins_training, params.data_origins_testing, origin_to_samples_and_labels)
    training_samples_and_labels, testing_samples_and_labels = split_dataset_if_needed(training_samples_and_labels, maybe_testing_samples_and_labels)

    # train the model
    with time_measure_result(
            f'word2vec training : ', 
            params.RESULTS_LOGGER, 
            params.get_results_writer(),
            "word2vec_training_duration"
        ):
        word2vec = word2vec_pipeline(params, training_samples_and_labels)

    # generate the embedding
    with time_measure_result(
            f'word2vec used to embedde : ', 
            params.RESULTS_LOGGER, 
            params.get_results_writer(),
            "word2vec_embedding_duration"
        ):
        gen_and_save_embedding(params, training_samples_and_labels, testing_samples_and_labels, word2vec)

    end_time = time.time()
    params.set_result_for("end_time", str(end_time))
    params.set_result_for("duration", str(end_time - start_time))

    # save results
    params.get_results_writer().save_results()

    