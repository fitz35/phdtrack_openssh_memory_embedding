
import os
from gensim.models import Word2Vec
import numpy as np
import timeout_decorator
from timeout_decorator import TimeoutError
import pandas as pd

from research_base.utils.results_utils import time_measure_result


from commons.data_loading.data_types import SamplesAndLabels
from embedding_generation.data.data_processing import split_into_chunks
from embedding_generation.data.hyperparams_word2vec import Word2vecHyperparams, get_word2vec_hyperparams_instances
from params.common_params import USER_DATA_COLUMN, CommonProgramParams
from testing_pipelines.pipeline import pipeline as testing_pipeline




def word2vec_pipeline(
        params : CommonProgramParams,
        samples_and_sample_str_train: SamplesAndLabels,
        samples_and_sample_str_test: SamplesAndLabels,
):
    # prepare data for word2vec
    samples_and_sample_train_samples = samples_and_sample_str_train.sample
    

    samples_and_sample_test_samples = samples_and_sample_str_test.sample
    

    instances: list[Word2vecHyperparams] = get_word2vec_hyperparams_instances()

    folder = params.OUTPUT_FOLDER
    os.makedirs(folder, exist_ok=True)

    for instance in instances:
        params.RESULTS_LOGGER.info(f"!!!!!!!!!!!!! Word2vec instance : {instance.to_dir_name()} !!!!!!!!!!!!!")
        instance_folder_name = f"embedding_word2vec_{instance.to_dir_name()}"
        instance_folder = os.path.join(folder, instance_folder_name)
        if os.path.exists(instance_folder):
            params.RESULTS_LOGGER.info(f"Word2Vec instance {instance} already computed")
            continue

        params.RESULTS_LOGGER.info(f"Word2Vec instance : {instance}")

        samples_and_sample_train_samples_new = __transform_hex_data(instance, samples_and_sample_train_samples)
        max_length_train = samples_and_sample_train_samples_new[USER_DATA_COLUMN].apply(len).max()
        samples_and_sample_test_samples_new = __transform_hex_data(instance, samples_and_sample_test_samples)
        max_length_test = samples_and_sample_test_samples_new[USER_DATA_COLUMN].apply(len).max()

        params.RESULTS_LOGGER.info(f"max_length_train : {max(max_length_train, max_length_test)}")

        @timeout_decorator.timeout(seconds=params.TIMEOUT_DURATION, timeout_exception=TimeoutError)
        def train_model(input_data : pd.DataFrame, params :CommonProgramParams):
            sentences = input_data[USER_DATA_COLUMN].tolist()
            model = Word2Vec(
                sentences, 
                vector_size=instance.output_size, 
                window=int(instance.window_character_size/instance.word_character_size), 
                min_count=instance.min_count, 
                workers=params.MAX_ML_WORKERS,
                seed=params.RANDOM_SEED
            )
            return model

        try:
            # train the model
            with time_measure_result(
                    f'word2vec training : ', 
                    params.RESULTS_LOGGER
                ):
                model = train_model(samples_and_sample_train_samples_new, params)

            
            # generate the embedding
            with time_measure_result(
                    f'word2vec used to embedde : ', 
                    params.RESULTS_LOGGER,
                ):
                train_embedded = __gen_embedding(
                    SamplesAndLabels(
                        samples_and_sample_train_samples_new, 
                        samples_and_sample_str_train.labels
                        ), 
                    model
                )
                test_embedded = __gen_embedding(
                    SamplesAndLabels(
                        samples_and_sample_test_samples_new, 
                        samples_and_sample_str_test.labels
                        ),
                    model
                )

            os.makedirs(instance_folder, exist_ok=True)

            # test the model
            testing_pipeline(params, (train_embedded, test_embedded))
        except TimeoutError:
            params.RESULTS_LOGGER.error(f"Timeout error in transformers pipeline {instance.index}, skipping")
        except MemoryError:
            params.RESULTS_LOGGER.error(f"Memory error in pipeline {instance.index}, skipping")
        except Exception as e:
            params.RESULTS_LOGGER.error(f"Exception in pipeline {instance.index}, skipping: {e}")


def __transform_hex_data(params: Word2vecHyperparams, df: pd.DataFrame) -> pd.DataFrame:
    """Transforms the specified column of a DataFrame into lists of 2-byte length strings."""
    transformed_df = df.copy()
    transformed_df[USER_DATA_COLUMN] = transformed_df[USER_DATA_COLUMN].apply(lambda x: split_into_chunks(x, params.word_character_size))
    return transformed_df

def __gen_embedding(
    samples_and_sample_str: SamplesAndLabels,
    model : Word2Vec,
):
    
    def get_average_embedding(word_sequence):
        """
        Get the average embedding of a sequence of words. (SKIP oov = out-of-vocabulary words)
        """
        embeddings = [model.wv[word] for word in word_sequence if word in model.wv]
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(model.vector_size)
    
    samples = samples_and_sample_str.sample
    labels = samples_and_sample_str.labels


    # Directly create the embeddings DataFrame
    embeddings_list = samples[USER_DATA_COLUMN].tolist()
    embeddings_list = [get_average_embedding(word_sequence) for word_sequence in embeddings_list]
    embeddings_df = pd.DataFrame(embeddings_list, columns=[f"feature_{i}" for i in range(model.vector_size)])

    assert len(embeddings_df) == len(labels), "The DataFrame and Series do not have the same number of rows!"

    return SamplesAndLabels(embeddings_df, labels)

