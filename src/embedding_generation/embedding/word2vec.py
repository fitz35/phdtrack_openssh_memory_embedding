
import os
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

from research_base.utils.results_utils import time_measure_result

from commons.data_loading.data_types import SamplesAndLabels
from embedding_generation.params.pipelines import Pipeline
from embedding_generation.params.params import USER_DATA_COLUMN, WORD2VEC_MIN_COUNT, WORD2VEC_VECTOR_SIZE, WORD2VEC_WINDOW_BYTES_SIZE, ProgramParams
from embedding_generation.data.data_processing import split_into_chunks




def word2vec_pipeline(
        params : ProgramParams,
        samples_and_sample_str_train: SamplesAndLabels,
        samples_and_sample_str_test: SamplesAndLabels,
):
    # prepare data for word2vec
    samples_and_sample_train_samples = samples_and_sample_str_train.sample
    samples_and_sample_train_samples = __transform_hex_data(params, samples_and_sample_train_samples)

    samples_and_sample_test_samples = samples_and_sample_str_test.sample
    samples_and_sample_test_samples = __transform_hex_data(params, samples_and_sample_test_samples)

    # train the model
    with time_measure_result(
            f'word2vec training : ', 
            params.RESULTS_LOGGER, 
            params.get_results_writer(pipeline=Pipeline.Word2Vec),
            "model_training_duration"
        ):
        
        sentences = samples_and_sample_train_samples[USER_DATA_COLUMN].tolist()
        model = Word2Vec(
            sentences, 
            vector_size=WORD2VEC_VECTOR_SIZE, 
            window=int(WORD2VEC_WINDOW_BYTES_SIZE/params.WORD_BYTE_SIZE), 
            min_count=WORD2VEC_MIN_COUNT, 
            workers=params.MAX_ML_WORKERS
        )

    
    # generate the embedding
    with time_measure_result(
            f'word2vec used to embedde : ', 
            params.RESULTS_LOGGER, 
            params.get_results_writer(Pipeline.Word2Vec),
            "gen_embedding_duration"
        ):
        train_embedded = __gen_embedding(samples_and_sample_str_train, model)
        test_embedded = __gen_embedding(samples_and_sample_str_test, model)

    train_embedded.save_to_csv(os.path.join(params.OUTPUT_FOLDER, f"training_word2vec_embedding.csv"))
    test_embedded.save_to_csv(os.path.join(params.OUTPUT_FOLDER, f"validation_word2vec_embedding.csv"))


def __transform_hex_data(params : ProgramParams, df: pd.DataFrame) -> pd.DataFrame:
    """Transforms the specified column of a DataFrame into lists of 2-byte length strings."""
    df[USER_DATA_COLUMN] = df[USER_DATA_COLUMN].apply(lambda x: split_into_chunks(x, params.WORD_BYTE_SIZE))
    return df

def __gen_embedding(
    samples_and_sample_str: SamplesAndLabels,
    model : Word2Vec,
):
    def get_average_embedding(word_sequence : list[str]) :
        if word_sequence:
            return np.mean([model.wv[word] for word in word_sequence], axis=0)
        return np.zeros(model.vector_size)
    
    samples = samples_and_sample_str.sample
    labels = samples_and_sample_str.labels


    # Directly create the embeddings DataFrame
    embeddings_list = samples[USER_DATA_COLUMN].tolist()
    embeddings_list = [get_average_embedding(word_sequence) for word_sequence in embeddings_list]
    embeddings_df = pd.DataFrame(embeddings_list, columns=[f"feature_{i}" for i in range(model.vector_size)])

    assert len(embeddings_df) == len(labels), "The DataFrame and Series do not have the same number of rows!"

    return SamplesAndLabels(embeddings_df, labels)

