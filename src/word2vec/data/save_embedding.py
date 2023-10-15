import os
from gensim.models import Word2Vec
from commons.data_loading.data_types import SamplesAndLabels
import numpy as np
import pandas as pd

from word2vec.params.params import USER_DATA_COLUMN, ProgramParams


def gen_and_save_embedding(
        params : ProgramParams,
        samples_and_sample_str_train: SamplesAndLabels,
        samples_and_sample_str_test: SamplesAndLabels,
        model : Word2Vec,
):
    def get_average_embedding(word_sequence : list[str]) :
        if word_sequence:
            return np.mean([model.wv[word] for word in word_sequence], axis=0)
        return np.zeros(model.vector_size)
    
    samples = pd.concat([samples_and_sample_str_train.sample, samples_and_sample_str_test.sample], ignore_index=True)
    labels = pd.concat([samples_and_sample_str_train.labels, samples_and_sample_str_test.labels], ignore_index=True)


    # Directly create the embeddings DataFrame
    embeddings_list = samples[USER_DATA_COLUMN].tolist()
    embeddings_list = [get_average_embedding(word_sequence) for word_sequence in embeddings_list]
    embeddings_df = pd.DataFrame(embeddings_list, columns=[f"feature_{i}" for i in range(model.vector_size)])

    assert len(embeddings_df) == len(labels), "The DataFrame and Series do not have the same number of rows!"

    embeddings_df["label"] = labels.astype("int16").tolist()

    embeddings_df.to_csv(os.path.join(params.OUTPUT_FOLDER, "word2vec_embedding.csv"), index=False)

