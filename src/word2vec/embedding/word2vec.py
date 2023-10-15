
from gensim.models import Word2Vec

from commons.data_loading.data_types import SamplesAndLabels
from word2vec.params.params import USER_DATA_COLUMN, WORD2VEC_MIN_COUNT, WORD2VEC_VECTOR_SIZE, WORD2VEC_WINDOW_BYTES_SIZE, ProgramParams


def word2vec_pipeline(
        params : ProgramParams,
        samples_and_sample_str_train: SamplesAndLabels,
):
    
    sentences = samples_and_sample_str_train.sample[USER_DATA_COLUMN].tolist()
    model = Word2Vec(
        sentences, 
        vector_size=WORD2VEC_VECTOR_SIZE, 
        window=int(WORD2VEC_WINDOW_BYTES_SIZE/params.WORD_BYTE_SIZE), 
        min_count=WORD2VEC_MIN_COUNT, 
        workers=params.MAX_ML_WORKERS
    )


    return model