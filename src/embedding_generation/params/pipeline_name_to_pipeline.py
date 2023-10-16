

from embedding_generation.embedding.word2vec import word2vec_pipeline
from embedding_generation.params.pipelines import Pipeline


PIPELINES_NAME_TO_PIPELINE = {
    Pipeline.Word2Vec : word2vec_pipeline
}