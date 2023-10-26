
from params.pipelines import Pipeline
from embedding_generation.pipeline import pipeline as deep_learning_pipeline
from testing_pipelines.pipeline import pipeline as testing_pipeline


PIPELINES_NAME_TO_PIPELINE = {
    Pipeline.DeepLearning : deep_learning_pipeline,
    Pipeline.TestingEmbedding : testing_pipeline,
}