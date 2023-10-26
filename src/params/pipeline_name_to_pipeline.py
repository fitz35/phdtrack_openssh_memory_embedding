
from params.pipelines import Pipeline
from embedding_coherence.pipeline import pipeline as embedding_coherence_pipeline
from embedding_generation.pipeline import pipeline as deep_learning_pipeline
from embedding_quality.pipeline import pipeline as embedding_quality_pipeline


PIPELINES_NAME_TO_PIPELINE = {
    Pipeline.DeepLearning : deep_learning_pipeline,
    Pipeline.EmbeddingCoherence : embedding_coherence_pipeline,
    Pipeline.EmbeddingQuality : embedding_quality_pipeline,
}