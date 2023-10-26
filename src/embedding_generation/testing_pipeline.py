
from commons.data_loading.data_types import SamplesAndLabels
from embedding_coherence.pipeline import pipeline as coherence_pipeline
from embedding_quality.pipeline import pipeline as quality_pipeline
from params.common_params import CommonProgramParams


def testing_pipeline(params : CommonProgramParams, training : SamplesAndLabels, validation : SamplesAndLabels):
    """
    launch the testing pipeline
    """
    
    coherence_pipeline(params, (training, validation))

    quality_pipeline(params, (training, validation))