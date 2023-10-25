
import sys
from commons.data_loading.data_types import SamplesAndLabels
from embedding_quality.params.params import ProgramParams as ClassifierParams
from embedding_quality.pipeline import pipeline as classifier_pipeline
from embedding_coherence.params.params import ProgramParams as ClusteringParams
from embedding_coherence.pipeline import pipeline as clustering_pipeline


def testing_pipeline(dataset : str, training : SamplesAndLabels, validation : SamplesAndLabels):
    """
    launch the testing pipeline
    """
    if "filtered" in dataset:
        args = ["-d", dataset, "-otr", "training", "-ots", "validation", "--no_balancing"]
    else:
        args = ["-d", dataset, "-otr", "training", "-ots", "validation"]

    # random forest
    sys.argv[1:] = args
    params = ClassifierParams(dotenv_path="embedding_quality/.env", construct_log=False)
    
    classifier_pipeline(params, (training, validation))


    # clustering
    sys.argv[1:] = ["-d", dataset, "-otr", "training", "-ots", "validation"]
    params = ClusteringParams(dotenv_path="embedding_coherence/.env", construct_log=False)
    clustering_pipeline(params, (training, validation))
