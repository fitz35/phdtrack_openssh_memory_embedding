
import os
from commons.data_loading.data_types import SamplesAndLabels


def testing_pipeline(dataset_path : str, training : SamplesAndLabels, validation : SamplesAndLabels):
    """
    launch the testing pipeline
    """

    # save the embeddings
    training.save_to_csv(os.path.join(dataset_path, "training.csv"))
    validation.save_to_csv(os.path.join(dataset_path, "validation.csv"))


    if "filtered" in dataset_path:
        filtered = "--no_balancing"
    else:
        filtered = ""
    

    os.system(f"python3 embedding_coherence_main.py -d {dataset_path} -otr training -ots validation")

    os.system(f"python3 embedding_quality_main.py -d {dataset_path} -otr training -ots testing {filtered}")
    

    # remove the embeddings
    os.remove(os.path.join(dataset_path, "training.csv"))
    os.remove(os.path.join(dataset_path, "validation.csv"))