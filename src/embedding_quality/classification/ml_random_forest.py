from typing import Optional
from sklearn.ensemble import RandomForestClassifier

from research_base.utils.ml_utils.ml_evaluate import evaluate
from embedding_quality.params.params import ProgramParams
from embedding_quality.data_loading.data_types import SamplesAndLabels


def ml_random_forest_pipeline(
        params: ProgramParams, 
        samples_and_labels_train: SamplesAndLabels,
        samples_and_labels_test: SamplesAndLabels,
    ) -> None:
    """
    A pipeline for training a RandomForestClassifier.
    """
    # Train a RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=params.RANDOM_SEED, n_jobs = params.MAX_ML_WORKERS)
    clf.fit(samples_and_labels_train.sample, samples_and_labels_train.labels)
    
    # Evaluate model
    evaluate(
        clf,
        samples_and_labels_test.sample,
        samples_and_labels_test.labels,
        params.RESULTS_LOGGER,
        params.get_results_writer(),
    )
