import pandas as pd

from commons.utils.results_utils import time_measure_result

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from embedding_quality.data_balancing.balancing_params import BalancingStrategies
from embedding_quality.data_loading.data_types import SamplesAndLabels
from embedding_quality.params.params import BALANCING_STRATEGY, ProgramParams


SAMPLING_STRATEGY_TO_RESAMPLING_FUNCTION = {
    BalancingStrategies.UNDERSAMPLING: RandomUnderSampler,
    BalancingStrategies.OVERSAMPLING: RandomOverSampler,
    BalancingStrategies.SMOTE: SMOTE,
    BalancingStrategies.ADASYN: ADASYN,
}

def resample_data(
    sampler_class, 
    params: ProgramParams,
    samples: pd.DataFrame,
    labels: pd.Series,
) -> SamplesAndLabels:
    with time_measure_result(
            f'resample_data ({sampler_class.__name__})', 
            params.RESULTS_LOGGER, 
            params.results_writer, 
            "data_balancing_duration"
        ):
        sampler = sampler_class(random_state=params.RANDOM_SEED)
        X_res, y_res = sampler.fit_resample(samples, labels)
    return SamplesAndLabels(X_res, y_res)

def apply_balancing(
    params: ProgramParams,
    samples: pd.DataFrame,
    labels: pd.Series
) -> SamplesAndLabels:
    """
    Get the rebalanced data.
    """    
    if BALANCING_STRATEGY == BalancingStrategies.NO_BALANCING:
        return SamplesAndLabels(samples, labels)
    elif BALANCING_STRATEGY in SAMPLING_STRATEGY_TO_RESAMPLING_FUNCTION.keys():
        params.results_writer.set_result(
            "nb_training_samples_before_balancing",
            str(len(samples))
        )
        params.results_writer.set_result(
            "nb_positive_training_samples_before_balancing",
            str(len(labels[labels == 1]))
        )

        X_res, y_res = resample_data(
            SAMPLING_STRATEGY_TO_RESAMPLING_FUNCTION[BALANCING_STRATEGY],
            params, 
            samples, 
            labels
        )

        params.results_writer.set_result(
            "nb_training_samples_after_balancing",
            str(len(X_res))
        )
        params.results_writer.set_result(
            "nb_positive_training_samples_after_balancing",
            str(len(y_res[y_res == 1]))
        )

        return SamplesAndLabels(X_res, y_res)
    else:
        raise ValueError(f"Invalid balancing strategy: {params.balancing_strategy}")
