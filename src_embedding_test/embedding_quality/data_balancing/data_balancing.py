import pandas as pd

from research_base.utils.results_utils import time_measure_result
from research_base.utils.data_utils import dict_to_csv_value, count_labels

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from embedding_quality.data_balancing.balancing_params import BalancingStrategies
from commons.data_loading.data_types import SamplesAndLabels
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
            params.get_results_writer(), 
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
        params.set_result_for(
            "nb_samples_before_balancing",
            dict_to_csv_value(count_labels(labels))
        )

        sample_and_labels = resample_data(
            SAMPLING_STRATEGY_TO_RESAMPLING_FUNCTION[BALANCING_STRATEGY],
            params, 
            samples, 
            labels
        )
        X_res, y_res = sample_and_labels.sample, sample_and_labels.labels
        params.set_result_for(
            "nb_samples_after_balancing",
            dict_to_csv_value(count_labels(y_res))
        )

        return SamplesAndLabels(X_res, y_res)
    else:
        raise ValueError(f"Invalid balancing strategy: {BALANCING_STRATEGY}")