import pandas as pd

from research_base.utils.results_utils import time_measure_result
from research_base.utils.data_utils import dict_to_csv_value, count_labels

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from embedding_quality.data_balancing.balancing_params import BalancingStrategies
from commons.data_loading.data_types import SamplesAndLabels
from params.common_params import BALANCING_STRATEGY, CommonProgramParams


SAMPLING_STRATEGY_TO_RESAMPLING_FUNCTION = {
    BalancingStrategies.UNDERSAMPLING: RandomUnderSampler,
    BalancingStrategies.OVERSAMPLING: RandomOverSampler,
    BalancingStrategies.SMOTE: SMOTE,
    BalancingStrategies.ADASYN: ADASYN,
}

def resample_data(
    sampler_class, 
    params: CommonProgramParams,
    samples: pd.DataFrame,
    labels: pd.Series,
) -> SamplesAndLabels:
    with time_measure_result(
            f'resample_data ({sampler_class.__name__})', 
            params.RESULTS_LOGGER,
        ):
        sampler = sampler_class(random_state=params.RANDOM_SEED)
        X_res, y_res = sampler.fit_resample(samples, labels)
    return SamplesAndLabels(X_res, y_res)

def apply_balancing(
    params: CommonProgramParams,
    samples: pd.DataFrame,
    labels: pd.Series
) -> SamplesAndLabels:
    """
    Get the rebalanced data.
    """    
    
    params.RESULTS_LOGGER.info(f"Number of samples before balancing: {dict_to_csv_value(count_labels(labels))}")

    if params.no_balancing or BALANCING_STRATEGY == BalancingStrategies.NO_BALANCING:
        sample_and_labels = SamplesAndLabels(samples, labels)
    elif BALANCING_STRATEGY in SAMPLING_STRATEGY_TO_RESAMPLING_FUNCTION.keys():
        sample_and_labels = resample_data(
            SAMPLING_STRATEGY_TO_RESAMPLING_FUNCTION[BALANCING_STRATEGY],
            params, 
            samples, 
            labels
        )
    else:
        raise ValueError(f"Invalid balancing strategy: {BALANCING_STRATEGY}")
    
    params.RESULTS_LOGGER.info(f"Number of samples after balancing: {dict_to_csv_value(count_labels(sample_and_labels.labels))}")

    return sample_and_labels
    