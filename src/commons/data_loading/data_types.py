from typing import Optional, Tuple
from pandas import DataFrame, Series




from dataclasses import dataclass
import pandas as pd

from sklearn.model_selection import train_test_split

from commons.data_loading.data_origin import DataOriginEnum
from embedding_quality.params.params import ProgramParams


@dataclass
class SamplesAndLabels:
    sample : DataFrame
    labels : Series

    def remove_columns(self, params : ProgramParams, columns: list[str]) -> 'SamplesAndLabels':
        """
        Remove columns from the sample.
        """
        self.sample = self.sample.drop(columns=columns)
        params.RESULTS_LOGGER.info(f'Removing {len(columns)} : {columns}')

        return self

    def keep_columns(self, params : ProgramParams, columns: list[str]) -> 'SamplesAndLabels':
        """
        Keep columns from the sample.
        """
        self.sample = self.sample[columns]
        params.RESULTS_LOGGER.info(f'Keeping {len(columns)} : {columns}')

        return self


def split_dataset_if_needed(
    samples_and_labels_train: SamplesAndLabels, 
    samples_and_labels_test: Optional[SamplesAndLabels]
) -> tuple[SamplesAndLabels, SamplesAndLabels]:
    """
    Split data into training and test sets if needed.
    NOTE: Needed when no testing data is provided).
    """
    if samples_and_labels_test is None:
        # Split data into training and test sets
        __samples, __labels = samples_and_labels_train.sample, samples_and_labels_train.labels
        X_train, X_test, y_train, y_test = train_test_split(__samples, __labels, test_size=0.2, random_state=42)
        return SamplesAndLabels(X_train, y_train), SamplesAndLabels(X_test, y_test)
    else:
        return samples_and_labels_train, samples_and_labels_test


def handle_data_origin(
    data_origins: set[DataOriginEnum],
    origin_to_preprocessed_data: dict[DataOriginEnum, SamplesAndLabels]
) -> SamplesAndLabels:
    """
    Helper function for handling data origins.
    """
    __samples: list[pd.DataFrame] = []
    __labels: list[pd.Series] = []
    for origin in data_origins:
        preprocessed_data = origin_to_preprocessed_data[origin]

        samples, labels = preprocessed_data.sample, preprocessed_data.labels
        __samples += [samples]
        __labels += [labels]
       
    return SamplesAndLabels(pd.concat(__samples), pd.concat(__labels))


def split_preprocessed_data_by_origin(
        params: ProgramParams, 
        origin_to_samples_and_labels: dict[DataOriginEnum, SamplesAndLabels]
) -> Tuple[SamplesAndLabels, Optional[SamplesAndLabels]]:
    """
    Split samples and labels into training and testing sets.
    """
    preprocessed_data_train = handle_data_origin(
        params.data_origins_training,
        origin_to_samples_and_labels
    )
    preprocessed_data_test = None
    if params.data_origins_testing is not None:
        preprocessed_data_test = handle_data_origin(
            params.data_origins_testing,
            origin_to_samples_and_labels
        )
    
    return preprocessed_data_train, preprocessed_data_test


