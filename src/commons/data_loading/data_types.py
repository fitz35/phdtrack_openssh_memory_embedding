from logging import Logger
from typing import Optional, Tuple
from pandas import DataFrame, Series




from dataclasses import dataclass
import pandas as pd

from sklearn.model_selection import train_test_split

from commons.data_loading.data_origin import DataOriginEnum


@dataclass
class SamplesAndLabels:
    sample : DataFrame
    labels : Series

    def remove_columns(self, columns: list[str], logger: Logger) -> 'SamplesAndLabels':
        """
        Remove columns from the sample.
        """
        # Filter out columns that don't exist in the dataframe
        columns_to_remove = [col for col in columns if col in self.sample.columns]
        
        # Drop the columns
        self.sample.drop(columns=columns_to_remove, inplace=True)
        
        # Log the information
        logger.info(f'Removing {len(columns_to_remove)} columns (keeping {self.sample.shape[1]} columns): {columns_to_remove}')

        return self

    def keep_columns(self, columns: list[str], logger : Logger) -> 'SamplesAndLabels':
        """
        Keep columns from the sample.
        """
        self.sample = self.sample[columns]
        logger.info(f'Keeping {len(columns)} : {columns}')

        return self
    
    def save_to_csv(self, file_path : str):
        """
        Save the samples and labels to a CSV file.
        """
        embeddings_df = self.sample.copy()
        embeddings_df["label"] = self.labels.astype("int16").tolist()

        embeddings_df.to_csv(file_path, index=False)
    
    def copy(self) -> 'SamplesAndLabels':
        """
        Copy the samples and labels.
        """
        return SamplesAndLabels(self.sample.copy(), self.labels.copy())


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
        training_data_origins: set[DataOriginEnum],
        testing_data_origins: Optional[set[DataOriginEnum]],
        origin_to_samples_and_labels: dict[DataOriginEnum, SamplesAndLabels]
) -> Tuple[SamplesAndLabels, Optional[SamplesAndLabels]]:
    """
    Split samples and labels into training and testing sets.
    """
    preprocessed_data_train = handle_data_origin(
        training_data_origins,
        origin_to_samples_and_labels
    )
    preprocessed_data_test = None
    if testing_data_origins is not None:
        preprocessed_data_test = handle_data_origin(
            testing_data_origins,
            origin_to_samples_and_labels
        )
    
    return preprocessed_data_train, preprocessed_data_test


