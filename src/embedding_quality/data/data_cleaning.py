
import pandas as pd
from embedding_quality.params.params import INFO_COLUMNS, ProgramParams
from commons.data_loading.data_types import SamplesAndLabels


def clean(params: ProgramParams, samples_and_labels: SamplesAndLabels) -> SamplesAndLabels:
    """
    Clean data.
    1. Remove columns that are composed off only one value
    2. Remove columns that are in the INFO_COLUMNS list
    3. Rmove row with NaN values
    """
    # ---------- Remove columns with only one unique value ----------
    samples = samples_and_labels.sample
    labels = samples_and_labels.labels
    # Find the indices of columns with only one unique value
    unique_value_columns = samples.nunique() == 1

    # Log the removed columns
    removed_columns = unique_value_columns[unique_value_columns].index
    params.RESULTS_LOGGER.info(f'Removing {len(removed_columns)} columns with only one unique value: {list(removed_columns)}')

    # Remove the columns with only one unique value from the samples
    samples = samples.loc[:, ~unique_value_columns]

    # ---------- Remove rows with NaN values ----------
    # Find the indices of rows with NaN values
    rows_with_nan = samples.isnull().any(axis=1)
    # Log all the removed rows
# Log the removed rows where NaN is not in the "truc" column
    for _, row in samples[rows_with_nan].iterrows():
        if pd.notna(row['Skewness']) and pd.notna(row['Kurtosis']):
            file_path = row['file_path']
            params.RESULTS_LOGGER.info(f'WARN : Removing row with NaN values in file_path: {file_path}, f_dtns_addr: {row["f_dtns_addr"]}')
    
    
    # Remove the rows with NaN values from the samples and labels
    samples = samples.loc[~rows_with_nan]
    labels = labels.loc[~rows_with_nan]

    params.RESULTS_LOGGER.info(f'Removing {len(rows_with_nan)} row with nan value.')

    # ---------- Remove columns in the INFO_COLUMNS list ----------




    return SamplesAndLabels(samples, labels).remove_columns(params, INFO_COLUMNS)