from tqdm import tqdm
import numpy as np
import pandas as pd
from commons.data_loading.data_types import SamplesAndLabels
from commons.params.common_params import USER_DATA_COLUMN, CommonProgramParams
from commons.data_loading.data_origin import DataOriginEnum


def _clean(params: CommonProgramParams, samples_and_labels: SamplesAndLabels, info_column : list[str]) -> SamplesAndLabels:
    
    samples = samples_and_labels.sample
    labels = samples_and_labels.labels



    # ---------- Remove columns with only one unique value ----------
    
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
        if 'Skewness' in row and 'Kurtosis' in row and pd.notna(row['Skewness']) and pd.notna(row['Kurtosis']):
            file_path = row['file_path']
            params.RESULTS_LOGGER.info(f'WARN : Removing row with NaN values in file_path: {file_path}, f_dtns_addr: {row["f_dtns_addr"]}')
    
    
    # Remove the rows with NaN values from the samples and labels
    samples = samples.loc[~rows_with_nan]
    labels = labels.loc[~rows_with_nan]

    params.RESULTS_LOGGER.info(f'Removing {len(rows_with_nan)} row with nan value.')

    # ---------- Remove columns in the INFO_COLUMNS list ----------




    return SamplesAndLabels(samples, labels).remove_columns( info_column, params.RESULTS_LOGGER)


def clean_all(
        params: CommonProgramParams, 
        samples_and_labels: dict[DataOriginEnum, SamplesAndLabels], 
        info_column : list[str]
) -> dict[DataOriginEnum, SamplesAndLabels]:
    """
    Clean data.
    1. Remove columns that are composed off only one value
    2. If only one column in the embedding = chunk extract, so we will split it and pad it
    2. Remove columns that are in the INFO_COLUMNS list
    3. Rmove row with NaN values
    """

    # if we have only one column, explode the inside list into multiple columns, and pad the rows
    first_origin = next(iter(samples_and_labels))
    if len(samples_and_labels[first_origin].sample.columns) == 1 and samples_and_labels[first_origin].sample.columns[0] == USER_DATA_COLUMN:
        params.COMMON_LOGGER.info("Only one column, so we will split it and pad it")
        max_value = -1
        for origin in samples_and_labels:
            samples = samples_and_labels[origin].sample
            max_value = max(max_value, samples[USER_DATA_COLUMN].apply(len).max())
        params.RESULTS_LOGGER.info(f"max length of the user data : {max_value}")
        for origin in samples_and_labels:
            params.COMMON_LOGGER.info(f"Beginning the splitting for {origin} : padding")
            samples = samples_and_labels[origin].sample
            samples[USER_DATA_COLUMN] = samples[USER_DATA_COLUMN].apply(lambda x: x + ("0" * (max_value - len(x))))


            # split the column into multiple columns and convert them to int
            params.COMMON_LOGGER.info(f"Splitting the column for {origin}")
            samples = __split_and_convert_hexa(samples, USER_DATA_COLUMN, 8)


            samples_and_labels[origin] = SamplesAndLabels(samples, samples_and_labels[origin].labels)

    params.COMMON_LOGGER.info("Cleaning data")
    result = {}
    for origin in samples_and_labels:
        result[origin] = _clean(params, samples_and_labels[origin], info_column)
    
    return result


def __split_and_convert_hexa(samples: pd.DataFrame, column_name: str, slice_size: int) -> pd.DataFrame:
    """
    Splits hexastrings in a DataFrame column into chunks of a given size and converts them to int32.

    Parameters:
    - samples: The DataFrame containing the hex strings.
    - column_name: Name of the column containing the hex strings.
    - slice_size: Size of the chunks to split the hex strings.

    Returns:
    - A new DataFrame with the hex strings split and converted to int32.
    """
    
    # Determine the number of splits (columns)
    num_splits = len(samples[column_name].iloc[0]) // slice_size

    # Create a new DataFrame with the correct shape and dtype
    new_samples = pd.DataFrame(0, index=samples.index, columns=range(num_splits), dtype=np.int32)

    # Populate the new_samples DataFrame directly
    for idx, hex_str in tqdm(samples[column_name].items(), total=samples.shape[0], desc="Processing hex strings"):
        for col in range(num_splits):
            slice_start = col * slice_size
            hex_slice = hex_str[slice_start:slice_start+slice_size]
            new_samples.at[idx, col] = np.int32(int(hex_slice, 16))

    
    return new_samples
