


from concurrent.futures import ThreadPoolExecutor
import csv
import glob
import os
from typing import List, Tuple
import pandas as pd
from threading import Lock

from embedding_quality.data_loading.data_types import SamplesAndLabels
from embedding_quality.params.params import ProgramParams
from embedding_quality.params.data_origin import DataOriginEnum
from embedding_quality.data_loading.data_cleaning import clean
from commons.utils.data_utils import count_positive_and_negative_labels


PANDA_DTYPE_DEFAULT = "float64"

def log_positive_and_negative_labels(params: ProgramParams, labels: pd.Series, message: str = "") -> None:
    nb_positive_labels, nb_negative_labels = count_positive_and_negative_labels(labels)

    if message != "":
        params.RESULTS_LOGGER.info(message)

    params.RESULTS_LOGGER.info(f'Number of positive labels: {nb_positive_labels}')
    params.RESULTS_LOGGER.info(f'Number of negative labels: {nb_negative_labels}')


def __load_samples_and_labels_from_csv(
    params : ProgramParams,
    csv_file_path: str,
    column_dtypes: dict[str, str] | str = PANDA_DTYPE_DEFAULT,
) -> SamplesAndLabels | None:
    # Load the data from the CSV file
    data = pd.read_csv(csv_file_path, dtype=column_dtypes)

    # Check if data is empty
    if data.empty:
        return None

    try:
        # Extract the labels from the last column
        labels = data.iloc[:, -1]

        # Extract the samples from the other columns
        samples = data.iloc[:, :-1]

        # identify the row with nan values (do it in the cleaning stage)
        #rows_with_nan = samples.isnull().any(axis=1)
        #if rows_with_nan.any():
        #    params.RESULTS_LOGGER.warning(f'Found rows with NaN values in {csv_file_path}: {samples[rows_with_nan].index}')

    except Exception as e:
        raise type(e)(e.__str__() + f". Error loading data from {csv_file_path}")

    return SamplesAndLabels(samples, labels)

def __generate_dtype_dict(file_path: str, default_type: str = PANDA_DTYPE_DEFAULT, special_cols_type: str = 'str', special_cols_keyword: str = 'path') -> dict:
    """
    Generate a dictionary for dtype parameter in pd.read_csv where any column containing 
    special_cols_keyword is of type special_cols_type, and all the others are of type default_type.

    :param file_path: path to the csv file
    :param default_type: default type for columns
    :param special_cols_type: type for special columns
    :param special_cols_keyword: keyword to identify special columns
    :return: dtype dict
    """
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # get the first line, i.e., header

    # create dtype dict: column_name -> type
    dtype_dict = {col: special_cols_type if special_cols_keyword in col else default_type for col in header}

    return dtype_dict


def __parallel_load_samples_and_labels_from_all_csv_files(
        params: ProgramParams, csv_file_paths: List[str]
) -> SamplesAndLabels:
    """
    Load the samples and labels from all .csv files.
    Load using multiple threads.
    """
    # stats
    list_of_empty_files = []

    all_samples_list: List[pd.DataFrame] = []
    all_labels_list: List[pd.Series] = []

    # determine header types
    first_file = csv_file_paths[0]
    header_types = __generate_dtype_dict(first_file)

    # Define a lock for thread safety
    concat_lock = Lock()

    def load_samples_and_labels_from_csv_and_concatenate(
        csv_file_path: str, 
        threadId: int, 
        nb_threads: int
    ) -> None:
        """
        Load the samples and labels from one .csv file.
        """
        params.RESULTS_LOGGER.info(f"📋 [{threadId}/{nb_threads}] Loading samples and labels from {csv_file_path}")

        res = __load_samples_and_labels_from_csv(params, csv_file_path, header_types)
        if res is None:
            list_of_empty_files.append(csv_file_path)
        else:
            samples, labels = res.sample, res.labels

            # Print the shapes of the arrays
            params.COMMON_LOGGER.debug(f'shape of samples: {samples.shape}, shape of labels: {labels.shape}')

            # Acquire the lock
            with concat_lock:
                all_samples_list.append(samples)
                all_labels_list.append(labels)
            # The lock is released after the 'with' statement

    # multi-threaded loading and generation of samples and labels
    with ThreadPoolExecutor(max_workers=min(params.MAX_ML_WORKERS, 6)) as executor:
        results = executor.map(
            load_samples_and_labels_from_csv_and_concatenate, 
            csv_file_paths,
            range(len(csv_file_paths)),
            [len(csv_file_paths)] * len(csv_file_paths)
        )
        for _ in results:
            pass

    params.COMMON_LOGGER.info(f'Number of empty files: {len(list_of_empty_files)}')

    # Concatenate DataFrames and labels Series
    all_samples = pd.concat(all_samples_list, ignore_index=True)
    all_labels = pd.concat(all_labels_list, ignore_index=True)

    return SamplesAndLabels(all_samples, all_labels)

def get_all_filepath_per_type(dirpath: str) -> Tuple[list[str], list[str], list[str]]:
    """
    Determine the filepaths for all data .csv files inside the directory.
    Return the filepaths for the training, validation, and testing data.
    """
    extension = "csv"
    all_files = glob.glob(os.path.join(dirpath, "**", f"*.{extension}"), recursive=True)

    training_files = [file for file in all_files if DataOriginEnum.Training.value in file.lower()]
    validation_files = [file for file in all_files if DataOriginEnum.Validation.value in file.lower()]
    testing_files = [file for file in all_files if DataOriginEnum.Testing.value in file.lower()]
        
    return training_files, validation_files, testing_files

def load(
        params: ProgramParams,
        data_origin: set[DataOriginEnum] | None = None
) -> dict[DataOriginEnum, SamplesAndLabels]:
    """
    Load the samples and labels from all .csv files.
    Take into account the data origin: training, validation, testing.
    If data_origin is None, load all data.
    """
    
    # Get the filepaths for the training, validation, and testing data
    training_files, validation_files, testing_files = get_all_filepath_per_type(params.dataset_path)

    loaded = {}
    if data_origin is None:
        data_origin = {DataOriginEnum.Training, DataOriginEnum.Validation, DataOriginEnum.Testing}
    
    for origin in data_origin:
        if origin == DataOriginEnum.Training:
            files_to_load = training_files
        elif origin == DataOriginEnum.Validation:
            files_to_load = validation_files
        elif origin == DataOriginEnum.Testing:
            files_to_load = testing_files
        else:
            raise ValueError(f"Unknown data origin: {origin}")


        #training_samples, training_labels = load_samples_and_labels_from_all_csv_files(params, training_files)
        samples = __parallel_load_samples_and_labels_from_all_csv_files(params, files_to_load)

        samples_clean = clean(
            params,
            samples
        )

        log_positive_and_negative_labels(
            params, 
            samples_clean.labels, 
            "Loaded data: ({})".format(", ".join([origin.value for origin in data_origin])) if data_origin is not None else "All data"
        )

        loaded[origin] = samples_clean

    return loaded