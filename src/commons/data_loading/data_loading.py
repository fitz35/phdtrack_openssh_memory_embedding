


from concurrent.futures import ThreadPoolExecutor
import csv
import glob
from logging import Logger
import os
from typing import List, Tuple
import pandas as pd
from threading import Lock

from commons.data_loading.data_types import SamplesAndLabels
from commons.data_loading.data_origin import DataOriginEnum


PANDA_DTYPE_DEFAULT = "float64"


def __load_samples_and_labels_from_csv(
    csv_file_path: str,
    column_dtypes: dict[str, str] | str = PANDA_DTYPE_DEFAULT,
) -> SamplesAndLabels | None:
    # Load the data from the CSV file
    try:
        data = pd.read_csv(csv_file_path, dtype=column_dtypes)
    except pd.errors.EmptyDataError:
        print(f"The CSV {csv_file_path} file is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error during parsing the CSV {csv_file_path} file.")
        return None
    except FileNotFoundError:
        print(f"The specified CSV {csv_file_path} file was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the CSV {csv_file_path} file: {e}")
        return None

    # Check if data is empty
    if data.empty:
        return None

    try:
        # Extract the labels from the last column
        labels = data.iloc[:, -1]

        # Extract the samples from the other columns
        samples = data.iloc[:, :-1]

        # add file_path column
        samples["file_path"] = csv_file_path

        # identify the row with nan values (do it in the cleaning stage)
        #rows_with_nan = samples.isnull().any(axis=1)
        #if rows_with_nan.any():
        #    params.RESULTS_LOGGER.warning(f'Found rows with NaN values in {csv_file_path}: {samples[rows_with_nan].index}')

    except Exception as e:
        raise type(e)(e.__str__() + f". Error loading data from {csv_file_path}")

    return SamplesAndLabels(samples, labels)

def __generate_dtype_dict(file_path: str, 
                          default_type: str = PANDA_DTYPE_DEFAULT, 
                          special_cols_type: str = 'str', 
                          special_cols_keywords: list[str] = ['path', "hexa_representation"]) -> dict:
    """
    Generate a dictionary for dtype parameter in pd.read_csv where any column containing 
    any keyword from special_cols_keywords is of type special_cols_type, and all the others are of type default_type.

    :param file_path: path to the csv file
    :param default_type: default type for columns
    :param special_cols_type: type for special columns
    :param special_cols_keywords: list of keywords to identify special columns (NOTE : "hexa_representation" is a hack to load chunk user data as string)
    :return: dtype dict
    """
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # get the first line, i.e., header

    # create dtype dict: column_name -> type
    dtype_dict = {col: special_cols_type if any(keyword in col for keyword in special_cols_keywords) else default_type for col in header}

    return dtype_dict


def __parallel_load_samples_and_labels_from_all_csv_files(
        csv_file_paths: List[str],
        logger_common : Logger,
        logger_result : Logger,
        max_workers: int = 6,
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
        logger_common.info(f"📋 [{threadId}/{nb_threads}] Loading samples and labels from {csv_file_path}")

        res = __load_samples_and_labels_from_csv( csv_file_path, header_types)
        if res is None:
            list_of_empty_files.append(csv_file_path)
        else:
            samples, labels = res.sample, res.labels

            # Print the shapes of the arrays
            logger_common.debug(f'shape of samples: {samples.shape}, shape of labels: {labels.shape}')

            # Acquire the lock
            with concat_lock:
                all_samples_list.append(samples)
                all_labels_list.append(labels)
            # The lock is released after the 'with' statement
    
    logger_result.info(f'Loading samples and labels from {len(csv_file_paths)} files')

    # multi-threaded loading and generation of samples and labels
    with ThreadPoolExecutor(max_workers=min(max_workers, 6)) as executor:
        results = executor.map(
            load_samples_and_labels_from_csv_and_concatenate, 
            csv_file_paths,
            range(len(csv_file_paths)),
            [len(csv_file_paths)] * len(csv_file_paths)
        )
        for _ in results:
            pass

    logger_result.info(f'Number of loaded files: {len(csv_file_paths)}')
    logger_result.info(f'Number of empty files: {len(list_of_empty_files)}')

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
        dataset_path: str,
        logger_common : Logger,
        logger_result : Logger,
        data_origin: set[DataOriginEnum] | None = None,
        max_workers: int = 6,
) -> dict[DataOriginEnum, SamplesAndLabels]:
    """
    Load the samples and labels from all .csv files.
    Take into account the data origin: training, validation, testing.
    If data_origin is None, load all data.
    """
    
    # Get the filepaths for the training, validation, and testing data
    training_files, validation_files, testing_files = get_all_filepath_per_type(dataset_path)

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
        samples = __parallel_load_samples_and_labels_from_all_csv_files( files_to_load, logger_common, logger_result, max_workers)

        loaded[origin] = samples

    return loaded