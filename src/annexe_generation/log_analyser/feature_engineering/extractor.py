

import os
import re
import sys

import pandas as pd


sys.path.append(os.path.abspath('../../..'))
from annexe_generation.log_analyser.common_extractor import extract_instance_name, extract_instance_number, extract_lines_between
from annexe_generation.log_analyser.feature_engineering.feature_engineering_data import CorrelationSum, FeatureEngineeringData

PROJECT_ROOT_NAME = "phdtrack_openssh_memory_embedding/"


def __extract_feature_engineering_lines(log_lines: list[str], start_index: int):
    """
    A convenience function to extract log lines related to the random forest operation.
    
    This function is a wrapper around the extract_lines_between function.
    It specifies the start and end delimiters related to the random forest operation logs.
    
    Args:
    log_lines (list[str]): The list of log lines.
    start_index (int): The index to start searching from.
    
    Returns:
    list[str]: The extracted random forest related log lines (including the start and end delimiters.)
    int: The index of the line containing the start delimiter.
    """
    start_delimiter = "- results_logger - INFO - timer for feature_engineering started"
    end_delimiter = "- results_logger - INFO - End feature engineering"
    return extract_lines_between(log_lines, start_index, start_delimiter, end_delimiter)


def __extract_correlation_matrix_paths(log_lines: list[str]) -> list[str]:

    # Define the regex pattern to extract the paths
    path_regex = r"(/[\w/._-]+)"
    
    # Search for the specific log line pattern and extract the paths
    for log_line in log_lines:
        match = re.search(r"Correlation matrix saved at: (.+)$", log_line)
        if match:
            paths = re.findall(path_regex, match.group(1))
            return paths
    assert False, "No correlation matrix path found in the log file."


def __extract_correlation_sum(log_lines: list[str]) -> list[CorrelationSum]:
    start_extraction = False
    extracted_lines : list[str] = []

    # Define the start and end patterns
    timestamp_pattern = r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}"
    start_pattern = re.compile(f"{timestamp_pattern} - results_logger - INFO - Sorted correlation sums:")
    end_pattern = re.compile("Length: \d+, dtype: float64")

    # Iterate through the lines and extract the relevant section
    for line in log_lines:
        if start_pattern.search(line):
            start_extraction = True
        elif end_pattern.search(line):
            break
        elif start_extraction:
            extracted_lines.append(line.strip())

    # Create a dictionary to hold feature and corresponding value
    feature_dict : dict[str, float] = {}
    feature_name: list[str] = []
    for line in extracted_lines:
        parts = line.split()
        if len(parts) == 2 and parts[1].replace('.', '', 1).isdigit():
            feature, value = parts
            feature_dict[feature] = float(value)
            feature_name.append(feature)

    # Convert the dictionary to a the CorrelationSum
    result_list : list[CorrelationSum] = []
    for feature in feature_name:
        result_list.append(CorrelationSum(feature, feature_dict[feature]))
        
    
    return result_list


def __get_right_correlation_matrix_path(correlation_matrix_paths: list[str], output_correlation_matrix_dir_relative_path : str) -> list[str]:
   
    project_dir_path = os.path.abspath(__file__)
    while PROJECT_ROOT_NAME not in project_dir_path:
        project_dir_path = os.path.dirname(project_dir_path)

    project_dir_path = os.path.join(project_dir_path, PROJECT_ROOT_NAME)

    right_paths: list[str] = []
    for path in correlation_matrix_paths:
        relevant_path = os.path.basename(path)

        right_paths.append(os.path.join(project_dir_path, output_correlation_matrix_dir_relative_path, relevant_path))
    
    return right_paths

def feature_engineering_extractor(all_lines : list[str], begin_index : int, dataset_path : str, output_correlation_matrix_dir_relative_path : str) -> FeatureEngineeringData :
    dataset_name = os.path.basename(dataset_path)
    # get the random forest lines
    feature_engineering_lines, feature_engineering_start_index = __extract_feature_engineering_lines(all_lines, begin_index)

    instance_name = extract_instance_name(all_lines, begin_index, feature_engineering_start_index)
    instance_number = extract_instance_number(all_lines, begin_index, feature_engineering_start_index)

    instance_name = instance_name + " " + str(instance_number)


    # get the correlation matrix paths
    correlation_matrix_paths = __get_right_correlation_matrix_path(
        __extract_correlation_matrix_paths(feature_engineering_lines), 
        output_correlation_matrix_dir_relative_path
        )
    assert len(correlation_matrix_paths) == 2, "There should be two correlation matrix paths"

    # get the correlation pd
    correlation_matrix = pd.read_csv(correlation_matrix_paths[0], index_col=0)

    # Replacing missing values (if any) with NaN
    correlation_matrix = correlation_matrix.apply(pd.to_numeric, errors='coerce')

    # get the correlation sum
    correlation_sum = __extract_correlation_sum(feature_engineering_lines)

    # get the correlation image path
    correlation_image_path = correlation_matrix_paths[1]

    return FeatureEngineeringData(
        dataset_name,
        instance_name,
        correlation_matrix,
        correlation_image_path,
        correlation_sum
    )
