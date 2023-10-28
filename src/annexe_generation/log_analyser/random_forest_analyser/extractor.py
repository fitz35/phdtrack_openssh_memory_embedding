

import json
import os
import re
import sys
from typing import List, Tuple

sys.path.append(os.path.abspath('../../..'))
from annexe_generation.log_analyser.common_extractor import extract_lines_between
from annexe_generation.log_analyser.random_forest_analyser.classifier_data import ClassificationResults

def __extract_dataset_path(log_line: str):
    """
    Extracts the dataset path from a log line.
    
    The function searches for the pattern "Launching embedding pipeline on dataset " followed by a non-whitespace sequence.
    If found, it returns the non-whitespace sequence (i.e., the dataset path).
    If not found, it returns None.
    
    Args:
    log_line (str): The log line to search in.
    
    Returns:
    str or None: The extracted dataset path or None if not found.
    """
    match = re.search(r"- results_logger - INFO - ///---!!!! Launching embedding pipeline on dataset (\S+)", log_line)
    if not match:
        return None
    return match.group(1)


def __extract_instance_name(log_line: str):
    """
    Extracts the instance name from a log line.
    
    The function searches for the pattern "INFO - !+  " followed by a non-whitespace sequence, ending with " instance : ".
    If found, it returns the non-whitespace sequence (i.e., the instance name).
    If not found, it returns None.
    
    Args:
    log_line (str): The log line to search in.
    
    Returns:
    str or None: The extracted instance name or None if not found.
    """
    match = re.search(r"results_logger - INFO - !!!!!!!!!!!!! (\S+) instance : ", log_line)
    if not match:
        return None
    return match.group(1)


def __extract_instance_number(log_line):
    """
    Extracts the instance number from a log line.
    
    The function searches for the pattern "index=" followed by one or more digits.
    If found, it converts the digit sequence to an integer and returns it.
    If not found, it returns None.
    
    Args:
    log_line (str): The log line to search in.
    
    Returns:
    int or None: The extracted instance number or None if not found.
    """
    match = re.search(r"index=(\d+)", log_line)
    if not match:
        return None
    return int(match.group(1))



def __extract_random_forest_lines(log_lines: list[str], start_index: int):
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
    start_delimiter = "///---!!!! Launching testing pipeline on dataset"
    end_delimiter = "Time elapsed since the begining of random forest"
    return extract_lines_between(log_lines, start_index, start_delimiter, end_delimiter)


def __extract_and_decode_json(log_lines: list[str]) -> dict:
    """
    Extracts a JSON string from a list of log lines when a specific delimiter is found,
    decodes it, and returns the resulting dictionary.

    Args:
    log_lines (list[str]): The list of log lines.

    Returns:
    dict: The decoded JSON object.
    """
    json_string = ""
    capturing = False
    brace_count = 0

    for line in log_lines:
        if "- results_logger - INFO - {" in line:
            capturing = True
            brace_count += line.count("{")
            # Remove the log timestamp and logger information from the first line
            line = line.split("- results_logger - INFO - ", 1)[1]
            json_string += line
        elif capturing:
            brace_count += line.count("{")
            brace_count -= line.count("}")
            json_string += line
            if brace_count == 0:
                break

    if json_string:
        return json.loads(json_string)
    else:
        raise ValueError("No JSON string found in log lines")

def __extract_metrics(log_lines: List[str]) -> Tuple[int, int, int, int, float]:
    """
    Extract classification metrics and AUC from a list of log lines.

    Args:
    log_lines (List[str]): A list of log lines to search through.

    Returns:
    Tuple[int, int, int, int, float]: A tuple containing the extracted metrics in the order:
                                      (True Positives, True Negatives, False Positives, False Negatives, AUC).

    Raises:
    ValueError: If any of the required metrics are not found in the log lines.
    """
    # Dictionary to store the extracted metrics
    metrics = {}
    
    # Regular expression pattern to match the metrics in the log lines
    pattern = re.compile(r"INFO - (True Positives|True Negatives|False Positives|False Negatives|AUC): (\d+\.?\d*)")
    
    # Iterate through each line in the log lines
    for line in log_lines:
        # Search for the pattern in the current line
        match = pattern.search(line)
        if match:
            # If a match is found, extract the metric name and its value
            metric, value = match.groups()
            # Convert the value to float if it contains a decimal point, otherwise convert to int
            metrics[metric.lower().replace(" ", "_")] = float(value) if '.' in value else int(value)
    
    print(metrics)
    # Check if all required metrics were found
    required_metrics = ["true_positives", "true_negatives", "false_positives", "false_negatives", "auc"]
    if not all(key in metrics.keys() for key in required_metrics):
        raise ValueError("Not all required metrics were found in the log lines, found metrics: " + str(metrics))

    # Return the extracted metrics as a tuple
    return (
        metrics["true_positives"],
        metrics["true_negatives"],
        metrics["false_positives"],
        metrics["false_negatives"],
        metrics["auc"]
    )

def __extract_time_elapsed(log_lines: List[str]) -> float:
    """
    Extracts the time elapsed since the beginning of a random forest operation from log lines.

    Args:
    log_lines (List[str]): A list of log lines to search through.

    Returns:
    float: The time elapsed in seconds.

    Raises:
    ValueError: If the time elapsed information is not found in the log lines.
    """
    # Regular expression pattern to match the time elapsed information
    pattern = re.compile(r"Time elapsed since the begining of random forest : : (\d+\.\d+) s")

    # Iterate through each line in the log lines
    for line in log_lines:
        # Search for the pattern in the current line
        match = pattern.search(line)
        if match:
            # If a match is found, extract the time elapsed value and convert it to float
            return float(match.group(1))

    # If the time elapsed information was not found, raise an error
    raise ValueError("Time elapsed information not found in log lines")



def __random_forest_extractor(all_lines : list[str], begin_index : int, dataset_path : str) -> Tuple[ClassificationResults, int] :
        """
        Extracts information related to a random forest operation from log lines.

        Args:
        all_lines (list[str]): All the lines in the log file.
        begin_index (int): The index in the log lines to start the extraction from.

        Returns:
        Tuple[ClassificationResults, int]: A tuple containing the extracted classification results and the index of the next line after the extraction.

        Raises:
        AssertionError: If the dataset path, instance name, or instance number is not found.
        """
        # get the random forest lines
        random_forest_lines, random_forest_start_index = __extract_random_forest_lines(all_lines, begin_index)
        #print(random_forest_start_index)

        # iterate through the line preceding the random forest lines to get the dataset path and instance name
        
        instance_name = None
        instance_number = None

        for i in range(begin_index, random_forest_start_index, 1):
            if instance_name is None:
                instance_name = __extract_instance_name(all_lines[i])
            if instance_number is None:
                instance_number = __extract_instance_number(all_lines[i])
        
        assert instance_name is not None, "Instance name not found"
        assert instance_number is not None, "Instance number not found"

        # extract the metrics from the random forest lines
        true_positives, true_negatives, false_positives, false_negatives, auc = __extract_metrics(random_forest_lines)

        # extract the JSON string from the random forest lines
        json_data = __extract_and_decode_json(random_forest_lines)

        # extract the duration of the random forest operation
        duration = __extract_time_elapsed(random_forest_lines)

        # create the ClassificationResults object
        return ClassificationResults.from_json(
            json_data, 
            dataset_path, 
            instance_name, 
            true_positives, 
            true_negatives,
            false_positives, 
            false_negatives, 
            auc, 
            duration), random_forest_start_index + len(random_forest_lines) + 1


def extract_all_dataset_random_forest_results(log_lines: list[str]) -> List[ClassificationResults]:
    """
    Extracts classification results from a list of log lines.

    Args:
    log_lines (list[str]): The list of log lines.

    Returns:
    List[ClassificationResults]: A list of ClassificationResults objects.
    """
    # List to store the extracted classification results
    results : list[ClassificationResults] = []

    dataset_path = None

    # Index to start searching from
    begin_index = 0

    # Iterate through the log lines
    while begin_index < len(log_lines):
        if dataset_path is None:
            dataset_path = __extract_dataset_path(log_lines[begin_index])
            begin_index += 1
        else:
            try:
                # Extract the classification results and the index of the next line
                classification_results, next_index = __random_forest_extractor(log_lines, begin_index, dataset_path)
                # Add the classification results to the list
                results.append(classification_results)
                # Set the begin_index to the index of the next line
                begin_index = next_index
            except AssertionError as e:
                # If an error occurs, increment the begin_index and continue
                begin_index += 1
            except ValueError as e:
                # If an error occurs, print it and continue
                print(f"An error occurred: {e}")
                begin_index += 1

    # Return the list of classification results
    return results
