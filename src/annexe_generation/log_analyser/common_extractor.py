

from typing import Callable, List, Tuple, Type, TypeVar
import traceback
import re


def extract_lines_between(log_lines: list[str], start_index: int, start_delimiter: str, end_delimiter: str):
    """
    Extracts lines from a list of log lines between lines that include specified start and end delimiters.
    
    The function starts searching from the start_index position in the log_lines list.
    It sets a flag to True when a line containing the start delimiter is found.
    All subsequent lines are added to the result list until a line containing the end delimiter is found.
    
    Args:
    log_lines (list[str]): The list of log lines.
    start_index (int): The index to start searching from.
    start_delimiter (str): The start delimiter.
    end_delimiter (str): The end delimiter.
    
    Returns:
    list[str]: The extracted lines between the start and end delimiters. (The start and end delimiters are included.)
    int: The index of the line containing the start delimiter.
    """
    inside_section = False
    start_delimiter_index = start_index
    extracted_lines : list[str] = []
    
    for line in log_lines[start_index:]:
        if start_delimiter in line:
            inside_section = True
            extracted_lines.append(line)
        elif end_delimiter in line:
            inside_section = False
            extracted_lines.append(line)
            break
        elif inside_section:
            extracted_lines.append(line)
        else:
            start_delimiter_index += 1
            
    return extracted_lines, start_delimiter_index


def extract_instance_name(log_lines: List[str], start_index: int, end_index: int) -> str:
    """
    Extracts the instance name from a subset of log lines.

    Iterates over a specified subset of log lines to find a line that matches the pattern "INFO - !+  [InstanceName] instance : ".
    If a match is found, the function returns the extracted instance name. 
    If no match is found within the specified lines, an AssertionError is raised.

    Args:
        log_lines (List[str]): The list of log lines.
        start_index (int): The starting index for the subset of lines to search.
        end_index (int): The ending index for the subset of lines to search.

    Returns:
        str: The extracted instance name.

    Raises:
        AssertionError: If no instance name is found in the specified lines.
    """
    for line in log_lines[start_index:end_index]:
        match = re.search(r"results_logger - INFO - !+ (\S+) instance : ", line)
        if match:
            return str(match.group(1))
    
    assert False, "Instance name not found"

def extract_instance_number(log_lines: List[str], start_index: int, end_index: int) -> int:
    """
    Extracts the instance number from a subset of log lines.

    Iterates over a specified subset of log lines to find a line that contains the pattern "index=[number]".
    If a match is found, the function returns the extracted number as an integer. 
    If no match is found within the specified lines, an AssertionError is raised.

    Args:
        log_lines (List[str]): The list of log lines.
        start_index (int): The starting index for the subset of lines to search.
        end_index (int): The ending index for the subset of lines to search.

    Returns:
        int: The extracted instance number.

    Raises:
        AssertionError: If no instance number is found in the specified lines.
    """
    for line in log_lines[start_index:end_index]:
        match = re.search(r"index=(\d+)", line)
        if match:
            return int(match.group(1))
    
    assert False, "Instance number not found"


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

def __get_next_instance(log_lines: list[str], begin_line : int) -> int | None:
    result_index = begin_line
    for line in log_lines[begin_line:]:
        match = re.search(r"results_logger - INFO - !+ (\S+) instance : ", line)
        if match:
            return result_index
        result_index += 1

    return None


def __is_timeout_lines(log_lines: List[str], begin_line : int, next_instance_line : int) -> bool:

    for line in log_lines[begin_line:next_instance_line]:
        match = re.search(r"- results_logger - ERROR - Timeout error in transformers pipeline \d+, skipping \(and marking\)", line)
        if match:
            return True
        
    return False


def __is_already_computed(log_lines: List[str], begin_line : int) -> bool:
    pattern = re.search(r"- results_logger - INFO - \S+ instance .+ already computed", log_lines[begin_line + 1])
    return pattern is not None

def __is_end(log_lines: List[str], begin_line: int) -> bool:
    end_time_pattern = re.compile(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2} - results_logger - INFO - Pipeline end time : \d+\.\d+ seconds")
    duration_pattern = re.compile(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2} - results_logger - INFO - Pipeline duration : \d+\.\d+ seconds")

    end_time_match = end_time_pattern.match(log_lines[begin_line])
    duration_match = duration_pattern.match(log_lines[begin_line + 1])

    return end_time_match is not None and duration_match is not None


T = TypeVar('T')

def extract_all_dataset_results(log_lines: list[str], extractor: Callable[[list[str], int, str, str], T], output_correlation_matrix_dir_relative_path : str) -> Tuple[List[T], List[dict[str, str]]]:
    """
    Extracts results from a list of log lines using a specified extractor function and data class.

    Args:
    log_lines (list[str]): The list of log lines.
    extractor (Callable[[list[str], int, str], tuple[T, int]]): The extractor function to be used.
    dataclass (Type[T]): The data class for the results.

    Returns:
    List[T]: A list of results as instances of the specified data class.
    List[dict[str, str]]: A list of dictionaries containing the dataset path and instance name for each timeout instance.
    """
    results = []
    timeout_instances = []
    dataset_path = None
    i = 0
    while dataset_path is None and i < len(log_lines):
        dataset_path = __extract_dataset_path(log_lines[i])
        i += 1

    if dataset_path is None:
        return results, timeout_instances

    begin_index = __get_next_instance(log_lines, 0)
    if begin_index is None:
        return results, timeout_instances

    while begin_index < len(log_lines):
        try:
            if __is_already_computed(log_lines, begin_index):
                begin_index += 2
                if __is_end(log_lines, begin_index):
                    return results, timeout_instances
                continue

            maybe_next_instance = __get_next_instance(log_lines, begin_index + 1)

            if maybe_next_instance is not None and __is_timeout_lines(log_lines, begin_index, maybe_next_instance):
                instance_name = extract_instance_name(log_lines, begin_index, maybe_next_instance)
                instance_number = extract_instance_number(log_lines, begin_index, maybe_next_instance)
                timeout_instances.append({
                    "instance" : instance_name + "_" + str(instance_number),
                    "dataset" : dataset_path
                })
                
                begin_index = maybe_next_instance
                continue

            # check if we have minimum 10 lines (remove corrupted files)
            if maybe_next_instance is not None:
                if maybe_next_instance - begin_index < 10:
                    begin_index = maybe_next_instance
                    continue
            else:
                if len(log_lines) - begin_index < 10:
                    return results, timeout_instances

            result = extractor(log_lines, begin_index, dataset_path, output_correlation_matrix_dir_relative_path)
            results.append(result)
            
            # check end of file
            if maybe_next_instance is None:
                return results, timeout_instances
            
            if __is_end(log_lines, maybe_next_instance):
                return results, timeout_instances
                
            begin_index = maybe_next_instance
        except AssertionError as e:
            print(f"An error occurred: {e}, line {begin_index}")
            traceback.print_exc()
            return results, timeout_instances

    return results, timeout_instances
