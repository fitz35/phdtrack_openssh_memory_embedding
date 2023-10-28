

from typing import Callable, List, Type, TypeVar
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


def extract_instance_name(log_line: str):
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


def extract_instance_number(log_line):
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


T = TypeVar('T')

def extract_all_dataset_results(log_lines: list[str], extractor: Callable[[list[str], int, str], tuple[T, int]]) -> List[T]:
    """
    Extracts results from a list of log lines using a specified extractor function and data class.

    Args:
    log_lines (list[str]): The list of log lines.
    extractor (Callable[[list[str], int, str], tuple[T, int]]): The extractor function to be used.
    dataclass (Type[T]): The data class for the results.

    Returns:
    List[T]: A list of results as instances of the specified data class.
    """
    results = []
    dataset_path = None
    begin_index = 0

    while begin_index < len(log_lines):
        if dataset_path is None:
            dataset_path = __extract_dataset_path(log_lines[begin_index])
            begin_index += 1
        else:
            try:
                result, next_index = extractor(log_lines, begin_index, dataset_path)
                results.append(result)
                begin_index = next_index
            except AssertionError as e:
                begin_index += 1
            except ValueError as e:
                begin_index += 1

    return results