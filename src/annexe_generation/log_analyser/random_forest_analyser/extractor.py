

import json
import re

def extract_dataset_path(log_line: str):
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
    match = re.search(r"Launching embedding pipeline on dataset (\S+)", log_line)
    if not match:
        return None
    return match.group(1)


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
    match = re.search(r"INFO - !+  (\S+) instance : ", log_line)
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
    list[str]: The extracted lines between the start and end delimiters.
    """
    inside_section = False
    extracted_lines = []
    
    for line in log_lines[start_index:]:
        if start_delimiter in line:
            inside_section = True
        elif end_delimiter in line:
            inside_section = False
            break
        elif inside_section:
            extracted_lines.append(line)
            
    return extracted_lines


def extract_random_forest_lines(log_lines: list[str], start_index: int):
    """
    A convenience function to extract log lines related to the random forest operation.
    
    This function is a wrapper around the extract_lines_between function.
    It specifies the start and end delimiters related to the random forest operation logs.
    
    Args:
    log_lines (list[str]): The list of log lines.
    start_index (int): The index to start searching from.
    
    Returns:
    list[str]: The extracted random forest related log lines.
    """
    start_delimiter = "///---!!!! Launching testing pipeline on dataset"
    end_delimiter = "Time elapsed since the begining of random forest"
    return extract_lines_between(log_lines, start_index, start_delimiter, end_delimiter)


def extract_and_decode_json(log_lines: list[str]) -> dict:
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
