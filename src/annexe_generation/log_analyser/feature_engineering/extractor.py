

import os
import sys


sys.path.append(os.path.abspath('../../..'))
from annexe_generation.log_analyser.common_extractor import extract_lines_between


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

