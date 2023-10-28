


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