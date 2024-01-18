from datetime import datetime, timedelta
import os
import re


EMBEDDING_RESULTS_DIR_PATH = 'results_serv/archive/combining/embedding_test/'
DEEP_LEARNING_PATH = 'results_serv/archive/deeplearning/nohup.out'


def list_embedding_result_log(folder_path : str) -> list[str]:
    # Initialize an empty list to store matching file names
    pattern = r'^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{6}_results\.log$'

    matching_files = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the filename matches the specified pattern
        if re.match(pattern, filename):
            matching_files.append(filename)

    return matching_files


def extract_timestamp_and_validate(line : str) -> tuple[bool, datetime | None]:
    # Define the pattern to match
    pattern = r'^(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}) - results_logger - INFO -'

    # Use re.match to check if the line matches the pattern
    match = re.match(pattern, line)
    
    if match:
        timestamp = match.group(1)  # Extract the timestamp
        return True, datetime.strptime(timestamp, "%Y_%m_%d_%H_%M_%S")
    else:
        return False, None
    
def get_duration_from_log_file(file_path : str) -> timedelta : 
    file_content = open(file_path, 'r').readlines()
    first_log_timestamp : datetime | None = None

    for line in file_content:
        is_matched, timestamp = extract_timestamp_and_validate(line)
        if is_matched:
            assert timestamp is not None
            first_log_timestamp = timestamp
            break

    if first_log_timestamp is None:
        raise ValueError("No timestamp found in the log file.")


    last_log_timestamp : datetime | None = None

    for line in reversed(file_content):
        is_matched, timestamp = extract_timestamp_and_validate(line)
        if is_matched:
            assert timestamp is not None
            last_log_timestamp = timestamp
            break
    
    if last_log_timestamp is None:
        raise ValueError("No timestamp found in the log file.")

    return last_log_timestamp - first_log_timestamp


if __name__ == '__main__':
    embedding_result_log_files = list_embedding_result_log(EMBEDDING_RESULTS_DIR_PATH)
    embedding_result_log_files.sort()

    aggregate_duration = timedelta()

    for file in embedding_result_log_files:
        aggregate_duration +=  get_duration_from_log_file(EMBEDDING_RESULTS_DIR_PATH + file)

    print("Embedding: ", aggregate_duration)

    print("Deep learning: ", get_duration_from_log_file(DEEP_LEARNING_PATH))