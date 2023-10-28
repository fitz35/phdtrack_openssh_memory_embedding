

import ast
import datetime
import os
import re
import sys
from typing import List, Tuple

from annexe_generation.log_analyser.clustering_analyser.clustering_data import ClusterInfo, ClusteringResult, LabelAssociation


sys.path.append(os.path.abspath('../../..'))
from annexe_generation.log_analyser.common_extractor import extract_instance_name, extract_instance_number, extract_lines_between


def __extract_clustering_lines(log_lines: List[str], start_index: int) -> Tuple[List[str], int]:
    """
    A convenience function to extract log lines related to the clustering operation.
    
    This function is a wrapper around the extract_lines_between function.
    It specifies the start and end delimiters related to the clustering operation logs.
    
    Args:
    log_lines (List[str]): The list of log lines.
    start_index (int): The index to start searching from.
    
    Returns:
    Tuple[List[str], int]: The extracted clustering related log lines (including the start and end delimiters.)
                            And the index of the line containing the start delimiter.
    """
    start_delimiter = "- results_logger - INFO - timer for clustering started"
    end_delimiter = "- results_logger - INFO - Time elapsed since the begining of clustering:"
    return extract_lines_between(log_lines, start_index, start_delimiter, end_delimiter)


def __extract_samples(log_lines: List[str]) -> Tuple[dict[float, int] | None, dict[float, int] | None]:
    initial_samples = None
    final_samples = None

    for line in log_lines:
        if "Number of samples before rebalancing and limiting rows" in line:
            initial_samples = {float(k): int(v) for k, v in re.findall(r"class-(\d+\.\d+)=(\d+)", line)}
        elif "Number of samples after rebalancing and limiting rows" in line:
            final_samples = {float(k): int(v) for k, v in re.findall(r"class-(\d+\.\d+)=(\d+)", line)}

    return initial_samples, final_samples

def __extract_cluster_info(log_lines: List[str]) -> List[ClusterInfo]:
    """
    Extract clustering information from log lines.

    Args:
    log_lines (List[str]): A list of log lines as strings.

    Returns:
    List[ClusterInfo]: A list of ClusterInfo dataclass instances containing extracted information.
    """
    # Initialize a list to store the ClusterInfo instances
    cluster_infos = []
    # Initialize a variable to store the current duration of clustering
    current_duration = 0.0

    for line in log_lines:
        # Check for the line that contains the duration of clustering and extract it
        duration_match = re.match(r".*Time elapsed since the begining of clustering_duration_for_(\d+\.\d+): (\d+\.\d+) s", line)
        if duration_match:
            # Update the current duration with the extracted value
            current_duration = float(duration_match.group(2))
            continue

        # Check for the line that contains cluster information and extract it
        info_match = re.match(r".*eps: (\d+\.\d+), number of clusters: (\d+), silhouette score: ([\d.-]+), noise points: (\d+)", line)
        if info_match:
            # Extract information from the log line
            eps = float(info_match.group(1))
            number_of_clusters = int(info_match.group(2))
            silhouette_score = float(info_match.group(3))
            noise_points = int(info_match.group(4))
            # Create a ClusterInfo instance with the extracted information and the current duration
            cluster_info = ClusterInfo(eps, number_of_clusters, silhouette_score, noise_points, current_duration)
            # Add the ClusterInfo instance to the list
            cluster_infos.append(cluster_info)

    # Return the list of ClusterInfo instances
    return cluster_infos




def __extract_label_association(log_lines: List[str]) -> List[LabelAssociation]:
    """
    Extract label associations from log lines.

    Args:
    log_lines (List[str]): A list of log lines as strings.

    Returns:
    Set[LabelAssociation]: A set of LabelAssociation dataclass instances containing extracted information.

    Raises:
    ValueError: If no label association information is found in the log lines.
    """
    label_associations = []
    next_line_contains_data = False

    for line in log_lines:
        if next_line_contains_data:
            association_dict = ast.literal_eval(line.strip())
            for cluster_id, label_counts in association_dict.items():
                label_association = LabelAssociation(
                    float(cluster_id), 
                    {float(label): count for label, count in label_counts.items()}
                )
                label_associations.append(label_association)
            next_line_contains_data = False
        elif "Associating clusters to labels :" in line:
            next_line_contains_data = True

    if not label_associations:
        raise ValueError("No label association information found in the log lines.")

    return label_associations

def __extract_best_eps(log_text: str) -> float | None:
    """
    Extract the best eps value from a log text.

    Args:
    log_text (str): Log text as a single string.

    Returns:
    float: The best eps value.

    Raises:
    ValueError: If the best eps value is not found in the log text.
    """
    match = re.search(r"Best eps: (\d+\.\d+)", log_text)
    if match:
        return float(match.group(1))
    else:
        return None
    

def __extract_time(log_line: str) -> float | None:
    """
    Extract the time duration from a log line.

    Args:
    log_line (str): A single string of log line.

    Returns:
    float: The extracted time duration.

    Raises:
    ValueError: If the time duration is not found in the log line.
    """
    match = re.search(r"Time elapsed since the begining of clustering: (\d+\.\d+) s", log_line)
    if match:
        return float(match.group(1))
    else:
        return None
    

def __extract_min_samples(log_line: str) -> int | None:
    """
    Extract the min_samples value from a log line.

    Args:
    log_line (str): A single string of log line.

    Returns:
    int: The extracted min_samples value.

    Raises:
    ValueError: If the min_samples value is not found in the log line.
    """
    match = re.search(r"min_samples: (\d+)", log_line)
    if match:
        return int(match.group(1))
    else:
        return None

def clustering_extractor(all_lines : list[str], begin_index : int, dataset_path : str) -> Tuple[ClusteringResult, int] :
    dataset_name = os.path.basename(dataset_path)
    # get the random forest lines
    clustering_lines, clustering_index = __extract_clustering_lines(all_lines, begin_index)
    #print(random_forest_start_index)

    # iterate through the line preceding the random forest lines to get the dataset path and instance name
    
    instance_name = None
    instance_number = None

    for i in range(begin_index, clustering_index, 1):
        if instance_name is None:
            instance_name = extract_instance_name(all_lines[i])
        if instance_number is None:
            instance_number = extract_instance_number(all_lines[i])
    
    assert instance_name is not None, "Instance name not found"
    assert instance_number is not None, "Instance number not found"

    instance_name = instance_name + " " + str(instance_number)

    # get the initial and final samples
    initial_samples, final_samples = __extract_samples(clustering_lines)

    assert initial_samples is not None, "Initial samples not found"
    assert final_samples is not None, "Final samples not found"

    # get the cluster info
    cluster_info = __extract_cluster_info(clustering_lines)

    # get the label association
    label_association = __extract_label_association(clustering_lines)


    best_eps = None
    min_samples = None
    total_duration = None
    for line in clustering_lines:
        if best_eps is None:
            best_eps = __extract_best_eps(line)
        if total_duration is None:
            total_duration = __extract_time(line)
        if min_samples is None:
            min_samples = __extract_min_samples(line)

    assert best_eps is not None, "Best eps not found"
    assert total_duration is not None, "Total duration not found"
    assert min_samples is not None, "Min samples not found"

    best_eps_info = next((info for info in cluster_info if info.eps == best_eps), None)

    # Raise an error if the best_eps is not found in the cluster_info list
    if best_eps_info is None:
        raise ValueError(f"Best eps value {best_eps} not found in clustering info.")


    clustering_result = ClusteringResult(
        dataset_name=dataset_name, 
        instance=instance_name, 
        initial_samples=initial_samples, 
        final_samples=final_samples, 
        min_samples=min_samples, 
        clustering_info=cluster_info, 
        best_eps_info=best_eps_info, 
        label_association=label_association, 
        total_duration=total_duration
    )

    return clustering_result, clustering_index + len(clustering_lines) + 1
