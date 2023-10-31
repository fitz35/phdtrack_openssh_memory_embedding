import argparse
import os
import sys



sys.path.append(os.path.abspath('../..'))
from annexe_generation.log_analyser.clustering_analyser.extractor import clustering_extractor
from annexe_generation.log_analyser.common_extractor import extract_all_dataset_results
from annexe_generation.log_analyser.random_forest_analyser.extractor import random_forest_extractor
from annexe_generation.log_analyser.random_forest_analyser.classifier_data import ClassificationResults, get_best_instances, plot_metrics, save_classification_results_to_json
from annexe_generation.log_analyser.clustering_analyser.clustering_data import ClusteringResult, clustering_pie_charts, save_clustering_results_to_json

def compare_list_of_dicts(list1: list[dict[str, str]], list2: list[dict[str, str]]) -> bool:
    """
    Compare if two lists of dictionaries contain the same values.

    Args:
    list1 (list[dict[str, str]]): The first list of dictionaries.
    list2 (list[dict[str, str]]): The second list of dictionaries.

    Returns:
    bool: True if both lists contain the same dictionaries, False otherwise.
    """
    return all(dict_item in list2 for dict_item in list1) and len(list1) == len(list2)

def read_file(file_path: str):
    try:
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file]
            return lines
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

def list_of_dicts_to_latex_table(list_of_dicts: list[dict[str, str]], caption: str = "Table caption", label: str = "tab:mytable") -> str:
    """
    Convert a list of dictionaries to a LaTeX tabular environment.

    Args:
    list_of_dicts (list[dict[str, str]]): List of dictionaries to convert.
    caption (str, optional): Caption for the LaTeX table. Defaults to "Table caption".
    label (str, optional): Label for the LaTeX table. Defaults to "tab:mytable".

    Returns:
    str: LaTeX code for the table.
    """
    if not list_of_dicts:
        return "The list is empty."

    # Extracting the column names from the keys of the first dictionary
    columns = list(list_of_dicts[0].keys())
    num_columns = len(columns)

    # Creating the LaTeX tabular environment
    latex_code = "\\begin{table}[h]\n"
    latex_code += "\\centering\n"
    latex_code += "\\begin{tabular}{" + "l" * num_columns + "}\n"
    latex_code += "\\hline\n"

    # Adding the column headers
    latex_code += " & ".join(columns) + " \\\\ \n"
    latex_code += "\\hline\n"

    # Adding the rows
    for item in list_of_dicts:
        row = " & ".join(item.values()) + " \\\\"
        latex_code += row + "\n"

    latex_code += "\\hline\n"
    latex_code += "\\end{tabular}\n"
    latex_code += f"\\caption{{{caption}}}\n"
    latex_code += f"\\label{{{label}}}\n"
    latex_code += "\\end{table}"

    return latex_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a file and process its contents.')
    parser.add_argument('files_dir_path', type=str, help='Path to the directory to be read (get all results log files)')
    parser.add_argument('output', type=str, help='Path to the output directory')
    args = parser.parse_args()

    # ------------------------- Reset the output
    if os.path.exists(args.output):
        os.system(f"rm -r {args.output}")
    os.mkdir(args.output)



    clustering_timeouts : list[dict[str, str]] = []
    classification_timeouts : list[dict[str, str]] = []

    clustering_results : list[ClusteringResult] = []
    classification_results : list[ClassificationResults] = []

    clustering_results_by_dataset: dict[str, list[ClusteringResult]] = {}
    classification_results_by_dataset : dict[str, list[ClassificationResults]] = {}

    # ------------------------- Read the files and extract data
    # Get all files in the directory
    files = [os.path.join(args.files_dir_path, file) for file in os.listdir(args.files_dir_path) if file.endswith(".log") and not file.startswith("common_log")]

    # Read all files
    for file in files:
        lines: list[str] = read_file(file)
        clustering, clustering_timeout = extract_all_dataset_results(lines, clustering_extractor)
        classification, classification_timeout = extract_all_dataset_results(lines, random_forest_extractor)

        clustering_results.extend(clustering)
        classification_results.extend(classification)

        clustering_timeouts.extend(clustering_timeout)
        classification_timeouts.extend(classification_timeout)

    assert compare_list_of_dicts(clustering_timeouts, classification_timeouts), "Clustering and classification timeouts are not the same"
    # extract the instance by dataset
    # Organize clustering and classification results by dataset
    for result in clustering_results:
        dataset_name = result.dataset_name
        if dataset_name not in clustering_results_by_dataset:
            clustering_results_by_dataset[dataset_name] = []
        clustering_results_by_dataset[dataset_name].append(result)

    for result in classification_results:
        dataset_name = result.dataset_name
        if dataset_name not in classification_results_by_dataset:
            classification_results_by_dataset[dataset_name] = []
        classification_results_by_dataset[dataset_name].append(result)

    # ------------------------- Extract the clustering results

    # make the list of timeout instances as latex

    
    for dataset_name, results in clustering_results_by_dataset.items():
        dataset_path = os.path.join(args.output, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)

        # save latex
        clustering_latex_file_path = os.path.join(dataset_path, f"clustering_results.txt")
        with open(clustering_latex_file_path, 'w') as f:
            f.write("")
        for result in results:
            with open(clustering_latex_file_path, 'a') as f:
                f.write(result.to_latex() + "\n\n")

        clustering_pie_folder_path = os.path.join(dataset_path, "clustering_pie_charts")
        os.makedirs(clustering_pie_folder_path, exist_ok=True)
        clustering_pie_charts(results, clustering_pie_folder_path)


        save_clustering_results_to_json(results, os.path.join(dataset_path, "clustering_results.json"))

    # ----------------------- Extract classification results

    for dataset_name, results in classification_results_by_dataset.items():
        dataset_path = os.path.join(args.output, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)

        # save latex
        classification_latex_file_path = os.path.join(dataset_path, f"classification_results.txt")

        with open(classification_latex_file_path, 'w') as f:
            f.write("")
        for result in results:
            with open(classification_latex_file_path, 'a') as f:
                f.write(result.to_latex() + "\n\n")


        plot_metrics(results, dataset_path, f"{dataset_name} - Metrics")

        save_classification_results_to_json(results, os.path.join(dataset_path, "classification_results.json")  )
    

    plot_metrics(get_best_instances(classification_results_by_dataset, "accuracy"), args.output, "Best Accuracy")