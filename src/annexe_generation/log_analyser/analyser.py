import argparse
import os
import shutil
import sys
from typing import Any, Dict, List, Set







sys.path.append(os.path.abspath('../..'))
from annexe_generation.log_analyser.clustering_analyser.extractor import clustering_extractor
from annexe_generation.log_analyser.common_extractor import extract_all_dataset_results, extract_dataset_path
from annexe_generation.log_analyser.random_forest_analyser.extractor import random_forest_extractor
from annexe_generation.log_analyser.random_forest_analyser.classifier_data import ClassificationResults, get_best_instances, plot_metrics, save_classification_results_to_json
from annexe_generation.log_analyser.clustering_analyser.clustering_data import ClusteringResult, save_clustering_results_to_json
from annexe_generation.log_analyser.feature_engineering.feature_engineering_data import FeatureEngineeringData, features_engineering_list_to_json
from annexe_generation.log_analyser.feature_engineering.extractor import feature_engineering_extractor
from annexe_generation.log_analyser.dataset_data.dataset_data import DatasetData, datasets_to_latex_longtable
from embedding_generation.data.hyperparams_transformers import get_transformers_hyperparams
from embedding_generation.data.hyperparams_word2vec import get_word2vec_hyperparams_instances

LOG_DIR_NAME = "embedding_test"
FEATURE_ENGINEERING_DIR_NAME = "feature_correlation_matrices"
NB_FEATURE_ENGINEERING_FEATURES = 8 # Number of features to keep for the feature engineering results (if different, put the instance in the feature engineering list)



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
    columns.sort()
    num_columns = len(columns)

    # sort the list by the dataset name
    list_of_dicts.sort(key=lambda x: x["dataset"])

    # Creating the LaTeX tabular environment
    latex_code = "\\begin{table}[ht]\n"
    latex_code += "\\centering\n"
    latex_code += "\\begin{tabular}{" + "l" * num_columns + "}\n"
    latex_code += "\\hline\n"

    # Adding the column headers
    latex_code += " & ".join(columns) + " \\\\ \n"
    latex_code += "\\hline\n"

    # Adding the rows
    for item in list_of_dicts:
        row = ""
        for key in columns:
            if key in item:
                if key == "dataset":
                    row += os.path.basename(item[key]).replace("_", "\\_") + " & "
                else:
                    row += item[key] + " & "
            else:
                row += " & "
        row = row[:-2] + "\\\\ \n"
        latex_code += row

    latex_code += "\\hline\n"
    latex_code += "\\end{tabular}\n"
    latex_code += f"\\caption{{{caption}}}\n"
    latex_code += f"\\label{{{label}}}\n"
    latex_code += "\\end{table}"

    return latex_code

def image_real_path_to_latex_path(image_real_path : str) -> str :

    image_split_path = image_real_path.split("img/")
    return os.path.join("img/", image_split_path[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a file and process its contents.')
    parser.add_argument('files_dir_path', type=str, help='Path to the directory to be read (get all results log files)')
    parser.add_argument('output', type=str, help='Path to the output directory')
    args = parser.parse_args()

    log_dir_path = os.path.join(args.files_dir_path, LOG_DIR_NAME)
    feature_engineering_dir_path = os.path.join(args.files_dir_path, FEATURE_ENGINEERING_DIR_NAME)

    LATEX_FILE_NAME = "latex.txt"
    IMG_FOLDER_NAME = "img/annexes"

    img_folder_path = os.path.join(args.output, IMG_FOLDER_NAME)

    # ------------------------- Reset the output
    if os.path.exists(args.output):
        os.system(f"rm -r {args.output}")
    os.mkdir(args.output)

    dataset_informations_set : set[DatasetData] = set()

    feature_engineering_fails : list[dict[str, str]] = []

    clustering_timeouts : list[dict[str, str]] = []
    classification_timeouts : list[dict[str, str]] = []
    feature_engineering_timeouts : list[dict[str, str]] = []

    clustering_results : list[ClusteringResult] = []
    classification_results : list[ClassificationResults] = []
    feature_engineering_results : list[FeatureEngineeringData] = []

    clustering_results_by_dataset_number: dict[str, list[ClusteringResult]] = {}
    classification_results_by_dataset_number : dict[str, list[ClassificationResults]] = {}
    feature_engineering_results_by_dataset_number : dict[str, list[FeatureEngineeringData]] = {}

    clustering_results_by_dataset: dict[str, list[ClusteringResult]] = {}
    classification_results_by_dataset : dict[str, list[ClassificationResults]] = {}
    feature_engineering_results_by_dataset : dict[str, list[FeatureEngineeringData]] = {}

    # --------------------------- hyper params

    transformers_instances_index = ["Transformers " + str(x.index) for x in get_transformers_hyperparams()]
    word2vec_instances = ["Word2vec " + str(x.index) for x in get_word2vec_hyperparams_instances()]

    hyperparams_instances = transformers_instances_index + word2vec_instances

    # ------------------------- Read the files and extract data
    # Get all files in the directory
    files = [os.path.join(args.files_dir_path, LOG_DIR_NAME, file) for file in os.listdir(log_dir_path) if file.endswith(".log") and not file.startswith("common_log")]
    #files = ["/home/clement/Documents/github/phdtrack_openssh_memory_embedding/results_serv/archive/deeplearning/embedding_test/2023_10_29_08_42_19_410766_results.log"]
    print(f"Found {len(files)} log files")
    for file in files:
        print(file)
    print("\n")
    # Read all files
    for file in files:
        lines: list[str] = read_file(file)

        feature_engineering, feature_engineering_timeout = extract_all_dataset_results(lines, feature_engineering_extractor, feature_engineering_dir_path)
        feature_engineering_results.extend(feature_engineering)
        feature_engineering_timeouts.extend(feature_engineering_timeout)
        

        clustering, clustering_timeout = extract_all_dataset_results(lines, clustering_extractor, feature_engineering_dir_path)
        classification, classification_timeout = extract_all_dataset_results(lines, random_forest_extractor, feature_engineering_dir_path)

        clustering_results.extend(clustering)
        classification_results.extend(classification)
        
        clustering_timeouts.extend(clustering_timeout)
        classification_timeouts.extend(classification_timeout)

        dataset_path = extract_dataset_path(lines)
        assert dataset_path is not None, "Could not extract the dataset path"

        dataset_informations_set.add(DatasetData.from_str(os.path.basename(dataset_path)))
    


    assert compare_list_of_dicts(clustering_timeouts, classification_timeouts), "Clustering and classification timeouts are not the same"
    assert compare_list_of_dicts(clustering_timeouts, feature_engineering_timeouts), "Clustering and feature engineering timeouts are not the same"
    
    dataset_informations = list(dataset_informations_set)
    dataset_informations.sort(key=lambda x: x.dataset_number)
    print(f"Found {len(dataset_informations)} datasets")
    for dataset_information in dataset_informations:
        print(dataset_information.get_display_name())
    print("\n")


    def get_dataset_information_from_number(number: int) -> DatasetData:
        for dataset_information in dataset_informations:
            if dataset_information.dataset_number == number:
                return dataset_information
        raise Exception(f"Could not find dataset information for dataset number {number}")

    for feature_engineering_instance in feature_engineering_results:
        if len(feature_engineering_instance.best_columns) != NB_FEATURE_ENGINEERING_FEATURES:
            feature_engineering_fails.append({"dataset": str(feature_engineering_instance.dataset_name.dataset_number), "instance": feature_engineering_instance.instance, "nb_features": str(len(feature_engineering_instance.best_columns))})


    # extract the instance by dataset
    # Organize clustering and classification results by dataset
    for dataset in dataset_informations:
        dataset_name = str(dataset.dataset_number)
        if dataset_name not in clustering_results_by_dataset_number:
            clustering_results_by_dataset_number[dataset_name] = []
        if dataset_name not in classification_results_by_dataset_number:
            classification_results_by_dataset_number[dataset_name] = []
        if dataset_name not in feature_engineering_results_by_dataset_number:
            feature_engineering_results_by_dataset_number[dataset_name] = []

    for result in clustering_results:
        dataset_name = str(result.dataset_name.dataset_number)
        if dataset_name not in clustering_results_by_dataset_number:
            clustering_results_by_dataset_number[dataset_name] = []
        clustering_results_by_dataset_number[dataset_name].append(result)

    for result in classification_results:
        dataset_name = str(result.dataset_name.dataset_number)
        if dataset_name not in classification_results_by_dataset_number:
            classification_results_by_dataset_number[dataset_name] = []
        classification_results_by_dataset_number[dataset_name].append(result)

    for result in feature_engineering_results:
        dataset_name = str(result.dataset_name.dataset_number)
        if dataset_name not in feature_engineering_results_by_dataset_number:
            feature_engineering_results_by_dataset_number[dataset_name] = []
        feature_engineering_results_by_dataset_number[dataset_name].append(result)



    for dataset in dataset_informations:
        dataset_name = dataset.dataset_name
        if dataset_name not in clustering_results_by_dataset:
            clustering_results_by_dataset[dataset_name] = []
        if dataset_name not in classification_results_by_dataset:
            classification_results_by_dataset[dataset_name] = []
        if dataset_name not in feature_engineering_results_by_dataset:
            feature_engineering_results_by_dataset[dataset_name] = []

    for result in clustering_results:
        dataset_name = result.dataset_name.dataset_name
        if dataset_name not in clustering_results_by_dataset:
            clustering_results_by_dataset[dataset_name] = []
        clustering_results_by_dataset[dataset_name].append(result)

    for result in classification_results:
        dataset_name = result.dataset_name.dataset_name
        if dataset_name not in classification_results_by_dataset:
            classification_results_by_dataset[dataset_name] = []
        classification_results_by_dataset[dataset_name].append(result)

    for result in feature_engineering_results:
        dataset_name = result.dataset_name.dataset_name
        if dataset_name not in feature_engineering_results_by_dataset:
            feature_engineering_results_by_dataset[dataset_name] = []
        feature_engineering_results_by_dataset[dataset_name].append(result)

    best_instance_by_instance_accuracy = get_best_instances(classification_results_by_dataset_number, "accuracy")
    best_instance_by_dataset_accuracy = get_best_instances(classification_results_by_dataset, "accuracy")

    best_instance_by_instance_accuracy_name = [(result.dataset_name, result.instance) for result in best_instance_by_instance_accuracy]
    best_instance_by_dataset_accuracy_name = [(result.dataset_name, result.instance) for result in best_instance_by_dataset_accuracy]
    # ------------------------- prepare files and folder
    # create the folder for the images
    os.makedirs(img_folder_path)


    all_dataset_names = set(
        list(clustering_results_by_dataset_number.keys()) +
        list(classification_results_by_dataset_number.keys()) +
        list(feature_engineering_results_by_dataset_number.keys())
    )
    FEATURE_ENGINEERING_LATEX_FILE_NAME = "feature_engineering_results.txt"
    CLUSTERING_LATEX_FILE_NAME = "clustering_results.txt"
    CLASSIFICATION_LATEX_FILE_NAME = "classification_results.txt"

    for dataset_name in all_dataset_names:
        dataset_path = os.path.join(args.output, dataset_name)
        img_dataset_path = os.path.join(img_folder_path, dataset_name)
        os.makedirs(dataset_path)
        os.makedirs(img_dataset_path)
    # ------------------------- Extract the feature engineering results

    for dataset_name in all_dataset_names:
        dataset_path = os.path.join(args.output, dataset_name)
        latex_file_path = os.path.join(dataset_path, FEATURE_ENGINEERING_LATEX_FILE_NAME)
    
        with open(latex_file_path, 'a') as f:
            f.write("")

    for dataset_name, results in feature_engineering_results_by_dataset_number.items():
        dataset_path = os.path.join(args.output, dataset_name)
        img_dataset_path = os.path.join(img_folder_path, dataset_name)
        
        latex_file_path = os.path.join(dataset_path, FEATURE_ENGINEERING_LATEX_FILE_NAME)

        features_engineering_list_to_json(results, os.path.join(dataset_path, "features_engineering_list.json"))

        # save latex
        for result in results:
            correlation_matrix_path = os.path.join(img_dataset_path, result.instance + "_correlation_matrix.png")

            if (result.dataset_name, result.instance) in best_instance_by_instance_accuracy_name:
                with open(latex_file_path, 'a') as f:
                    f.write(result.to_latex(image_real_path_to_latex_path(correlation_matrix_path)) + "\n\n")
                    #f.write(result.correlation_matrix_to_latex() + "\n\n")
        
            # save the correlation matrix
            result.save_correlation_matrix_as_heatmap(correlation_matrix_path)
    # ------------------------- Extract the clustering results

    for dataset_name in all_dataset_names:
        dataset_path = os.path.join(args.output, dataset_name)
        latex_file_path = os.path.join(dataset_path, CLUSTERING_LATEX_FILE_NAME)
    
        with open(latex_file_path, 'a') as f:
            f.write("")

    # treat the data
    for dataset_name, results in clustering_results_by_dataset_number.items():
        dataset_path = os.path.join(args.output, dataset_name)
        img_dataset_path = os.path.join(img_folder_path, dataset_name)

        # save latex
        clustering_latex_file_path = os.path.join(dataset_path, CLUSTERING_LATEX_FILE_NAME)
   
        for result in results:
            # save the result only if the clustering was successful
            if len(result.label_association) == 0:
                continue

            clustering_pie_folder_path = os.path.join(img_dataset_path, "clustering_pie_charts")
            os.makedirs(clustering_pie_folder_path, exist_ok=True)
            clustering_image_path = os.path.join(clustering_pie_folder_path, f'{result.instance}.png')

            with open(clustering_latex_file_path, 'a') as f:
                f.write(result.to_latex(image_real_path_to_latex_path((clustering_image_path))) + "\n\n")
            
            # save the pie charts
            result.save_pie_charts((clustering_image_path))

        save_clustering_results_to_json(results, os.path.join(dataset_path, "clustering_results.json"))

    # ----------------------- Extract classification results

    
    for dataset_name in all_dataset_names:
        dataset_path = os.path.join(args.output, dataset_name)
        latex_file_path = os.path.join(dataset_path, CLASSIFICATION_LATEX_FILE_NAME)
    
        with open(latex_file_path, 'a') as f:
            f.write("")

    for dataset_name, results in classification_results_by_dataset_number.items():

        dataset_path = os.path.join(args.output, dataset_name)
        img_dataset_path = os.path.join(img_folder_path, dataset_name)
        # save latex
        classification_latex_file_path = os.path.join(dataset_path, CLASSIFICATION_LATEX_FILE_NAME)

        image_file_path = os.path.join(img_dataset_path, f'{dataset_name} - Metrics.png')

        if len(results) > 0:
            with open(classification_latex_file_path, 'a') as f:
                f.write("\\begin{figure}[H]\n")
                f.write("\\centering\n")
                f.write("\\includegraphics[width=0.6\\textwidth]{" + image_real_path_to_latex_path(image_file_path) + "}\n")
                f.write("\\caption{Metrics for the instances of the dataset" + dataset_name + "}\n")
                f.write("\\label{fig:" + dataset_name + "_metrics_instance}\n")
                f.write("\\end{figure}\n\n")

        for result in results:
             # treat only the best instance by instance accuracy
            if (result.dataset_name, result.instance) in best_instance_by_instance_accuracy_name:
                    
                with open(classification_latex_file_path, 'a') as f:
                    f.write(result.to_latex() + "\n\n")

        if len(results) > 0:
            plot_metrics(results, image_file_path)

        save_classification_results_to_json(results, os.path.join(dataset_path, "classification_results.json")  )
    
    best_accuracy_by_instance_file_path = os.path.join(img_folder_path, "Best Accuracy (by instances).png")
    best_accuracy_by_dataset_file_path = os.path.join(img_folder_path, "Best Accuracy (by dataset).png")
    plot_metrics(best_instance_by_instance_accuracy, best_accuracy_by_instance_file_path)
    plot_metrics(best_instance_by_dataset_accuracy, best_accuracy_by_dataset_file_path)




    # ------------------------- fusionne the latex file -------------------------
    latex_file_path = os.path.join(args.output, LATEX_FILE_NAME)

    with open(latex_file_path, 'w') as f:
        f.write("\\chapter{Machin Learning Results}\n")

    # datasets informations table
    with open(latex_file_path, 'a') as f:
        f.write("\\section{Datasets informations}\n\n")
        f.write(datasets_to_latex_longtable(dataset_informations))
        f.write("\n\n")

    # timeout instances
    with open(latex_file_path, 'a') as f:
        f.write("\\section{Timeout instances}\n\n")
        f.write("\\label{sec:annexe:timeout_instances}\n\n")
        f.write(list_of_dicts_to_latex_table(clustering_timeouts, "Timeouts instances", "tab:timeouts"))
        f.write("\n\n")

    # feature engineering fails
    with open(latex_file_path, 'a') as f:
        f.write("\\section{Feature engineering fails}\n\n")
        f.write("\\label{sec:annexe:feature_engineering_fails}\n\n")
        f.write(list_of_dicts_to_latex_table(feature_engineering_fails, "Feature engineering fails", "tab:feature_engineering_fails"))
        f.write("\n\n")
    
    # Out of Memory instances (all instances who aren't in the timeout instances, the feature engineering fails and the right instances)
    list_of_all_instances_by_dataset_number_commun : dict[int, set[str]] = {}
    for value in clustering_timeouts:
        dataset_number = int(value["dataset"])
        if dataset_number not in list_of_all_instances_by_dataset_number_commun:
            list_of_all_instances_by_dataset_number_commun[dataset_number] = set()
        list_of_all_instances_by_dataset_number_commun[dataset_number].add(value["instance"])
    
    for value in feature_engineering_fails:
        dataset_number = int(value["dataset"])
        if dataset_number not in list_of_all_instances_by_dataset_number_commun:
            list_of_all_instances_by_dataset_number_commun[dataset_number] = set()
        list_of_all_instances_by_dataset_number_commun[dataset_number].add(value["instance"])
        
    def process_out_of_memory_instances(
        results_by_dataset_number: Dict[str, List[Any]], 
        list_of_all_instances_by_dataset_number_commun: Dict[int, Set[str]], 
        hyperparams_instances: List[str], 
        category: str, 
        latex_file_path: str
    ) -> None:
        instances_by_dataset_number: Dict[int, Set[str]] = list_of_all_instances_by_dataset_number_commun.copy()

        for dataset_number, values in results_by_dataset_number.items():
            instances_by_dataset_number.setdefault(int(dataset_number), set())
            instances_by_dataset_number[int(dataset_number)].update(value.instance for value in values)
        
        out_of_memory_instances: List[Dict[str, str]] = []
        for dataset_number, instances in instances_by_dataset_number.items():
            for instance_to_test in hyperparams_instances:
                if instance_to_test not in instances:
                    out_of_memory_instances.append({"dataset": get_dataset_information_from_number(dataset_number).get_display_name(), "instance": instance_to_test})

        with open(latex_file_path, 'a') as f:
            f.write(f"\\section{{Out of memory instances ({category})}}\n\n")
            f.write(f"\\label{{sec:annexe:out_of_memory_instances_{category.lower()}}}\n\n")
            f.write(list_of_dicts_to_latex_table(out_of_memory_instances, "Out of memory instances", f"tab:annexe:out_of_memory_instances_{category.lower()}"))
            f.write("\n\n")

    # classification
    # Example calls to the function:
    print(classification_results_by_dataset_number.keys())
    process_out_of_memory_instances(
        classification_results_by_dataset_number, 
        list_of_all_instances_by_dataset_number_commun, 
        hyperparams_instances, 
        'Classifications', 
        latex_file_path
    )

    process_out_of_memory_instances(
        clustering_results_by_dataset_number, 
        list_of_all_instances_by_dataset_number_commun, 
        hyperparams_instances, 
        'Clustering', 
        latex_file_path
    )
    # ...................... feature engineering
    with open(latex_file_path, 'a') as f:
        f.write("\\section{Feature Engineering results}\n\n")
        f.write("\\label{sec:annexe:feature_engineering_results}\n\n")


    for dataset_name in all_dataset_names:
        dataset_path = os.path.join(args.output, dataset_name)

        
        feature_engineering_latex_file_path = os.path.join(dataset_path, FEATURE_ENGINEERING_LATEX_FILE_NAME)

        with open(feature_engineering_latex_file_path, 'r') as feature_engineering_file:
            with open(latex_file_path, 'a') as f:
                f.write("\\subsection{" + get_dataset_information_from_number(int(dataset_name)).get_display_name() + "}\n\n")
                for line in feature_engineering_file:
                    f.write(line)

    # ..................... clustering
    with open(latex_file_path, 'a') as f:
        f.write("\\section{Clustering results}\n\n")
        f.write("\\label{sec:annexe:clustering_results}\n\n")
    
    for dataset_name in all_dataset_names:
        dataset_path = os.path.join(args.output, dataset_name)

        clustering_latex_file_path = os.path.join(dataset_path, CLUSTERING_LATEX_FILE_NAME)

        with open(clustering_latex_file_path, 'r') as clustering_file:
            with open(latex_file_path, 'a') as f:
                f.write("\\subsection{" + get_dataset_information_from_number(int(dataset_name)).get_display_name() + "}\n\n")
                for line in clustering_file:
                    f.write(line)
    

    # .......................... classification
    with open(latex_file_path, 'a') as f:
        f.write("\\section{Classification results}\n\n")
        f.write("\\label{sec:annexe:classification_results}\n\n")

    for dataset_name in all_dataset_names:
        dataset_path = os.path.join(args.output, dataset_name)

        classification_latex_file_path = os.path.join(dataset_path, CLASSIFICATION_LATEX_FILE_NAME)

        with open(classification_latex_file_path, 'r') as classification_file:
            with open(latex_file_path, 'a') as f:
                f.write("\\subsection{" + get_dataset_information_from_number(int(dataset_name)).get_display_name() + "}\n\n")
                for line in classification_file:
                    f.write(line)
            



    
       