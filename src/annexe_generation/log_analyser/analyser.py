import argparse
import os
import sys



sys.path.append(os.path.abspath('../..'))
from annexe_generation.log_analyser.clustering_analyser.extractor import clustering_extractor
from annexe_generation.log_analyser.common_extractor import extract_all_dataset_results
from annexe_generation.log_analyser.random_forest_analyser.extractor import random_forest_extractor
from annexe_generation.log_analyser.random_forest_analyser.classifier_data import ClassificationResults, plot_metrics
from annexe_generation.log_analyser.clustering_analyser.clustering_data import clustering_pie_charts

def read_file(file_path: str):
    try:
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file]
            return lines
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a file and process its contents.')
    parser.add_argument('file_path', type=str, help='Path to the file to be read')
    parser.add_argument('output', type=str, help='Path to the output directory')
    args = parser.parse_args()
    
    lines: list[str] = read_file(args.file_path)

    # ------------------------- Extract the clustering results
    clustering_results = extract_all_dataset_results(lines, clustering_extractor)

    # print and save clustering results to LaTeX file
    dataset_name = None
    clustering_latex_file_path = None

    for result in clustering_results:
        if dataset_name != result.dataset_name:
            dataset_name = result.dataset_name
            clustering_latex_file_path = os.path.join(args.output, f"{dataset_name}_clustering_results.txt")
            with open(clustering_latex_file_path, 'w') as f:
                f.write("")

        if clustering_latex_file_path is not None:
            with open(clustering_latex_file_path, 'a') as f:
                f.write(result.to_latex() + "\n\n")
    clustering_pie_charts(clustering_results, args.output)
    # ----------------------- Extract classification results
    classification_results : list[ClassificationResults] =  extract_all_dataset_results(lines, random_forest_extractor)

    # Print and save classification results to LaTeX file
    dataset_name = None
    classification_latex_file_path = None

    for result in classification_results:
        if dataset_name != result.dataset_name:
            dataset_name = result.dataset_name
            classification_latex_file_path = os.path.join(args.output, f"{dataset_name}_classification_results.txt")
            with open(classification_latex_file_path, 'w') as f:
                f.write("")

        if classification_latex_file_path is not None:
            with open(classification_latex_file_path, 'a') as f:
                f.write(result.to_latex() + "\n\n")

    plot_metrics(classification_results, args.output)