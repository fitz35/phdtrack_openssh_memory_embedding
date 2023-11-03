from dataclasses import asdict, dataclass
import json
import os
import sys
from typing import Dict, List
import matplotlib.pyplot as plt

import numpy as np

sys.path.append(os.path.abspath('../../..'))
from annexe_generation.log_analyser.dataset_data.dataset_data import DatasetData

@dataclass(frozen=True)
class ClusterInfo:
    eps: float
    number_of_clusters: int
    silhouette_score: float | None
    noise_points: int | None
    duration: float


@dataclass(frozen=True)
class LabelAssociation:
    cluster_id: float
    label_counts: Dict[float, int]

@dataclass(frozen=True)
class ClusteringResult:
    dataset_name: DatasetData
    instance: str
    initial_samples: Dict[float, int]
    final_samples: Dict[float, int]
    min_samples: int
    clustering_info: List[ClusterInfo]
    best_eps_info: ClusterInfo | None
    label_association: List[LabelAssociation]
    total_duration: float


    def to_latex(self, cluster_image_path : str) -> str:
        # Start longtable
        if len(self.label_association) == 0:
            return "" # No array to display
        

        latex_str = "\\begin{longtable}{|c|c|c|c|c|}\n"
        latex_str += "\\caption{" + self.instance + " Clustering Results on " + str(self.dataset_name.dataset_number) + "} "
        latex_str += "\\label{tab:" + str(self.dataset_name.dataset_number) + "_" + self.instance.lower().replace(" ", "_") + "_clustering_results}\\\\\n"
        latex_str += "\\hline\n"
        
        # Part 1: General Information
        latex_str += "\\multicolumn{5}{|c|}{\\textbf{General Information}} \\\\\n"
        latex_str += "\\hline\n"
        latex_str += "\\multicolumn{2}{|c|}{Min Samples} & \\multicolumn{3}{c|}{" + str(self.min_samples) + "} \\\\\n"
        latex_str += "\\multicolumn{2}{|c|}{Total Duration} & \\multicolumn{3}{c|}{" + str(self.total_duration) + " s} \\\\\n"
        latex_str += "\\hline\n"
        
        # Part 2: Clustering Information
        latex_str += "\\multicolumn{5}{|c|}{\\textbf{Clustering Information}} \\\\\n"
        latex_str += "\\hline\n"
        latex_str += "EPS & Number of Clusters & Silhouette Score & Noise Points & Duration \\\\\n"
        for cluster_info in self.clustering_info:
            latex_str += f"{cluster_info.eps} & {cluster_info.number_of_clusters} & "
            latex_str += f"{cluster_info.silhouette_score} & {cluster_info.noise_points} & "
            latex_str += f"{cluster_info.duration} s\\\\\n"
        latex_str += "\\hline\n"
        
        # Part 3: Best EPS Information
        if self.best_eps_info is not None:
            latex_str += "\\multicolumn{5}{|c|}{\\textbf{Best EPS Information}} \\\\\n"
            latex_str += "\\hline\n"
            latex_str += f"{self.best_eps_info.eps} & {self.best_eps_info.number_of_clusters} & "
            latex_str += f"{self.best_eps_info.silhouette_score} & {self.best_eps_info.noise_points} & "
            latex_str += f"{self.best_eps_info.duration} s\\\\\n"
            latex_str += "\\hline\n"
        
        # Part 4: Label Association
        latex_str += "\\multicolumn{5}{|c|}{\\textbf{Label Association}} \\\\\n"
        latex_str += "\\hline\n"
        latex_str += "Cluster ID & \\multicolumn{2}{c|}{Label} & \\multicolumn{2}{c|}{Number of Samples} \\\\\n"
        latex_str += "\\hline\n"
        for label_assoc in self.label_association:
            num_rows = len(label_assoc.label_counts)
            if num_rows > 0:
                latex_str += "\\multirow{" + str(num_rows) + "}{*}{" + str(label_assoc.cluster_id) + "} & "
            for i, (label, count) in enumerate(label_assoc.label_counts.items()):
                if i > 0:
                    latex_str += "& "
                latex_str += f"\\multicolumn{{2}}{{c|}}{{{label}}} & \\multicolumn{{2}}{{c|}}{{{str(count)}}} \\\\\n"
            latex_str += "\\hline\n"

        # Add image in the last row spanning all columns, with adjusted size
        latex_str += "\\multicolumn{5}{|c|}{\\includegraphics[width=0.8\\linewidth]{" + cluster_image_path + "}} \\\\\n"

        # End longtable
        latex_str += "\\end{longtable}\n"
        return latex_str
    

    def to_dict(self) -> Dict:
        """
        Convert the ClusteringResult instance to a dictionary, converting nested objects as well.
        
        Returns:
            Dict: Dictionary representation of the ClusteringResult instance.
        """
        result_dict = asdict(self)
        return result_dict
    
    def save_pie_charts(self, latex_img_path: str):
        num_clusters = len(self.label_association)
        if num_clusters == 0:
            return
        
        # Calculate number of rows and columns for subplots
        num_cols = 2  # for example, can be adjusted
        num_rows = -(-num_clusters // num_cols) # ceiling division

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        fig.suptitle(f'Clusters for {self.instance} of the dataset {self.dataset_name.dataset_number}')

        # Make axs a 2D array for consistency
        axs = np.array(axs, ndmin=2)

        for i, label_assoc in enumerate(self.label_association):
            row = i // num_cols
            col = i % num_cols
            ax = axs[row, col]
            
            labels = list(label_assoc.label_counts.keys())
            sizes = list(label_assoc.label_counts.values())

            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            ax.set_title(f'Cluster : {label_assoc.cluster_id}')

        # If there is an odd number of clusters, remove the last subplot
        if num_clusters % num_cols != 0:
            fig.delaxes(axs[-1, -1])

        # Adjust the layout to make sure everything fits
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the figure
        plt.savefig(latex_img_path)
        plt.close()

def save_clustering_results_to_json(results_list: List[ClusteringResult], file_path: str) -> None:
    """
    Save a list of ClusteringResult instances to a JSON file.

    Args:
        results_list (List[ClusteringResult]): List of ClusteringResult instances.
        file_path (str): The path to the file where the data will be saved.
    """
    # Convert each ClusteringResult instance to a dictionary
    results_dicts = [result.to_dict() for result in results_list]

    # Save the list of dictionaries to a JSON file
    with open(file_path, 'w') as f:
        json.dump(results_dicts, f, indent=4)
    
def clustering_pie_charts(clustering_results: List[ClusteringResult], save_dir_path: str):
    for result in clustering_results:
        num_clusters = len(result.label_association)
        if num_clusters == 0:
            continue
        
        # Calculate number of rows and columns for subplots
        num_cols = 2  # for example, can be adjusted
        num_rows = -(-num_clusters // num_cols) # ceiling division

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        fig.suptitle(f'Clusters for {result.instance} of the dataset {result.dataset_name}')

        # Make axs a 2D array for consistency
        axs = np.array(axs, ndmin=2)

        for i, label_assoc in enumerate(result.label_association):
            row = i // num_cols
            col = i % num_cols
            ax = axs[row, col]
            
            labels = list(label_assoc.label_counts.keys())
            sizes = list(label_assoc.label_counts.values())

            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            ax.set_title(f'Cluster : {label_assoc.cluster_id}')

        # If there is an odd number of clusters, remove the last subplot
        if num_clusters % num_cols != 0:
            fig.delaxes(axs[-1, -1])

        # Adjust the layout to make sure everything fits
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Create directory if not exists
        os.makedirs(save_dir_path, exist_ok=True)

        # Save the figure
        plt.savefig(os.path.join(save_dir_path, f'{result.instance}.png'))
        plt.close()