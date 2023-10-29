from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class ClusterInfo:
    eps: float
    number_of_clusters: int
    silhouette_score: float
    noise_points: int
    duration: float

@dataclass(frozen=True)
class LabelAssociation:
    cluster_id: float
    label_counts: Dict[float, int]

@dataclass(frozen=True)
class ClusteringResult:
    dataset_name: str
    instance: str
    initial_samples: Dict[float, int]
    final_samples: Dict[float, int]
    min_samples: int
    clustering_info: List[ClusterInfo]
    best_eps_info: ClusterInfo
    label_association: List[LabelAssociation]
    total_duration: float


    def to_latex(self) -> str:
        # Start longtable
        latex_str = "\\begin{longtable}{|c|c|c|c|c|}\n"
        latex_str += "\\caption{" + self.instance + " Clustering Results on " + self.dataset_name.replace("_", "\\_") + "} "
        latex_str += "\\label{tab:" + self.instance.lower().replace(" ", "_") + "_clustering_results}\\\\\n"
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


        # End longtable
        latex_str += "\\end{longtable}\n"
        return latex_str