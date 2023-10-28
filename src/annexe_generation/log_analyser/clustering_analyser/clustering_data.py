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
        latex_str = f"\\begin{{array}}{{|c|c|}}\n"
        latex_str += "\\hline\n"
        latex_str += f"Dataset & {self.dataset_name} \\\\\n"
        latex_str += "\\hline\n"
        latex_str += f"Instance & {self.instance} \\\\\n"
        latex_str += "\\hline\n"
        latex_str += f"Min Samples & {self.min_samples} \\\\\n"
        latex_str += "\\hline\n"
        latex_str += f"Total Duration & {self.total_duration} s\\\\\n"
        latex_str += "\\hline\n"
        latex_str += "EPS & Number of Clusters & Silhouette Score & Noise Points & Duration \\\\\n"
        latex_str += "\\hline\n"
        for cluster_info in self.clustering_info:
            latex_str += f"{cluster_info.eps} & {cluster_info.number_of_clusters} & "
            latex_str += f"{cluster_info.silhouette_score} & {cluster_info.noise_points} & "
            latex_str += f"{cluster_info.duration} s\\\\\n"
            latex_str += "\\hline\n"
        latex_str += "Best EPS Info & \\\\\n"
        latex_str += "\\hline\n"
        latex_str += f"{self.best_eps_info.eps} & {self.best_eps_info.number_of_clusters} & "
        latex_str += f"{self.best_eps_info.silhouette_score} & {self.best_eps_info.noise_points} & "
        latex_str += f"{self.best_eps_info.duration} s\\\\\n"
        latex_str += "\\hline\n"
        latex_str += "Label Association & \\\\\n"
        latex_str += "\\hline\n"
        for label_assoc in self.label_association:
            latex_str += f"Cluster ID: {label_assoc.cluster_id} & Label Counts: {label_assoc.label_counts} \\\\\n"
            latex_str += "\\hline\n"
        latex_str += "\\end{array}\n"
        return latex_str