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

