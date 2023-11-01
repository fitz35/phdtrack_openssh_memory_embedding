from dataclasses import asdict, dataclass

import pandas as pd

@dataclass
class CorrelationSum:
    class_name: str
    correlation_sum: float


@dataclass
class FeatureEngineeringData:
    correlation_matrix: pd.DataFrame
    correlation_image_path : str
    correlation_sum_sorted_list: list[CorrelationSum]