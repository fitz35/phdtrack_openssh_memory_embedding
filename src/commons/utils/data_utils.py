from typing import Tuple
import numpy as np
import pandas as pd

def count_positive_and_negative_labels(labels: pd.Series) -> Tuple[int, int]:
    """
    Count the number of positive and negative labels.
    """
    nb_positive_labels = np.count_nonzero(labels)
    nb_negative_labels = len(labels) - nb_positive_labels

    return nb_positive_labels, nb_negative_labels
