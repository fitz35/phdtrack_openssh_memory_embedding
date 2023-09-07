
from typing import Tuple
import pandas as pd

from embedding_quality.params.params import ProgramParams


def clean(params: ProgramParams, samples: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean data.
    1. Remove columns that are composed off only one value
    2. Remove unecessary columns (when usefull ones are provided)
    """
    # Find the indices of columns with only one unique value
    unique_value_columns = samples.nunique() == 1

    # Log the removed columns
    removed_columns = unique_value_columns[unique_value_columns].index
    params.RESULTS_LOGGER.info(f'Removing {len(removed_columns)} columns with only one unique value: {list(removed_columns)}')

    # Remove the columns with only one unique value from the samples
    samples = samples.loc[:, ~unique_value_columns]

    return samples, labels

