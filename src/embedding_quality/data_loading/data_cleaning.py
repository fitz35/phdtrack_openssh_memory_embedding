
from embedding_quality.params.params import INFO_COLUMNS, ProgramParams
from embedding_quality.data_loading.data_types import SamplesAndLabels


def clean(params: ProgramParams, samples_and_labels: SamplesAndLabels) -> SamplesAndLabels:
    """
    Clean data.
    1. Remove columns that are composed off only one value
    2. Remove unecessary columns (when usefull ones are provided)
    """
    samples = samples_and_labels.sample
    labels = samples_and_labels.labels
    # Find the indices of columns with only one unique value
    unique_value_columns = samples.nunique() == 1

    # Log the removed columns
    removed_columns = unique_value_columns[unique_value_columns].index
    params.RESULTS_LOGGER.info(f'Removing {len(removed_columns)} columns with only one unique value: {list(removed_columns)}')

    # Remove the columns with only one unique value from the samples
    samples = samples.loc[:, ~unique_value_columns]


    return SamplesAndLabels(samples, labels).remove_columns(params, INFO_COLUMNS)