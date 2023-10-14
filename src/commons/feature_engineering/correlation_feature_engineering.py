
from datetime import datetime
from logging import Logger
from typing import List, Literal
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.switch_backend('agg') # to avoid tkinter error, backend no GUI, see https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop
import seaborn as sns

from research_base.utils.utils import DATETIME_FORMAT
from research_base.results.base_result_writer import BaseResultWriter

from commons.feature_engineering.correlation_type import CorrelationType
from commons.data_loading.data_types import SamplesAndLabels


from commons.data_loading.data_origin import DataOriginEnum
from commons.params.common_params import FEATURE_CORRELATION_TYPE, NB_COLUMNS_TO_KEEP


def __correlation_feature_selection( 
        samples_and_labels_train: SamplesAndLabels,
        correlation_method: CorrelationType,
        output_path : str,
        logger : Logger,
        result_writer : BaseResultWriter

    ) -> list[str]:
    """
    Pipeline for feature engineering correlation measurement.
    return: best columns names
    """
    # select algorithm
    correlation_algorithms : dict[CorrelationType, Literal['pearson', 'kendall', 'spearman']] = {
        CorrelationType.FE_CORR_PEARSON: "pearson",
        CorrelationType.FE_CORR_KENDALL: "kendall",
        CorrelationType.FE_CORR_SPEARMAN: "spearman",
    }
    correlation_algorithm = correlation_algorithms[correlation_method]

    # log and results
    logger.info(f"Computing correlation (algorithm: {correlation_algorithm})...")
    

    # Extract samples from training data
    samples = samples_and_labels_train.sample

    # scale the samples to avoid overflows (returns a numpy array)
    scaler = StandardScaler()
    scaled_samples = scaler.fit_transform(samples)

    # Convert scaled_samples back to DataFrame
    scaled_samples_df = pd.DataFrame(scaled_samples, columns=samples.columns)

    # Calculate correlation matrix
    corr_matrix = scaled_samples_df.corr(correlation_algorithm)

    # Print the correlation matrix
    logger.info(f"Correlation matrix (algorithm: {correlation_algorithm}): \n" + str(corr_matrix))

    # Visualize the correlation matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", square=True, cmap='coolwarm')
    plt.title(f"Feature Correlation Matrix (algorithm: {correlation_algorithm})")
    corr_matrix_save_path: str = (
        output_path + 
        "correlation_matrix_" + correlation_algorithm + "_" +
        datetime.now().strftime(DATETIME_FORMAT) +
        ".png"
    )
    plt.savefig(corr_matrix_save_path)
    plt.close()

    # keep best columns
    # Calculate the sum of correlations for each column
    corr_sums = corr_matrix.abs().sum()

    # keep results
    sorted_corr_sums = corr_sums.sort_values(ascending=False)
    result_writer.set_result(
        "descending_best_column_names",
        " ".join(
            sorted_corr_sums.index.tolist()
        )
    )
    result_writer.set_result(
        "descending_best_column_values",
        " ".join(map(str, sorted_corr_sums.tolist()))
    )
    
    # Find the names of the columns that have the smallest sums
    # NOTE: We drop the 1 correlation of the column with itself by substracting 1 to the sums
    corr_sums -= 1
    best_columns_names = corr_sums.nsmallest(NB_COLUMNS_TO_KEEP).index.tolist()
    
    logger.info(f"Keeping columns: {best_columns_names}")

    assert len(best_columns_names) == NB_COLUMNS_TO_KEEP, "The number of best columns is not correct, it should be equal to FEATURE_ENGINEERING_NB_KEEP_BEST_COLUMNS. Maybe there are not enough columns in the dataset."
    assert (type(best_columns_names) == list) and (type(best_columns_names[0]) == str)

    # return the best columns names
    return best_columns_names



def feature_engineering_correlation_measurement(
        origin_to_preprocessed_data: dict[DataOriginEnum, SamplesAndLabels],
        output_path : str,
        logger : Logger,
        result_writer : BaseResultWriter
    ) -> list[str]:
    """
    Pipeline for feature engineering correlation measurement.
    return: best columns names
    """

    preprocess_data_train_samples_list : List[pd.DataFrame] = []
    preprocess_data_train_labels_list : List[pd.Series] = []

    for origin in origin_to_preprocessed_data:
        preprocess_data_train_samples_list.append(origin_to_preprocessed_data[origin].sample)
        preprocess_data_train_labels_list.append(origin_to_preprocessed_data[origin].labels)
    
    # launch the pipeline
    return __correlation_feature_selection(
        SamplesAndLabels(pd.concat(preprocess_data_train_samples_list), pd.concat(preprocess_data_train_labels_list)),
        FEATURE_CORRELATION_TYPE,
        output_path,
        logger,
        result_writer
    )
