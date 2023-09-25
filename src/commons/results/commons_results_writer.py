from research_base.results.base_result_writer import BaseResultWriter

class CommonResultsWriter(BaseResultWriter):
    """
    This class is used to write the results of a classification pipeline to a CSV file.
    It keeps track of the headers of the CSV file and the results to write.
    It stores everything related to classification results.
    """
    __ADDITIONAL_HEADERS: list[str] = [
        "dataset_path",
        "data_loading_duration", 
        "training_dataset_origin", 
        "testing_dataset_origin",
        # feature selection results
        "descending_best_column_names",
        "descending_best_column_values",
        "feature_engineering_duration",
    ]

    def __init__(self, csv_file_path : str, more_header : list[str], pipeline_name: str):
        super().__init__(
            csv_file_path, 
            self.__ADDITIONAL_HEADERS + more_header, 
            pipeline_name
        )