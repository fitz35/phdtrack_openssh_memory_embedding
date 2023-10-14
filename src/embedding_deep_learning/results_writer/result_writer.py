import os

from research_base.results.base_result_writer import BaseResultWriter

class ResultsWriter(BaseResultWriter):
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
    ]

    # NOTE: The result writer passed to the program params class should only have
    # the :pipeline_name: parameter in its init args.
    def __init__(self, pipeline_name: str):

        result_csv_save_path = os.environ.get("RESULTS_LOGGER_DIR_PATH")
        if result_csv_save_path is None:
            raise Exception("ERROR: RESULTS_LOGGER_DIR_PATH env var not set.")
        elif not os.path.exists(result_csv_save_path):
            raise Exception("ERROR: RESULTS_LOGGER_DIR_PATH env var does not point to a valid path.")

        super().__init__(
            csv_file_path = result_csv_save_path, 
            more_header = self.__ADDITIONAL_HEADERS, 
            pipeline_name = pipeline_name,
        )