
from commons.results.commons_results_writer import CommonResultsWriter


class ResultsWriter(CommonResultsWriter):
    """
    This class is used to write the results of a classification pipeline to a CSV file.
    It keeps track of the headers of the CSV file and the results to write.
    It stores everything related to classification results.
    """
    ADDITIONAL_HEADERS: list[str] = [
        


    ]

    def __init__(self, pipeline_name: str):
        super().__init__(
            "/home/clement/Documents/github/phdtrack_openssh_memory_embedding/results/embedding_coherence/result.csv", 
            self.ADDITIONAL_HEADERS, 
            pipeline_name
        )