

from commons.results.commons_results_writer import CommonResultsWriter


class ResultsWriter(CommonResultsWriter):
    """
    This class is used to write the results of a classification pipeline to a CSV file.
    It keeps track of the headers of the CSV file and the results to write.
    It stores everything related to classification results.
    """
    ADDITIONAL_HEADERS: list[str] = [
        # data balancing results
        "data_balancing_duration",
        "nb_samples_before_balancing",
        "nb_samples_after_balancing",
        # classification results
        "classification_duration",
        "precision",
        "recall", 
        "accuracy",
        "f1_score", 
        "support", 
        "true_positives", 
        "true_negatives",
        "false_positives", 
        "false_negatives", 
        "auc"
    ]

    def __init__(self, pipeline_name: str):
        super().__init__(
            "/home/clement/Documents/github/phdtrack_openssh_memory_embedding/results/embedding_quality/result.csv", 
            self.ADDITIONAL_HEADERS, 
            pipeline_name
        )
