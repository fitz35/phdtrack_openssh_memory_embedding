from research_base.results.base_result_writer import BaseResultWriter

class ResultsWriter(BaseResultWriter):
    """
    This class is used to write the results of a classification pipeline to a CSV file.
    It keeps track of the headers of the CSV file and the results to write.
    It stores everything related to classification results.
    """
    ADDITIONAL_HEADERS: list[str] = [
        "dataset_path",
        "data_loading_duration", 
        "data_balancing_duration",
        "training_dataset_origin", 
        "testing_dataset_origin",
        "nb_training_samples_before_balancing",
        "nb_positive_training_samples_before_balancing",
        "nb_training_samples_after_balancing",
        "nb_positive_training_samples_after_balancing", 
        # feature selection results
        "descending_best_column_names",
        "descending_best_column_values",
        "feature_engineering_duration",
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
