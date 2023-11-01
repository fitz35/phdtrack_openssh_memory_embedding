from dataclasses import asdict, dataclass
import json
import os
import re
from typing import Any, Dict, List
import matplotlib.pyplot as plt

import pandas as pd

@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1_score: float
    support: float
    initial_samples: int

    def to_latex(self, label: str) -> str:
        return (f"\\multirow{{4}}{{*}}{{{label}}} & Precision & {self.precision} \\\\\n"
                f" & Recall & {self.recall} \\\\\n"
                f" & F1 Score & {self.f1_score} \\\\\n"
                f" & Support & {self.support} \\\\\n"
                f" & Initial Samples (before rebalancing) & {self.initial_samples} \\\\\n"
                f"\\hline\n")


@dataclass
class ClassificationResults:
    dataset_name: str
    instance: str
    class_metrics: Dict[str, ClassificationMetrics]
    accuracy: float
    macro_avg: ClassificationMetrics
    weighted_avg: ClassificationMetrics
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    auc: float
    duration: float  # Duration of the random forest operation in seconds

    @staticmethod
    def from_dict(d: dict, initial_samples: Dict[float, int], final_samples: Dict[float, int],
                  additional_metrics: dict, duration: float, dataset_name: str, instance: str) -> 'ClassificationResults':
        initial_samples_sum = sum(initial_samples.values())
        class_metrics = {key: ClassificationMetrics(**value, initial_samples=initial_samples[key]) for key, value in d.items() if key not in {"accuracy", "macro avg", "weighted avg"}}
        macro_avg = ClassificationMetrics(**d["macro avg"], initial_samples=initial_samples_sum)
        weighted_avg = ClassificationMetrics(**d["weighted avg"], initial_samples=initial_samples_sum)
        accuracy = d.get("accuracy", 0.0)
        return ClassificationResults(
            dataset_name=dataset_name, instance=instance,
            class_metrics=class_metrics, accuracy=accuracy,
            macro_avg=macro_avg, weighted_avg=weighted_avg,
            duration=duration, **additional_metrics
        )

    @staticmethod
    def from_json(json_data: Dict[str, Any], initial_samples: Dict[float, int], final_samples: Dict[float, int],
                  dataset_name: str, instance: str, true_positives: int, true_negatives: int,
                  false_positives: int, false_negatives: int, auc: float, duration: float) -> 'ClassificationResults':
        initial_samples_sum = sum(initial_samples.values())
        # Helper function to adjust the key names in the dictionary
        def adjust_keys(metrics_data: Dict[str, float]) -> Dict[str, float]:
            if 'f1-score' in metrics_data:
                metrics_data['f1_score'] = metrics_data.pop('f1-score')
            return metrics_data
        print(initial_samples)
        # Construct class metrics
        class_metrics = {key: ClassificationMetrics(**adjust_keys(value), initial_samples=initial_samples[float(key)]) 
                        for key, value in json_data.items() 
                        if key not in {"accuracy", "macro avg", "weighted avg"}}
        
        # Construct macro average and weighted average metrics
        macro_avg = ClassificationMetrics(**adjust_keys(json_data["macro avg"]), initial_samples=initial_samples_sum)
        weighted_avg = ClassificationMetrics(**adjust_keys(json_data["weighted avg"]), initial_samples=initial_samples_sum)
        
        # Retrieve accuracy, defaulting to 0.0 if not present
        accuracy = json_data.get("accuracy", 0.0)
        
        # Create and return the ClassificationResults object
        return ClassificationResults(
            dataset_name=dataset_name, instance=instance,
            class_metrics=class_metrics, accuracy=accuracy,
            macro_avg=macro_avg, weighted_avg=weighted_avg,
            true_positives=true_positives, true_negatives=true_negatives,
            false_positives=false_positives, false_negatives=false_negatives,
            auc=auc, duration=duration
        )
    

    def to_latex(self):
        latex_str = "\\begin{longtable}{|c|c|c|}\n"
        latex_str += "\\caption{" + self.instance + " Classification Results on " + self.dataset_name.replace("_", "\\_") + "} \\label{tab:" + self.dataset_name + "_" + self.instance.lower().replace(" ", "_") + "_classifiers_results} \\\\\n"
        latex_str += "\\hline\n"
        latex_str += "Class & Metric Name & Metric Value \\\\\n"
        latex_str += "\\hline\n"

        # Add class metrics
        for label, metrics in self.class_metrics.items():
            latex_str += metrics.to_latex(label)

        # Add macro average
        latex_str += self.macro_avg.to_latex("Macro Avg")

        # Add weighted average
        latex_str += self.weighted_avg.to_latex("Weighted Avg")

        # Add additional metrics
        additional_metrics = [
            ("Accuracy", self.accuracy),
            ("True Positives", self.true_positives),
            ("True Negatives", self.true_negatives),
            ("False Positives", self.false_positives),
            ("False Negatives", self.false_negatives),
            ("AUC", self.auc),
            ("Duration (seconds)", self.duration),
        ]
        for label, value in additional_metrics:
            latex_str += f"& {label} & {value} \\\\ \\hline\n"

        latex_str += "\\end{longtable}\n"
        return latex_str
    
    def to_dict(self) -> Dict:
        """
        Convert the ClassificationResults instance to a dictionary, converting nested ClassificationMetrics as well.
        
        Returns:
            Dict: Dictionary representation of the ClassificationResults instance.
        """
        result_dict = asdict(self)
        return result_dict

def save_classification_results_to_json(results_list: List[ClassificationResults], file_path: str) -> None:
    """
    Save a list of ClassificationResults instances to a JSON file.

    Args:
        results_list (List[ClassificationResults]): List of ClassificationResults instances.
        file_path (str): The path to the file where the data will be saved.
    """
    # Convert each ClassificationResults instance to a dictionary
    results_dicts = [result.to_dict() for result in results_list]

    # Save the list of dictionaries to a JSON file
    with open(file_path, 'w') as f:
        json.dump(results_dicts, f, indent=4)


def get_best_instances(classification_results: Dict[str, List[ClassificationResults]], metric='accuracy') -> List[ClassificationResults]:
    """
    Extract the best instances based on a specified metric for Word2Vec and Transformers for each dataset.
    
    Args:
    classification_results (Dict[str, List[ClassificationResults]]): A dictionary where keys are dataset names and 
                                                                      values are lists of ClassificationResults objects.
    metric (str): The metric to be used for comparing instances. Default is 'accuracy'.
    
    Returns:
    Dict[str, Dict[str, ClassificationResults]]: A dictionary with dataset names as keys. The values are another dictionary
                                                  with 'word2vec' and 'transformer' as keys, and the best ClassificationResults
                                                  instances as values.
    """
    # Initialize a dictionary to store the best instances.
    best_instances : list[ClassificationResults] = []
    
    # Iterate through each dataset and its corresponding classification results.
    for dataset_name, results in classification_results.items():
        # Initialize variables to store the best instances and their metric values for Word2Vec and Transformers.
        best_word2vec = None
        best_transformer = None
        max_word2vec_metric = -float('inf')
        max_transformer_metric = -float('inf')

        # Iterate through each classification result.
        for result in results:
            # Retrieve the value of the specified metric for the current result.
            current_metric_value = getattr(result, metric)

            # Check if the instance is a Word2Vec instance and if its metric value is greater than the current maximum.
            if 'word2vec' in result.instance.lower() and current_metric_value > max_word2vec_metric:
                max_word2vec_metric = current_metric_value
                best_word2vec = result

            # Check if the instance is a Transformer instance and if its metric value is greater than the current maximum.
            elif 'transformer' in result.instance.lower() and current_metric_value > max_transformer_metric:
                max_transformer_metric = current_metric_value
                best_transformer = result

        assert best_word2vec is not None and best_transformer is not None, "The best instances should not be None."

        # Store the best instances for the current dataset in the result dictionary.
        best_instances.append(best_word2vec)
        best_instances.append(best_transformer)

    return best_instances

def plot_metrics(classification_results_list: List[ClassificationResults], save_dir_path: str, file_name: str):
    if not classification_results_list:
        raise ValueError("The list of classification results is empty.")
    
    def extract_leading_number(text: str) -> int:
        """
        Extracts the leading number from a string.
        
        Args:
        text (str): The input string from which to extract the number.
        
        Returns:
        int: The extracted number.
        
        Raises:
        ValueError: If no leading number is found in the string.
        """
        match = re.match(r"(\d+)_", text)
        if match:
            return int(match.group(1))
        else:
            raise ValueError("No leading number found in the string")

    # Prepare the data
    data = []
    accuracies = []
    durations = []
    for result in classification_results_list:
        instance = str(extract_leading_number(result.dataset_name)) + "." + result.instance
        for label, metrics in result.class_metrics.items():
            if label != '0.0':
                data.append({
                    'Instance': instance,
                    'Class': label,
                    'Precision': metrics.precision,
                    'Recall': metrics.recall,
                    'F1 Score': metrics.f1_score,
                })
        accuracies.append({
            'Instance': instance,
            'Accuracy': result.accuracy
        })
        durations.append({
            'Instance': instance,
            'Duration': result.duration
        })

    df = pd.DataFrame(data)
    accuracy_df = pd.DataFrame(accuracies)
    duration_df = pd.DataFrame(durations)

    # Function to create a plot for a specific metric
    def create_plot(file_name : str):
        fig, axs = plt.subplots(5, 1, figsize=(10, 25), sharex=True)
        fig.suptitle('Metrics', fontsize=16)
        
        for i, metric in enumerate(['Precision', 'Recall', 'F1 Score', 'Accuracy', 'Duration']):
            if metric != 'Accuracy' and metric != 'Duration':
                for class_label in df['Class'].unique():
                    class_df = df[df['Class'] == class_label]
                    axs[i].plot(class_df['Instance'], class_df[metric], marker='o', linestyle='-', label=f'Class {class_label}')
                    axs[i].legend()
            elif metric == 'Accuracy':
                axs[i].plot(accuracy_df['Instance'], accuracy_df['Accuracy'], marker='o', linestyle='-', color='blue')
                axs[i].legend(['Accuracy'])
            else:
                axs[i].plot(duration_df['Instance'], duration_df['Duration'], marker='o', linestyle='-', color='green')
                axs[i].legend(['Duration (s)'])
                
            axs[i].set_ylabel(metric)
            axs[i].grid(True)
        
        axs[4].set_xlabel('Instance')
        plt.xticks(rotation=45)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir_path, f'{file_name.lower().replace(" ", "_")}.png'))
        #plt.show()

    # Create the plots and save them
    create_plot(file_name)
