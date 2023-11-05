from dataclasses import asdict, dataclass
import json
import os
import re
import sys
from typing import Any, Dict, List
import matplotlib.pyplot as plt

import pandas as pd

sys.path.append(os.path.abspath('../../..'))
from annexe_generation.log_analyser.dataset_data.dataset_data import DatasetData

@dataclass(frozen=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1_score: float
    support: float
    final_samples: int
    initial_samples: int

    def to_latex(self, label: str) -> str:
        return (f"\\multirow{{4}}{{*}}{{{label}}} & Precision & {self.precision} \\\\\n"
                f" & Recall & {self.recall} \\\\\n"
                f" & F1 Score & {self.f1_score} \\\\\n"
                f" & Support & {self.support} \\\\\n"
                f" & Final Samples (after rebalancing) & {self.final_samples} \\\\\n"
                f" & Initial Samples (before rebalancing) & {self.initial_samples} \\\\\n"
                f"\\hline\n")


@dataclass(frozen=True)
class ClassificationResults:
    dataset_name: DatasetData
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
        final_samples_sum = sum(final_samples.values())
        class_metrics = {key: ClassificationMetrics(**value, initial_samples=initial_samples[key], final_samples=final_samples[key]) for key, value in d.items() if key not in {"accuracy", "macro avg", "weighted avg"}}
        macro_avg = ClassificationMetrics(**d["macro avg"], initial_samples=initial_samples_sum, final_samples=final_samples_sum)
        weighted_avg = ClassificationMetrics(**d["weighted avg"], initial_samples=initial_samples_sum, final_samples=final_samples_sum)
        accuracy = d.get("accuracy", 0.0)
        return ClassificationResults(
            dataset_name=DatasetData.from_str(dataset_name), instance=instance,
            class_metrics=class_metrics, accuracy=accuracy,
            macro_avg=macro_avg, weighted_avg=weighted_avg,
            duration=duration, **additional_metrics
        )

    @staticmethod
    def from_json(json_data: Dict[str, Any], initial_samples: Dict[float, int], final_samples: Dict[float, int],
                  dataset_name: str, instance: str, true_positives: int, true_negatives: int,
                  false_positives: int, false_negatives: int, auc: float, duration: float) -> 'ClassificationResults':
        initial_samples_sum = sum(initial_samples.values())
        final_samples_sum = sum(final_samples.values())
        # Helper function to adjust the key names in the dictionary
        def adjust_keys(metrics_data: Dict[str, float]) -> Dict[str, float]:
            if 'f1-score' in metrics_data:
                metrics_data['f1_score'] = metrics_data.pop('f1-score')
            return metrics_data
        # Construct class metrics
        class_metrics = {key: ClassificationMetrics(**adjust_keys(value), initial_samples=initial_samples[float(key)], final_samples=final_samples[float(key)]) 
                        for key, value in json_data.items() 
                        if key not in {"accuracy", "macro avg", "weighted avg"}}
        
        # Construct macro average and weighted average metrics
        macro_avg = ClassificationMetrics(**adjust_keys(json_data["macro avg"]), initial_samples=initial_samples_sum, final_samples=final_samples_sum)
        weighted_avg = ClassificationMetrics(**adjust_keys(json_data["weighted avg"]), initial_samples=initial_samples_sum, final_samples=final_samples_sum)
        
        # Retrieve accuracy, defaulting to 0.0 if not present
        accuracy = json_data.get("accuracy", 0.0)
        
        # Create and return the ClassificationResults object
        return ClassificationResults(
            dataset_name=DatasetData.from_str(dataset_name), instance=instance,
            class_metrics=class_metrics, accuracy=accuracy,
            macro_avg=macro_avg, weighted_avg=weighted_avg,
            true_positives=true_positives, true_negatives=true_negatives,
            false_positives=false_positives, false_negatives=false_negatives,
            auc=auc, duration=duration
        )
    

    def to_latex(self):
        latex_str = "\\begin{longtable}{|c|c|c|}\n"
        latex_str += "\\caption{" + self.instance + " Classification Results on " + str(self.dataset_name.dataset_number) + "} \\label{tab:" + str(self.dataset_name.dataset_number) + "_" + self.instance.lower().replace(" ", "_") + "_classifiers_results} \\\\\n"
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


def get_best_instances(classification_results: Dict[str, List[ClassificationResults]], metric : list[str]=['accuracy']) -> List[ClassificationResults]:
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
    assert len(metric) >= 1, "The metric list must contain at least one metric."
    # Initialize a dictionary to store the best instances.
    best_instances : list[ClassificationResults] = []
    
    # Iterate through each dataset and its corresponding classification results.
    for dataset_name, results in classification_results.items():
        # Initialize variables to store the best instances and their metric values for Word2Vec and Transformers.
        best_word2vec = None
        best_transformer = None
        best_single_instance = None
        max_word2vec_metric = -float('inf')
        max_transformer_metric = -float('inf')

        # Iterate through each classification result.
        for result in results:
            # Retrieve the value of the specified metric for the current result.
            current_metric_value = getattr(result, metric[0])
            for i in range(1, len(metric)):
                if isinstance(current_metric_value, dict):
                    # If it's a dictionary, use key access
                    current_metric_value = current_metric_value[metric[i]]
                else:
                    # Otherwise, assume it's an attribute
                    current_metric_value = getattr(current_metric_value, metric[i])

            # Check if the instance is a Word2Vec instance and if its metric value is greater than the current maximum.
            if 'word2vec' in result.instance.lower() and current_metric_value > max_word2vec_metric:
                max_word2vec_metric = current_metric_value
                best_word2vec = result

            # Check if the instance is a Transformer instance and if its metric value is greater than the current maximum.
            elif 'transformer' in result.instance.lower() and current_metric_value > max_transformer_metric:
                max_transformer_metric = current_metric_value
                best_transformer = result

            # Check if the instance is a single instance and if its metric value is greater than the current maximum.
            elif 'single' in result.instance.lower() and current_metric_value > max_transformer_metric:
                max_transformer_metric = current_metric_value
                best_single_instance = result
        

        # Store the best instances for the current dataset in the result dictionary.
        if best_single_instance:
            best_instances.append(best_single_instance)
        if best_word2vec:
            best_instances.append(best_word2vec)
        if best_transformer:
            best_instances.append(best_transformer)

    return best_instances

def plot_metrics(classification_results_list: List[ClassificationResults], file_path: str):
    if not classification_results_list:
        raise ValueError("The list of classification results is empty.")
    
    dataset_names = set([result.dataset_name.dataset_number for result in classification_results_list])
    if len(dataset_names) != 1:
        dataset_name = "Multiple Datasets"
    else:
        dataset_name = "the Dataset " + str(dataset_names.pop())

    # Prepare the data
    data = []
    accuracies = []
    durations = []
    for result in classification_results_list:
        instance = str(result.dataset_name.dataset_number) + "." + result.instance
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

    def calculate_limited_axis(df : pd.DataFrame, metrics : str, margin_fraction=0.1):
        min_val = df[metrics].min()
        max_val = df[metrics].max()
        if min_val == max_val:
            return max_val - max_val * margin_fraction, max_val + max_val * margin_fraction
        margin = (max_val - min_val) * margin_fraction
        new_min = max(min_val - margin, 0)  # Ensure that the new_min is not less than 0
        new_max = min(max_val + margin, 1)  # Ensure that the new_max is not more than 1 for accuracy
        return new_min, new_max

    # Plotting
    num_metrics = 5  # Precision, Recall, F1 Score, Accuracy, Duration
    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, num_metrics * 4), sharex=True)
    fig.suptitle('Metrics by Class and Instance for ' + dataset_name, fontsize=16)
    
    for i, metric in enumerate(['Precision', 'Recall', 'F1 Score']):
        metric_df = df.pivot(index='Instance', columns='Class', values=metric)
        lower_bound, upper_bound = calculate_limited_axis(df, metric)
        metric_df.plot(kind='bar', ax=axs[i], legend=i == 0)
        axs[i].set_title(metric)
        axs[i].set_ylabel(metric)
        axs[i].set_ylim(lower_bound, upper_bound)
        if i == 0:
            axs[i].legend(title='Class')

    # Accuracy and Duration don't have classes, so just plot one bar per instance
    accuracy_df.set_index('Instance').plot(kind='bar', ax=axs[3], legend=False, color='blue')
    lower_bound, upper_bound = calculate_limited_axis(accuracy_df, 'Accuracy')
    axs[3].set_title('Accuracy')
    axs[3].set_ylabel('Accuracy')
    axs[3].set_ylim(lower_bound, upper_bound)

    duration_df.set_index('Instance').plot(kind='bar', ax=axs[4], legend=False, color='green')
    axs[4].set_title('Duration')
    axs[4].set_ylabel('Duration (s)')

    axs[-1].set_xlabel('Instance')
    plt.xticks(rotation=90)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(file_path)
