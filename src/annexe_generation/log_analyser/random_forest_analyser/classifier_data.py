from dataclasses import dataclass
import os
from typing import Any, Dict, List
import matplotlib.pyplot as plt

import pandas as pd

@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1_score: float
    support: float

    def to_latex(self, label: str) -> str:
        return (f"\\multirow{{4}}{{*}}{{{label}}} & Precision & {self.precision} \\\\\n"
                f" & Recall & {self.recall} \\\\\n"
                f" & F1 Score & {self.f1_score} \\\\\n"
                f" & Support & {self.support} \\\\\n"
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
    def from_dict(d: dict, additional_metrics: dict, duration: float, dataset_name: str, instance: str) -> 'ClassificationResults':
        class_metrics = {key: ClassificationMetrics(**value) for key, value in d.items() if key not in {"accuracy", "macro avg", "weighted avg"}}
        macro_avg = ClassificationMetrics(**d["macro avg"])
        weighted_avg = ClassificationMetrics(**d["weighted avg"])
        accuracy = d.get("accuracy", 0.0)
        return ClassificationResults(dataset_name=dataset_name, instance=instance, class_metrics=class_metrics, 
                                     accuracy=accuracy, macro_avg=macro_avg, weighted_avg=weighted_avg, 
                                     duration=duration, **additional_metrics)
    

    @staticmethod
    def from_json(json_data: Dict[str, Any], dataset_name: str, instance: str, true_positives: int, true_negatives: int,
                false_positives: int, false_negatives: int, auc: float, duration: float) -> 'ClassificationResults':
        # Helper function to adjust the key names in the dictionary
        def adjust_keys(metrics_data: Dict[str, float]) -> Dict[str, float]:
            if 'f1-score' in metrics_data:
                metrics_data['f1_score'] = metrics_data.pop('f1-score')
            return metrics_data
        
        # Construct class metrics
        class_metrics = {key: ClassificationMetrics(**adjust_keys(value)) 
                        for key, value in json_data.items() 
                        if key not in {"accuracy", "macro avg", "weighted avg"}}
        
        # Construct macro average and weighted average metrics
        macro_avg = ClassificationMetrics(**adjust_keys(json_data["macro avg"]))
        weighted_avg = ClassificationMetrics(**adjust_keys(json_data["weighted avg"]))
        
        # Retrieve accuracy, defaulting to 0.0 if not present
        accuracy = json_data.get("accuracy", 0.0)
        
        # Create and return the ClassificationResults object
        return ClassificationResults(dataset_name=dataset_name, instance=instance, class_metrics=class_metrics,
                                    accuracy=accuracy, macro_avg=macro_avg, weighted_avg=weighted_avg,
                                    true_positives=true_positives, true_negatives=true_negatives,
                                    false_positives=false_positives, false_negatives=false_negatives,
                                    auc=auc, duration=duration)
    

    def to_latex(self):
        latex_str = "\\begin{longtable}{|c|c|c|}\n"
        latex_str += "\\caption{" + self.instance + " Classification Results on " + self.dataset_name.replace("_", "\\_") + "} \\label{tab:" + self.instance.lower().replace(" ", "_") + "_classifiers_results} \\\\\n"
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



def plot_metrics(classification_results_list: List[ClassificationResults], save_dir_path: str):
    if not classification_results_list:
        raise ValueError("The list of classification results is empty.")

    # Check that all instances have the same dataset name
    dataset_names = {result.dataset_name for result in classification_results_list}
    if len(dataset_names) != 1:
        raise ValueError("All instances must have the same dataset name.")
    dataset_name = dataset_names.pop()

    # Prepare the data
    data = []
    accuracies = []
    durations = []
    for result in classification_results_list:
        for label, metrics in result.class_metrics.items():
            if label != '0.0':
                data.append({
                    'Instance': result.instance,
                    'Class': label,
                    'Precision': metrics.precision,
                    'Recall': metrics.recall,
                    'F1 Score': metrics.f1_score,
                })
        accuracies.append({
            'Instance': result.instance,
            'Accuracy': result.accuracy
        })
        durations.append({
            'Instance': result.instance,
            'Duration': result.duration
        })

    df = pd.DataFrame(data)
    accuracy_df = pd.DataFrame(accuracies)
    duration_df = pd.DataFrame(durations)

    # Function to create a plot for a specific metric
    def create_plot(metric_name):
        fig, axs = plt.subplots(5, 1, figsize=(10, 25), sharex=True)
        fig.suptitle(f'{dataset_name.replace("_", " ").title()} - {metric_name}', fontsize=16)
        
        for i, metric in enumerate(['Precision', 'Recall', 'F1 Score', 'Accuracy', 'Duration']):
            if metric != 'Accuracy' and metric != 'Duration':
                for class_label in df['Class'].unique():
                    class_df = df[df['Class'] == class_label]
                    axs[i].plot(class_df['Instance'], class_df[metric], marker='o', linestyle='-', label=f'Class {class_label}')
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
        plt.savefig(os.path.join(save_dir_path, f'{dataset_name.lower().replace(" ", "_")}_{metric_name.lower().replace(" ", "_")}.png'))
        #plt.show()

    # Create the plots and save them
    create_plot('Metrics')
