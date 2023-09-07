import json
import logging
from typing import Any
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc

from commons.results.base_result_writer import BaseResultWriter
from ..results_utils import time_measure_result

def __get_predicted_classes_from_report(clf_report: dict) -> list:
    """
    Return the classes from the classification report.
    """
    # Create a list to hold the classes
    classes = []
    # Iterate over the keys in the classification report
    for key in clf_report.keys():
        # Ignore the 'accuracy', 'macro avg' and 'weighted avg' keys,
        # as these are not classes
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            # Append the class (as an integer) to the classes list
            classes.append(int(key))
    # Return the classes list
    return classes


def evaluate(
        clf: Any, 
        test_samples: list[list[int]], 
        test_labels: list[int],
        logger : logging.Logger,
        result_saver: BaseResultWriter,
):
    """
    Evaluate the model.
    """
    # evaluate the model
    with time_measure_result('evaluate_model_score', logger):
        
        # make predictions on the test data
        y_pred = clf.predict(test_samples)

        # print information about the labels (test and predicted labels)
        logger.info(
            "Sample of predicted labels: %s \n versus actual labels: %s", y_pred[:20], test_labels[:20]
        )
        logger.info(
            "Number of predicted 1 labels: {} \n versus number of predicted 0 labels: {}".format(
                sum(y_pred), len(y_pred) - sum(y_pred)
            )
        )

        # calculate the accuracy of the model
        # NOTE: accuracy is not a good metric for imbalanced data
        #   here, the base dataset is hugely imbalanced and composed of 99% of 0 labels
        #   so, if the model predicts all 0 labels, it will have an accuracy of 99%...
        accuracy = accuracy_score(test_labels, y_pred)
        logger.info("Accuracy: {:.2f}%".format(accuracy * 100))
        result_saver.set_result("accuracy", str(accuracy))

        # Get the classification report in a dictionary format
        clf_report = classification_report(test_labels, y_pred, output_dict=True)

        # print the classification report as a dict, with 4 spaces as indentation
        logger.info(json.dumps(clf_report, indent=4))

        # Save classification report metrics
        # TODO: not sure about the 1 here, need to check
        precision_str = ""
        recall_str = ""
        f1_score_str = ""
        support_str = ""
        for predicted_class in __get_predicted_classes_from_report(clf_report):
            precision_str += "class-" + str(predicted_class) + "_" + str(clf_report[str(predicted_class)]['precision']) + " "
            recall_str += "class-" + str(predicted_class) + "_" + str(clf_report[str(predicted_class)]['recall']) + " "
            f1_score_str += "class-" +str(predicted_class) + "_" + str(clf_report[str(predicted_class)]['f1-score']) + " "
            support_str += "class-" + str(predicted_class) + "_" + str(clf_report[str(predicted_class)]['support']) + " "
        
        result_saver.set_result("precision", precision_str)
        result_saver.set_result("recall", recall_str)
        result_saver.set_result("f1_score", f1_score_str)
        result_saver.set_result("support", support_str)


        # calculate the confusion matrix
        cm = confusion_matrix(test_labels, y_pred)
        logger.info("Confusion Matrix: ")
        logger.info("True Positives: %d", cm[1, 1])
        logger.info("True Negatives: %d", cm[0, 0])
        logger.info("False Positives: %d", cm[0, 1])
        logger.info("False Negatives: %d", cm[1, 0])

        # save to results
        result_saver.set_result("true_positives", str(cm[1, 1]))
        result_saver.set_result("true_negatives", str(cm[0, 0]))
        result_saver.set_result("false_positives", str(cm[0, 1]))
        result_saver.set_result("false_negatives", str(cm[1, 0]))

        # calculate the false positive rate and true positive rate
        fpr, tpr, thresholds = roc_curve(test_labels, y_pred)

        # calculate the area under the ROC curve
        roc_auc = auc(fpr, tpr)
        logger.info("AUC: {:.2f}".format(roc_auc))
        result_saver.set_result("auc", str(roc_auc))