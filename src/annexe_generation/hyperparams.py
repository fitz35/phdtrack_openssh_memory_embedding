
from dataclasses import asdict
import os
import sys

sys.path.append(os.path.abspath('..'))
from embedding_generation.data.hyperparams_transformers import get_transformers_hyperparams
from embedding_generation.data.hyperparams_word2vec import get_word2vec_hyperparams_instances


def generate_latex_array(model_name : str, hyperparameters_instance : dict):
    latex_str = "\\begin{table}[ht]\n"
    latex_str += "\\centering\n"
    latex_str += "\\caption{" + model_name + " Hyperparameters (Configuration " + str(hyperparameters_instance['index']) + ")}\n"
    latex_str += "\\begin{tabular}{|c|c|}\n"
    latex_str += "\\hline\n"
    latex_str += "Hyperparameter & Value \\\\ \\hline\n"
    
    for hyper, val in hyperparameters_instance.items():
        if hyper != 'index':
            latex_str += f"{hyper} & {val} \\\\ \\hline\n"

    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{table}\n"

    return latex_str

transformers_instances =get_transformers_hyperparams()
word2vec_instances = get_word2vec_hyperparams_instances()

for instance in transformers_instances:
    print(generate_latex_array("Transformers", asdict(instance)))


for instance in word2vec_instances:
    print(generate_latex_array("Word2Vec", asdict(instance)))
