import os
import sys
from sklearn.cluster import OPTICS
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


sys.path.append(os.path.abspath('..'))
from embedding_coherence.data.hyperparams import CLUSTERIZATION_ALGORITHM, CLUSTERIZATION_METHOD, CLUSTERIZATION_METRIC

RANDOM_SEED = 42
MAX_ML_WORKERS = -1


def latex_escape(text):
    return text.replace('_', '\\_')

def extract_default_params(clf, exclude_params=[]):
    default_params = {k: v for k, v in clf.get_params().items() if k not in exclude_params}
    params_df = pd.DataFrame(list(default_params.items()), columns=['Parameter', 'Default Value'])
    return params_df

def generate_latex_table(params_df, caption="Default Hyperparameters"):
    latex_str = "    \\begin{table}[ht]\n"
    latex_str += "        \\centering\n"
    latex_str += f"        \\caption{{{caption}}}\n"
    latex_str += "        \\begin{tabular}{lc}\n"
    latex_str += "            \\textbf{Parameter} & \\textbf{Default Value} \\\\\n"
    for index, row in params_df.iterrows():
        latex_str += f"            {latex_escape(row['Parameter'])} & {row['Default Value']} \\\\\n"
    latex_str += "        \\end{tabular}\n"
    latex_str += "    \\end{table}\n"
    return latex_str

# Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=MAX_ML_WORKERS)
rf_params_df = extract_default_params(random_forest)
rf_latex_table = generate_latex_table(rf_params_df, caption="Default Parameters for Random Forest Classifier")

# OPTICS
optics = OPTICS(metric=CLUSTERIZATION_METRIC, n_jobs=MAX_ML_WORKERS, algorithm=CLUSTERIZATION_ALGORITHM, cluster_method=CLUSTERIZATION_METHOD)
optics_params_df = extract_default_params(optics, exclude_params=['eps', 'min_samples'])
optics_latex_table = generate_latex_table(optics_params_df, caption="Default Parameters for OPTICS Clustering")

# LaTeX Output
print("    \\subsection{Random Forest Classifier}\n")
print(rf_latex_table)
print("    \\subsection{OPTICS Clustering}\n")
print(optics_latex_table)
print(r"""
    \paragraph{Note for OPTICS:}
    \begin{description}
        \item[min\_samples:] Calculated dynamically for each embedding.
        \item[eps:] Takes five distinct values: 0.01, 0.02, 0.03, 0.04, and 0.05.
    \end{description}
""")