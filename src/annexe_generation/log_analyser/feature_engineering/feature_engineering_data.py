from dataclasses import asdict, dataclass

import pandas as pd

@dataclass
class CorrelationSum:
    feature_name: str
    correlation_sum: float


@dataclass
class FeatureEngineeringData:
    dataset_name: str
    instance: str
    correlation_matrix: pd.DataFrame
    correlation_image_path : str
    correlation_sum_sorted_list: list[CorrelationSum]
    best_columns : list[str]

    def to_latex(self):
        # Start of the LaTeX table
        latex_str = "\\begin{longtable}{|c|c|}\n"
        latex_str += "\\caption{" + self.instance + " Feature Engineering Results on " + self.dataset_name.replace("_", "\\_") + "} "
        latex_str += "\\label{tab:" + self.dataset_name + "_" + self.instance.lower().replace(" ", "_") + "_feature_engineering_results}\\\\\n"
        latex_str += "\\hline\n"

        # Dataset name and instance
        latex_str += "Dataset Name & " + self.dataset_name.replace("_", "\\_") + " \\\\ \\hline\n"
        latex_str += "Instance & " + self.instance + " \\\\ \\hline\n"

        # Best features
        if self.best_columns:
            latex_str += "\\multirow{" + str(len(self.best_columns)) + "}{*}{Best Features} & " + self.best_columns[0].replace("_", "\\_") + " \\\\ \\cline{2-2}\n"
            for feature in self.best_columns[1:]:
                latex_str += " & " + feature.replace("_", "\\_") + " \\\\ \\cline{2-2}\n"
        else:
            latex_str += "Best Features & None \\\\ \\hline\n"

        # End of the LaTeX table
        latex_str += "\\end{longtable}\n"
        return latex_str
    
    def correlation_matrix_to_latex(self):
        # Start of the LaTeX table
        latex_str = "\\begin{longtable}{|" + "c|"*(self.correlation_matrix.shape[1]+1) + "}\n"
        latex_str += "\\caption{" + self.instance + " Correlation Matrix on " + self.dataset_name.replace("_", "\\_") + "} "
        latex_str += "\\label{tab:" + self.dataset_name + "_" + self.instance.lower().replace(" ", "_") + "_correlation_matrix}\\\\\n"
        latex_str += "\\hline\n"

        # Column names
        latex_str += " & " + " & ".join(self.correlation_matrix.columns).replace("_", "\\_") + " \\\\ \\hline\n"

        # Rows of the correlation matrix
        for index, row in self.correlation_matrix.iterrows():
            row_str = " & ".join([str(val) for val in row])
            latex_str += str(index).replace("_", "\\_") + " & " + row_str + " \\\\ \\hline\n"

        # End of the LaTeX table
        latex_str += "\\end{longtable}\n"
        return latex_str