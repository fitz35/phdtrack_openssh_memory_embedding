from dataclasses import asdict, dataclass
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

    def to_latex(self, correlation_image_path : str):
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

        # Add some vertical space before the image
        latex_str += "\\noalign{\\vskip 5mm}\n"

        # Add image in the last row spanning all columns, with adjusted size
        latex_str += "\\multicolumn{2}{|c|}{\\includegraphics[width=0.8\\linewidth]{" + correlation_image_path + "}} \\\\\n"


        latex_str += "\\hline\n"
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
    
    def save_correlation_matrix_as_heatmap(self, output_path : str):
        MAX_ROWS = 16 # more row means to regenerate the heatmap
        num_rows, num_cols = self.correlation_matrix.shape

        if num_cols > MAX_ROWS or num_rows > MAX_ROWS:
            # Create a heatmap
            cmap = LinearSegmentedColormap.from_list('blue_white_red', ['blue', 'white', 'red'])
            plt.figure(figsize=(10, 10))  # You can adjust the size of the figure here
            heatmap = sns.heatmap(self.correlation_matrix, cmap=cmap, cbar_kws={'label': 'Correlation'})

            # Set labels and title if needed
            heatmap.set_title('Features Correlation Matrix (algorithm : Pearson)', fontdict={'fontsize':12}, pad=12)
            #heatmap.set_xlabel('X-axis Label', fontsize=10)
            #heatmap.set_ylabel('Y-axis Label', fontsize=10)

            # Save the heatmap
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            shutil.copy(self.correlation_image_path, output_path)
        