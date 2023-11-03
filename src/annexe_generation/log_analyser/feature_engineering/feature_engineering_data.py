from dataclasses import asdict, dataclass
import json
import os
import shutil
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import pandas as pd

sys.path.append(os.path.abspath('../../..'))
from annexe_generation.log_analyser.dataset_data.dataset_data import DatasetData

@dataclass(frozen=True)
class CorrelationSum:
    feature_name: str
    correlation_sum: float

    @staticmethod
    def from_correlation_dataframe(correlation_dataframe: pd.DataFrame) -> list['CorrelationSum']:
        """
        Create a list of CorrelationSum instances from a correlation matrix.

        This method processes a given correlation DataFrame, calculating the sum of
        absolute values for each column's correlation coefficients, excluding the correlation
        of each variable with itself (which is always 1). It then sorts these sums in ascending order
        to determine the variables with the lowest overall correlation to the others.
        
        Parameters:
        correlation_dataframe (pd.DataFrame): A pandas DataFrame where each element is a 
            correlation coefficient between two variables, with variables represented both in
            rows and columns.
            
        Returns:
        list[CorrelationSum]: A list of CorrelationSum instances, each containing the name of a 
            variable and the sum of its absolute correlations with all other variables, sorted in 
            ascending order of summed correlation.
        """

        # Initialize an empty list to hold the results.
        result_list: list[CorrelationSum] = []
        
        # Compute the sum of absolute values of the correlation coefficients for each column.
        # This gives us a measure of how each variable is correlated with all others.
        corr_sums = correlation_dataframe.abs().sum()
        
        # Adjust the sums by subtracting 1 to remove the perfect correlation each variable has with itself.
        # This step assumes that the main diagonal of the correlation matrix is filled with ones.
        corr_sums -= 1
        
        # Sort the columns by their adjusted correlation sum in ascending order.
        # This change in order implies that we are now interested in the variables with the
        # lowest overall correlation with others, as opposed to the highest.
        sorted_corr_sums = corr_sums.sort_values(ascending=True)
        
        # Iterate over the sorted series to populate the result_list with CorrelationSum instances.
        for column_name, correlation_sum in sorted_corr_sums.items():
            # Create an instance of CorrelationSum with the appropriate values.
            correlation_sum_instance = CorrelationSum(str(column_name), correlation_sum)
            
            # Append the new instance to the result list.
            result_list.append(correlation_sum_instance)
        
        # Return the list of CorrelationSum instances, now sorted by the sum of correlations.
        return result_list


@dataclass(frozen=True)
class FeatureEngineeringData:
    dataset_name: DatasetData
    instance: str
    correlation_matrix: pd.DataFrame
    correlation_image_path : str
    correlation_sum_sorted_list: list[CorrelationSum]
    best_columns : list[str]
    
    def to_latex(self, correlation_image_path : str):
        # Start of the LaTeX table
        latex_str = "\\begin{longtable}{|c|c|}\n"
        latex_str += "\\caption{" + self.instance + " Feature Engineering Results on " + str(self.dataset_name.dataset_number) + "} "
        latex_str += "\\label{tab:" + str(self.dataset_name.dataset_number) + "_" + self.instance.lower().replace(" ", "_") + "_feature_engineering_results}\\\\\n"
        latex_str += "\\hline\n"

        # Dataset name and instance
        latex_str += "Dataset Name & " + str(self.dataset_name.dataset_number) + " \\\\ \\hline\n"
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
        latex_str += "\\caption{" + self.instance + " Correlation Matrix on " + str(self.dataset_name.dataset_number) + "} "
        latex_str += "\\label{tab:" + str(self.dataset_name.dataset_number) + "_" + self.instance.lower().replace(" ", "_") + "_correlation_matrix}\\\\\n"
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


def features_engineering_list_to_json(data : list[FeatureEngineeringData], file_path : str) :
    # Convert the list of FeatureEngineeringData to a list of dictionaries
    data_dict_list = [asdict(feature_engineering_data) for feature_engineering_data in data]

    # Serialize the pandas DataFrame to a CSV format or convert to JSON
    for data_dict in data_dict_list:
        data_dict['correlation_matrix'] = json.loads(data_dict['correlation_matrix'].to_json())

    # Write the dictionary to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(data_dict_list, json_file, indent=4)