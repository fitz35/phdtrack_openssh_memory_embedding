



from dataclasses import asdict, dataclass
import re

@dataclass(frozen=True)
class DatasetData:
    dataset_full_name: str
    dataset_name: str
    dataset_number: int

    filter_entropy : bool
    filter_chunk_size : bool

    def get_display_name(self, excaping : bool = True) -> str:
        return_str = ""
        if self.filter_entropy and self.filter_chunk_size:
            return_str = f"{self.dataset_number} {self.dataset_name} (filtered entropy and chunk size)"
        elif self.filter_entropy:
            return_str = f"{self.dataset_number} {self.dataset_name} (filtered entropy)"
        elif self.filter_chunk_size:
            return_str = f"{self.dataset_number} {self.dataset_name} (filtered chunk size)"
        else:
            return_str = f"{self.dataset_number} {self.dataset_name}"
        if excaping:
            return_str = return_str.replace('_', '\\_')  # Escape underscores
        return return_str

    @staticmethod
    def from_str(dataset_full_name: str) -> 'DatasetData':
        dataset_full_name = dataset_full_name

        # Extract dataset number and name
        dataset_number = None
        dataset_name = None
        filter_entropy = None
        filter_chunk_size = None

        name_match = re.match(r'(\d+)_', dataset_full_name)
        if name_match:
            dataset_number = int(name_match.group(1))
            # The name will be the full name without the number and filters
            filtered_pattern = rf'{dataset_number}_(filtered_)?'
            flags_pattern = r'(-e_|-s_)[^_]+_?'
            dataset_name = re.sub(filtered_pattern, '', dataset_full_name)
            dataset_name = re.sub(flags_pattern, '', dataset_name, count=2)

            # remove last underscore if present
            if dataset_name[-1] == '_':
                dataset_name = dataset_name[:-1]


        # Check for entropy filter
        entropy_match = re.search(r'-e_([^_]+)', dataset_full_name)
        if entropy_match:
            filter_entropy = entropy_match.group(1).lower() != 'none'

        # Check for chunk size filter
        chunk_size_match = re.search(r'-s_([^_]+)', dataset_full_name)
        if chunk_size_match:
            filter_chunk_size = chunk_size_match.group(1).lower() != 'none'

        assert dataset_number is not None, f"Could not extract dataset number from {dataset_full_name}"
        assert dataset_name is not None, f"Could not extract dataset name from {dataset_full_name}"
        assert filter_entropy is not None, f"Could not extract entropy filter from {dataset_full_name}"
        assert filter_chunk_size is not None, f"Could not extract chunk size filter from {dataset_full_name}"
        
        return DatasetData(dataset_full_name, dataset_name, dataset_number, filter_entropy, filter_chunk_size)
    

def datasets_to_latex_longtable(datasets: list[DatasetData]) -> str:
    caption = "Datasets used in the experiments"
    label = "annexes:datasets_descriptions"

    header = f"""
            \\begin{{longtable}}{{|c|c|c|c|}}
            \\caption{{{caption}}}\\label{{{label}}} \\\\
            \\hline
            \\textbf{{Dataset Number}} & \\textbf{{Dataset Name}} & \\textbf{{Filter Entropy}} & \\textbf{{Filter Chunk Size}} \\\\
            \\hline
            \\endfirsthead
            \\multicolumn{{4}}{{c}}
            {{\\tablename\\ \\thetable\\ -- continued from previous page}} \\\\
            \\hline
            \\textbf{{Dataset Number}} & \\textbf{{Dataset Name}} & \\textbf{{Filter Entropy}} & \\textbf{{Filter Chunk Size}} \\\\
            \\hline
            \\endhead
            \\hline \\multicolumn{{4}}{{|r|}}{{Continued on next page}} \\\\ \\hline
            \\endfoot
            \\hline
            \\endlastfoot
            """
    # Start the table with the header

    # Create a row for each DatasetData instance
    rows = ""
    for data in datasets:
        row = f"{data.dataset_number} & {data.dataset_name} & {data.filter_entropy} & {data.filter_chunk_size} \\\\\n\\hline\n"
        row = row.replace('_', '\\_')  # Escape underscores
        rows += row

    footer = "\\end{longtable}"
    return header + rows + footer
