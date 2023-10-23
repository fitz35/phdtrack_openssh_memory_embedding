


import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("input_folder", type=str, help = "The folder containing the chunk to embedded.")
parser.add_argument("output_folder", type=str, help = "The folder which will contain the embedding.")


PIPELINES = [
    #"word2vec",
    "transformers",
]

# get all folder inside input_folder, and if the folder contain only csv, then run the embedding on it, else if it contain a dir,
# retry to run the embedding on the dir.
def find_csv_dirs(directory : str):
    """
    Recursively find all leaf directories containing only CSV files.
    
    :param directory: The root directory to start the search from
    :return: A list of paths to the directories that only contain CSV files
    """
    # List to store paths of directories that only contain CSV files
    csv_dirs : list[str] = []

    def search_dir(current_dir):
        """
        Recursive helper function to search for CSV-only directories.
        
        :param current_dir: The current directory to search in
        """
        nonlocal csv_dirs  # Allows access to csv_dirs defined in the outer function

        # Walk through the directory tree starting from current_dir
        for root, dirs, files in os.walk(current_dir):
            if not dirs:  # If there are no subdirectories, it's a leaf directory
                # Check if all files in the directory are CSV files
                if files and len(files) > 0 and all(file.endswith('.csv') for file in files):
                    # If so, add the directory to the list
                    csv_dirs.append(root)
            else:
                # If there are subdirectories, recursively search them
                for dir in dirs:
                    search_dir(os.path.join(root, dir))

    # Start the recursive search from the provided root directory
    search_dir(directory)
    
    # Return the list of directories containing only CSV files
    return csv_dirs


def main():
    args = parser.parse_args()

    output_folder = args.output_folder
    all_dirs = find_csv_dirs(args.input_folder)

    i = 0

    for dir in all_dirs:
        instance_output_folder = os.path.join(output_folder, os.path.basename(dir))
        os.makedirs(instance_output_folder, exist_ok=True)


        for pipeline in PIPELINES:
            print(f"running embedding on {dir} ( {i}/{len(all_dirs)}) with pipeline {pipeline} (output folder: {instance_output_folder})")
            os.system(f"python3 embedding_generation_main.py -d {dir} -p {pipeline} -o {instance_output_folder} -otr training -ots validation")
        
        i += 1



if __name__ == "__main__":
    main()