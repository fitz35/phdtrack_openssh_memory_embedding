



# get all folder inside input_folder, and if the folder contain only csv, then run the embedding on it, else if it contain a dir,
# retry to run the embedding on the dir.
import os


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