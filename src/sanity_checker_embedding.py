import os
import argparse

from tqdm import tqdm
import pandas as pd
import concurrent.futures


def check_csv_file(file_path):
    try:
        # Try reading the CSV file
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        return f"WARNING: {os.path.basename(file_path)} is empty"
    except Exception as e:
        return f"ERROR reading {os.path.basename(file_path)}: {e}"

    # Check if the CSV file is empty after reading
    if df.empty:
        return f"WARNING: {os.path.basename(file_path)} is empty after reading"

    return df.columns.tolist(), file_path

def check_csv_consistency(directory):
    for root, dirs, files in os.walk(directory):
        if not files:
            continue  # Skip empty directories

        print(f"Checking embedding in: {root}")

        # Filter out non-CSV files
        file_paths = [os.path.join(root, f) for f in files if f.endswith('.csv')]
        if not file_paths:
            print("  No CSV files found\n")
            continue

        first_header = None
        first_file_name = None

        # Using ThreadPoolExecutor to parallelize the task
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Map the function and its inputs using tqdm for a progress bar
            results = list(tqdm(executor.map(check_csv_file, file_paths), total=len(file_paths), desc="  Checking", unit="file"))

        for headers, file_path in results:
            if isinstance(headers, str):  # If the result is a string, it's a warning or error message
                print(headers)
            else:
                if first_header is None:
                    first_header = headers
                    first_file_name = os.path.basename(file_path)
                elif headers != first_header:
                    print(f"  ERROR: Header in {os.path.basename(file_path)} does not match header in {first_file_name}")

        print("  Checks complete\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check consistency of CSV files in embedding folders.")
    parser.add_argument('directory', type=str, help='Path to the directory containing embedding folders.')
    args = parser.parse_args()
    check_csv_consistency(args.directory)
