import os
import argparse
import pandas as pd

def check_csv_consistency(directory):
    for root, dirs, files in os.walk(directory):
        if not files:
            continue  # Skip empty directories

        print(f"Checking embedding in: {root}")

        # Filter out non-CSV files
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        if not csv_files:
            print("  No CSV files found\n")
            continue

        # Initialize variables to store the first header and file name
        first_header = None
        first_file_name = None

        # Iterate through the CSV files and perform the checks
        for csv_file in csv_files:
            file_path = os.path.join(root, csv_file)
            try:
                # Try reading the CSV file
                df = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                print(f"  WARNING: {csv_file} is empty")
                continue
            except Exception as e:
                print(f"  ERROR reading {csv_file}: {e}")
                continue

            # Check if the CSV file is empty after reading
            if df.empty:
                print(f"  WARNING: {csv_file} is empty after reading")
                continue

            # Check for consistency in headers
            if first_header is None:
                first_header = df.columns.tolist()
                first_file_name = csv_file
            elif df.columns.tolist() != first_header:
                print(f"  ERROR: Header in {csv_file} does not match header in {first_file_name}")

        print("  Checks complete\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check consistency of CSV files in embedding folders.")
    parser.add_argument('directory', type=str, help='Path to the directory containing embedding folders.')
    args = parser.parse_args()
    check_csv_consistency(args.directory)
