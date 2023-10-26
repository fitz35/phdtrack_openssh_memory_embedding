
# This script will run the embedding_coherence_main.py and embedding_quality_main.py on all the dataset in the input folder.
# The input folder should contain only dataset folder, and each dataset folder should contain only csv files.
# if "filtered" is in the dataset folder name, then the embedding_quality_main.py will be run with the --no_balancing option.

import argparse
import os
from commons.utils.utils import find_csv_dirs

from dotenv import load_dotenv


load_dotenv(".env")

parser = argparse.ArgumentParser()
parser.add_argument("input_folder", type=str, help = "The folder containing all the embedding.")



def main():
    args = parser.parse_args()

    all_dirs = find_csv_dirs(args.input_folder)

    os.makedirs(os.environ["FEATURE_CORRELATION_MATRICES_RESULTS_DIR_PATH"], exist_ok=True)
    os.makedirs(os.environ["COMMON_LOGGER_DIR_PATH"], exist_ok=True)
    os.makedirs(os.environ["RESULTS_LOGGER_DIR_PATH"], exist_ok=True)

    i = 0
    for dir in all_dirs:
        print(f"running test on {dir} ( {i}/{len(all_dirs)})")

        if "filtered" in dir:
            filtered = "--no_balancing"
        else:
            filtered = ""

        os.system(f"python3 main.py -d {dir} -o ../log -otr training -ots validation {filtered} -p testingembedding")

        i += 1



if __name__ == "__main__":
    main()


