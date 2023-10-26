


import argparse
import os

from commons.utils.utils import find_csv_dirs
from dotenv import load_dotenv


load_dotenv(".env")

parser = argparse.ArgumentParser()
parser.add_argument("input_folder", type=str, help = "The folder containing the chunk to embedded.")
parser.add_argument("output_folder", type=str, help = "The folder which will contain the embedding.")


def main():
    args = parser.parse_args()

    output_folder = args.output_folder
    all_dirs = find_csv_dirs(args.input_folder)

    os.makedirs(os.environ["FEATURE_CORRELATION_MATRICES_RESULTS_DIR_PATH"], exist_ok=True)
    os.makedirs(os.environ["COMMON_LOGGER_DIR_PATH"], exist_ok=True)
    os.makedirs(os.environ["RESULTS_LOGGER_DIR_PATH"], exist_ok=True)

    i = 0

    for dir in all_dirs:
        instance_output_folder = os.path.join(output_folder, os.path.basename(dir))
        os.makedirs(instance_output_folder, exist_ok=True)


        print(f"running embedding on {dir} ( {i}/{len(all_dirs)}) with pipeline (output folder: {instance_output_folder})")
        os.system(f"python3 main.py -d {dir} -p deeplearning -o {instance_output_folder} -otr training -ots validation")
        
        i += 1



if __name__ == "__main__":
    main()