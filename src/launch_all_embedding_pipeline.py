


import argparse
import os

from commons.utils.utils import find_csv_dirs


parser = argparse.ArgumentParser()
parser.add_argument("input_folder", type=str, help = "The folder containing the chunk to embedded.")
parser.add_argument("output_folder", type=str, help = "The folder which will contain the embedding.")


PIPELINES = [
    "word2vec",
    "transformers",
]


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