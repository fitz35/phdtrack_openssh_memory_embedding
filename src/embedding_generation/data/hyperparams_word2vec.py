
import csv
from dataclasses import dataclass
import logging
import os
import re


@dataclass
class Word2vecHyperparams:

    index : int

    output_size: int
    # word2vec window size, in bytes. To have the number of words in a window, divide by the word size in bytes
    window_bytes_size: int

    word_byte_size: int

    min_count: int = 1

    def to_dir_name(self) -> str:
        attributes = [
            "output_size", "window_bytes_size", "word_byte_size",
            "min_count"
        ]
        dir_name = "_".join(f"{attr}={getattr(self, attr)}" for attr in attributes)
        
        # Replace or remove special characters
        dir_name = re.sub(r"[^\w\s-]", "", dir_name)
        dir_name = re.sub(r"[-\s]+", "-", dir_name).strip("-_")
        
        return dir_name

    def log(self, logger : logging.Logger):
        logger.info(f"TransformersHyperParams : {self.__dict__}")

    def append_to_csv(self, csv_file_path : str) :
        """Append a dataclass instance to a CSV file."""
        # Check if the file exists to determine if headers should be written
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, 'a', newline='') as csvfile:
            # Convert dataclass to dictionary
            data_dict = self.__dict__
            headers = list(data_dict.keys())
            
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            
            # Write header only if the file didn't exist before
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(data_dict)


def get_word2vec_hyperparams_instances() -> list[Word2vecHyperparams]:
    instances : list[Word2vecHyperparams] = []

    outputs_sizes = [8, 16, 100]
    window_bytes_sizes = [8, 16]
    word_byte_sizes = [2, 4] # size of the word in bytes
    

    index = 0

    for output_size in outputs_sizes:
        for window_bytes_size in window_bytes_sizes:
            for word_byte_size in word_byte_sizes:
                instances.append(Word2vecHyperparams(
                    index=index,
                    output_size=output_size,
                    window_bytes_size=window_bytes_size,
                    word_byte_size=word_byte_size
                ))

                index += 1




    return instances