

import csv
from dataclasses import dataclass
import logging
import os

# number of file to handle in parallel by the transformers pipeline
TRANSFORMERS_BATCH_SIZE = 5


@dataclass
class TransformersHyperParams:
    """
    Hyperparameters for the Transformers model.
    """
    index : int
    word_byte_size : int
    embedding_dim : int
    transformer_units : int
    num_heads : int
    num_transformer_layers : int
    dropout_rate : float
    activation : str

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



def get_transformers_hyperparams() -> list[TransformersHyperParams]:
    all_hyperparams = []


    index = 0
    word_byte_sizes = [32, 16] # size of the word in bytes
    embedding_dims = [16, 32, 64, 128] # output of the embedding (result size)
    transformer_units = [8, 64, 128, 256] # dimension of the transformer units (see .md)
    num_heads = [2, 4, 8] # attention heads
    num_transformer_layers = [2, 4, 8] # number of transformer layers
    dropout_rates = [0.1, 0.2, 0.3]
    activations = ["relu"]

    for word_byte_size in word_byte_sizes:
        for embedding_dim in embedding_dims:
            for transformer_unit in transformer_units:
                for num_head in num_heads:
                    for num_transformer_layer in num_transformer_layers:
                            for dropout_rate in dropout_rates:
                                for activation in activations:
                                    all_hyperparams.append(TransformersHyperParams(
                                        index=index,
                                        word_byte_size=word_byte_size,
                                        embedding_dim=embedding_dim,
                                        transformer_units=transformer_unit,
                                        num_heads=num_head,
                                        num_transformer_layers=num_transformer_layer,
                                        dropout_rate=dropout_rate,
                                        activation=activation
                                    ))
                                    index += 1



    return all_hyperparams
