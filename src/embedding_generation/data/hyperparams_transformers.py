

import csv
import re
from dataclasses import dataclass
import logging
import os


@dataclass
class TransformersHyperParams:
    """
    Hyperparameters for the Transformers model.
    """
    index : int
    word_character_size : int
    embedding_dim : int
    transformer_units : int
    num_heads : int
    num_transformer_layers : int
    dropout_rate : float
    activation : str


    def to_dir_name(self) -> str:
        attributes = [
            "word_character_size", "embedding_dim", "transformer_units",
            "num_heads", "num_transformer_layers", "dropout_rate", "activation"
        ]
        dir_name = "_".join(f"{attr}={getattr(self, attr)}" for attr in attributes)
        
        # Replace or remove special characters
        dir_name = re.sub(r"[^\w\s-]", "", dir_name)
        dir_name = re.sub(r"[-\s]+", "-", dir_name).strip("-_")
        
        return "transformers_" + dir_name


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
    word_character_sizes = [16, 8] # size of the word in bytes (take care to not overflow f64, so max 8 bytes, ie 16 characters)
    embedding_dims = [8, 16] # output of the embedding (result size)
    transformer_units = [2, 4] # dimension of the transformer units (see .md)
    num_heads = [2, 4] # attention heads
    num_transformer_layers = [2, 4] # number of transformer layers
    dropout_rates = [0.1, 0.3]
    activations = ["relu"]

    zipped = zip(zip(zip(transformer_units, num_heads), num_transformer_layers), dropout_rates)

    for (((transformer_unit, num_head), num_transformer_layer), dropout_rate) in zipped:
        for word_character_size in word_character_sizes:
            for embedding_dim in embedding_dims:
           
                for activation in activations:
                    all_hyperparams.append(TransformersHyperParams(
                        index=index,
                        word_character_size=word_character_size,
                        embedding_dim=embedding_dim,
                        transformer_units=transformer_unit,
                        num_heads=num_head,
                        num_transformer_layers=num_transformer_layer,
                        dropout_rate=dropout_rate,
                        activation=activation
                    ))
                    index += 1



    return all_hyperparams
