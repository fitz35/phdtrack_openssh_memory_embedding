
import os
from typing import Any
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.sequence import pad_sequences

from research_base.utils.results_utils import time_measure_result


from commons.data_loading.data_types import SamplesAndLabels
from embedding_generation.params.params import USER_DATA_COLUMN, ProgramParams
from embedding_generation.data.data_processing import split_into_chunks
from embedding_generation.params.pipelines import Pipeline
from embedding_generation.data.hyperparams_transformers import TransformersHyperParams, get_transformers_hyperparams


# Hyperparameters
input_length = None  # This allows variable length input



def transformers_pipeline(
        params : ProgramParams,
        samples_and_sample_str_train: SamplesAndLabels,
        samples_and_sample_str_test: SamplesAndLabels,
):
    train_nb_samples = len(samples_and_sample_str_train.labels) # to slice the joined_samples later
    

    all_hyper_params = get_transformers_hyperparams()

    input_data = __get_input_encoder_from_samplesAndLabels(samples_and_sample_str_train, samples_and_sample_str_test)

    for hyperparam in all_hyper_params:
        # test if we have already computed this hyperparam
        trannsformers_folder = os.path.join(params.OUTPUT_FOLDER, "transformers")
        embedding_folder = os.path.join(trannsformers_folder, f"embedding_index_{hyperparam.index}")
        if os.path.exists(embedding_folder):
            continue

        # compute the encoder

        encoder = __get_encoder(params, hyperparam)

        
        # train the model
        with time_measure_result(
                f'transformers training : ', 
                params.RESULTS_LOGGER, 
                params.get_results_writer(pipeline=Pipeline.Transformers),
                "model_training_duration"
            ):
            embedded : np.ndarray[Any, Any] = encoder.predict(input_data, batch_size=1, verbose="1")


        # split the embedded data into train and test
        trained_embedded = pd.DataFrame(embedded[:train_nb_samples], columns=[f'embedded_{i}' for i in range(hyperparam.embedding_dim)])
        assert len(trained_embedded) == len(samples_and_sample_str_train.labels), f"len(trained_embedded)={len(trained_embedded)} != len(samples_and_sample_str_train.labels)={len(samples_and_sample_str_train.labels)}"

        tested_embedded = pd.DataFrame(embedded[train_nb_samples:], columns=[f'embedded_{i}' for i in range(len(embedded) - hyperparam.embedding_dim)])
        assert len(tested_embedded) == len(samples_and_sample_str_test.labels), f"len(tested_embedded)={len(tested_embedded)} != len(samples_and_sample_str_test.labels)={len(samples_and_sample_str_test.labels)}"

        trained = SamplesAndLabels(trained_embedded, samples_and_sample_str_train.labels)
        tested = SamplesAndLabels(tested_embedded, samples_and_sample_str_test.labels)

        # save the embedded data
        
        os.makedirs(embedding_folder, exist_ok=True)

        trained.save_to_csv(os.path.join(embedding_folder, f"training_transformers_embedding.csv"))
        tested.save_to_csv(os.path.join(embedding_folder, f"validation_transformers_embedding.csv"))

        #save the run hyperparams

        hyperparam.log(params.RESULTS_LOGGER)
        hyperparam.append_to_csv(os.path.join(embedding_folder, f"hyperparams.csv"))


    

def __get_input_encoder_from_samplesAndLabels(df_train: SamplesAndLabels, df_test: SamplesAndLabels):
    df = pd.concat([df_train.sample, df_test.sample])

    output = __transform_hex_data(df)
    output = output[USER_DATA_COLUMN].apply(lambda list_x : [float(int(x, 16)) for x in list_x ]).tolist()
    # Pad the sequences
    # first dimension is the number of samples, second is the length of the sequence
    padded_sequences = pad_sequences(output, padding='post', dtype='float32') 

    # Reshape for the encoder like (batch_size, sequence_length, feature_size) (3d)
    padded_sequences = padded_sequences.reshape(padded_sequences.shape[0], padded_sequences.shape[1], 1)

    return padded_sequences


def __transform_hex_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms the specified column of a DataFrame into lists of 2-byte length strings."""
    df[USER_DATA_COLUMN] = df[USER_DATA_COLUMN].apply(lambda x: split_into_chunks(x, 1))
    return df

def __get_encoder(params: ProgramParams, hyperparams: TransformersHyperParams) -> tf.keras.Model:
    def transformer_encoder(units: int, num_heads: int) -> tf.keras.Model:
        # Input
        inputs = layers.Input(shape=(None, hyperparams.embedding_dim))
        
        # Multi-head Self Attention
        attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=units)(inputs, inputs)
        attention = layers.Dropout(hyperparams.dropout_rate, seed=params.RANDOM_SEED)(attention)
        add_attention = layers.Add()([inputs, attention])
        attention_out = layers.LayerNormalization()(add_attention)
        
        # Feed-forward Neural Network
        ffn = layers.Dense(units, hyperparams.activation)(attention_out)
        ffn = layers.Dense(hyperparams.embedding_dim)(ffn)
        ffn = layers.Dropout(hyperparams.dropout_rate)(ffn)
        add_ffn = layers.Add()([attention_out, ffn])
        output = layers.LayerNormalization()(add_ffn)
        
        return models.Model(inputs, output)

    # Define the encoder model
    inputs = layers.Input(shape=(input_length, 1))

    # Embedding layer
    embedded = layers.Dense(hyperparams.embedding_dim, activation=hyperparams.activation)(inputs)

    # Stacking multiple transformer encoders
    for _ in range(hyperparams.num_transformer_layers):
        embedded = transformer_encoder(hyperparams.transformer_units, hyperparams.num_heads)(embedded)

    # Pooling layer to get fixed size output
    pooled = layers.GlobalAveragePooling1D()(embedded)
    output = layers.Dense(hyperparams.embedding_dim, hyperparams.activation)(pooled)

    encoder = models.Model(inputs, output)

    return encoder
