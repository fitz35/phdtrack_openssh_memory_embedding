
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


# Hyperparameters
embedding_dim = 64
transformer_units = 128
num_heads = 2
num_transformer_layers = 2
input_length = None  # This allows variable length input
output_dim = 128  # Fixed-size vector dimension



def transformers_pipeline(
        params : ProgramParams,
        samples_and_sample_str_train: SamplesAndLabels,
        samples_and_sample_str_test: SamplesAndLabels,
):
    samples_and_sample_train_samples = __get_input_encoder_from_samplesAndLabels(samples_and_sample_str_train)

    samples_and_sample_test_samples = __get_input_encoder_from_samplesAndLabels(samples_and_sample_str_test)

    encoder = __get_encoder()

    train_nb_samples = samples_and_sample_train_samples.shape[0] # to slice the joined_samples later

    joined_samples = np.dstack((samples_and_sample_train_samples, samples_and_sample_test_samples))


    # train the model
    with time_measure_result(
            f'transformers training : ', 
            params.RESULTS_LOGGER, 
            params.get_results_writer(pipeline=Pipeline.Transformers),
            "model_training_duration"
        ):
        embedded : np.ndarray[Any, Any] = encoder.predict(joined_samples)
    trained_embedded = pd.DataFrame(embedded[:train_nb_samples], columns=[USER_DATA_COLUMN])
    test_embedded = pd.DataFrame(embedded[train_nb_samples:], columns=[USER_DATA_COLUMN])
    

def __get_input_encoder_from_samplesAndLabels(df_train: SamplesAndLabels):
    output = __transform_hex_data(df_train.sample)
    output = output[USER_DATA_COLUMN].tolist()
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

def __get_encoder():
    def transformer_encoder(units: int, num_heads: int) -> tf.keras.Model:
        # Input
        inputs = layers.Input(shape=(None, embedding_dim))
        
        # Multi-head Self Attention
        attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=units)(inputs, inputs)
        attention = layers.Dropout(0.1)(attention)
        add_attention = layers.Add()([inputs, attention])
        attention_out = layers.LayerNormalization()(add_attention)
        
        # Feed-forward Neural Network
        ffn = layers.Dense(units, activation='relu')(attention_out)
        ffn = layers.Dense(embedding_dim)(ffn)
        ffn = layers.Dropout(0.1)(ffn)
        add_ffn = layers.Add()([attention_out, ffn])
        output = layers.LayerNormalization()(add_ffn)
        
        return models.Model(inputs, output)

    # Define the encoder model
    inputs = layers.Input(shape=(input_length, 1))

    # Embedding layer
    embedded = layers.Dense(embedding_dim, activation='relu')(inputs)

    # Stacking multiple transformer encoders
    for _ in range(num_transformer_layers):
        embedded = transformer_encoder(transformer_units, num_heads)(embedded)

    # Pooling layer to get fixed size output
    pooled = layers.GlobalAveragePooling1D()(embedded)
    output = layers.Dense(output_dim, activation='relu')(pooled)

    encoder = models.Model(inputs, output)

    return encoder
