
import contextlib
import io
import os
from typing import Any

import numpy as np
import pandas as pd
import timeout_decorator
from timeout_decorator import TimeoutError
import tensorflow as tf
from keras import layers, models
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from research_base.utils.results_utils import time_measure_result


from commons.data_loading.data_types import SamplesAndLabels
from embedding_generation.data.data_processing import split_into_chunks
from embedding_generation.data.hyperparams_transformers import TransformersHyperParams, get_transformers_hyperparams
from params.common_params import USER_DATA_COLUMN, CommonProgramParams
from testing_pipelines.pipeline import pipeline as testing_pipeline


def transformers_pipeline(
        params : CommonProgramParams,
        samples_and_sample_str_train: SamplesAndLabels,
        samples_and_sample_str_test: SamplesAndLabels,
):
    train_nb_samples = len(samples_and_sample_str_train.labels) # to slice the joined_samples later
    

    all_hyper_params = get_transformers_hyperparams()

    for hyperparam in all_hyper_params:
        params.RESULTS_LOGGER.info(f"!!!!!!!!!!!!! Transformers instance : {hyperparam.to_dir_name()} !!!!!!!!!!!!!")
        
        # test if we have already computed this hyperparam
        embedding_folder_name = f"embedding_transformers_{hyperparam.to_dir_name()}"
        embedding_folder = os.path.join(params.OUTPUT_FOLDER, embedding_folder_name)
        if os.path.exists(embedding_folder):
            params.RESULTS_LOGGER.info(f"Transformers instance {hyperparam} already computed")
            continue
        
        params.RESULTS_LOGGER.info(f"Transformers instance : {hyperparam}")

        input_data = __get_input_encoder_from_samplesAndLabels(hyperparam, samples_and_sample_str_train, samples_and_sample_str_test)
        tocken_number = input_data.shape[1]
        
        params.RESULTS_LOGGER.info(f"token number for instance {hyperparam.index} (with padding) : {tocken_number}")
        

        params.COMMON_LOGGER.info(f"input_data shape : {input_data.shape}")

        # compute the encoder

        encoder = __get_encoder(params, hyperparam, tocken_number)
        params.RESULTS_LOGGER.info(f"encoder summary : {__get_model_summary(encoder)}")

        
        @timeout_decorator.timeout(seconds=params.TIMEOUT_DURATION, timeout_exception=TimeoutError)
        def train_model(encoder : Model, input_data : np.ndarray, params :CommonProgramParams):
            embedded: np.ndarray = encoder.predict(input_data, batch_size=params.TRANSFORMERS_BATCH_SIZE, verbose="1")
            return embedded

        try:
            # train the model
            with time_measure_result(
                    f'transformers training : ', 
                    params.RESULTS_LOGGER,
                ):
                
                embedded : np.ndarray[Any, Any] = train_model(encoder, input_data, params)


            # split the embedded data into train and test
            colums = [f'embedded_{i}' for i in range(hyperparam.embedding_dim)]
            trained_embedded = pd.DataFrame(embedded[:train_nb_samples], columns=colums)
            assert len(trained_embedded) == len(samples_and_sample_str_train.labels), f"len(trained_embedded)={len(trained_embedded)} != len(samples_and_sample_str_train.labels)={len(samples_and_sample_str_train.labels)}"

            tested_embedded = pd.DataFrame(embedded[train_nb_samples:], columns=colums)
            assert len(tested_embedded) == len(samples_and_sample_str_test.labels), f"len(tested_embedded)={len(tested_embedded)} != len(samples_and_sample_str_test.labels)={len(samples_and_sample_str_test.labels)}"

            trained = SamplesAndLabels(trained_embedded, samples_and_sample_str_train.labels)
            tested = SamplesAndLabels(tested_embedded, samples_and_sample_str_test.labels)

            # save the embedded data

            os.makedirs(embedding_folder, exist_ok=True)

            #save the run hyperparams

            hyperparam.log(params.RESULTS_LOGGER)
            hyperparam.append_to_csv(os.path.join(params.OUTPUT_FOLDER, f"hyperparams.csv"))


            # test the model
            testing_pipeline(params, (trained, tested))
        except TimeoutError:
            params.RESULTS_LOGGER.error(f"Timeout error in transformers pipeline {hyperparam.index}, skipping")
        except MemoryError:
            params.RESULTS_LOGGER.error(f"Memory error in pipeline {hyperparam.index}, skipping")
        except Exception as e:
            params.RESULTS_LOGGER.error(f"Exception in pipeline {hyperparam.index}, skipping: {e}")

    

def __get_input_encoder_from_samplesAndLabels(hyperparam : TransformersHyperParams, df_train: SamplesAndLabels, df_test: SamplesAndLabels):
    df = pd.concat([df_train.sample, df_test.sample])

    output = __transform_hex_data(hyperparam, df=df)
    del df
    output = output[USER_DATA_COLUMN].apply(lambda list_x : [float(int(x, 16)) for x in list_x ]).tolist()
    
    # Pad the sequences
    # first dimension is the number of samples, second is the length of the sequence
    padded_sequences = pad_sequences(output, padding='post', dtype='float64') 
    # Reshape for the encoder like (batch_size, sequence_length, feature_size) (3d)
    padded_sequences = np.expand_dims(padded_sequences, axis=-1)
    return padded_sequences


def __transform_hex_data(hyperparam : TransformersHyperParams, df: pd.DataFrame) -> pd.DataFrame:
    """Transforms the specified column of a DataFrame into lists of 2-byte length strings."""
    transformed_df = df
    transformed_df[USER_DATA_COLUMN] = transformed_df[USER_DATA_COLUMN].apply(lambda x: split_into_chunks(x, hyperparam.word_character_size))
    return transformed_df

def __get_encoder(params: CommonProgramParams, hyperparams: TransformersHyperParams, input_size :int) -> tf.keras.Model:
    def transformer_encoder(units: int, num_heads: int) -> tf.keras.Model:
        # Input
        inputs = layers.Input(shape=(input_size, hyperparams.embedding_dim))
        
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
    inputs = layers.Input(shape=(None, 1))

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


def __get_model_summary(model: tf.keras.Model):
    """
    Captures the summary of a Keras model as a string.
    
    Parameters:
    - model (keras.Model): The Keras model for which the summary is needed.

    Returns:
    - str: A string containing the summary of the model.
    """

    # Create a StringIO object to capture the standard output
    stream = io.StringIO()

    # Redirect the standard output to the StringIO object
    with contextlib.redirect_stdout(stream):
        model.summary()

    # Retrieve the model summary from the StringIO object
    summary_string = stream.getvalue()
    stream.close()

    return summary_string