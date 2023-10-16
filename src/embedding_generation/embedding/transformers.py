import tensorflow as tf
from tensorflow.keras import layers, models

from commons.data_loading.data_types import SamplesAndLabels
from embedding_generation.params.params import ProgramParams


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
    encoder = __get_encoder()
    


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
