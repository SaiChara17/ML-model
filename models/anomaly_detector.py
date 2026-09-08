import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_dim):
    """
    Builds a simple Autoencoder for anomaly detection.
    The model tries to reconstruct the input. High reconstruction error indicates an anomaly.
    """
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(16, activation='relu')(input_layer)
    encoded = layers.Dense(8, activation='relu')(encoded)
    
    # Bottleneck
    bottleneck = layers.Dense(4, activation='relu')(encoded)
    
    # Decoder
    decoded = layers.Dense(8, activation='relu')(bottleneck)
    decoded = layers.Dense(16, activation='relu')(decoded)
    output_layer = layers.Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder
