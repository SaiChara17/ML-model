import tensorflow as tf
from tensorflow.keras import layers, models

def build_classifier(input_dim, num_classes=3):
    """
    Builds a simple Multi-Layer Perceptron (MLP) for supervised fault classification.
    Since we are using extracted statistical and frequency features (not raw 1D signals), 
    an MLP is appropriate and lightweight.
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
