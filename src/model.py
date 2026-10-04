"""
The paper's conv1d + BiLSTM network: an exact Keras replica of the app's buildModel()
(client/src/features/fsl-recognition/category-core.js). Layer order, sizes and
BatchNormalization settings must match so exported weights load in the browser.
"""
import keras
from keras import layers

import config


def build(classes: int) -> keras.Sequential:
    return keras.Sequential([
        keras.Input(shape=(config.FRAMES, config.FEATURES)),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(momentum=0.99, epsilon=1e-4),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(momentum=0.99, epsilon=1e-4),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True), merge_mode="concat"),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False), merge_mode="concat"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(classes, activation="softmax"),
    ])
