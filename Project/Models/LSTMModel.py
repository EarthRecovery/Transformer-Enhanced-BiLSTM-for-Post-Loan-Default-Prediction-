from Models.ModelInterface import ModelInterface
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Flatten
import numpy as np
import os


class LSTMModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self.model_name = "LSTMModel"

    def getModel(self, train_data, train_labels):
        inputs = Input(shape=(train_data.shape[1], train_data.shape[-1]))
        lstm1 = LSTM(128, return_sequences=True)(inputs)
        lstm2 = LSTM(64, return_sequences=True)(lstm1)
        flattened_out = Flatten()(lstm2)  # Flatten to 1D
        output2 = Dense(25, activation='sigmoid')(flattened_out)
        output = Dense(1, activation='sigmoid')(output2)

        model = Model(inputs=inputs, outputs=output)
        return model
