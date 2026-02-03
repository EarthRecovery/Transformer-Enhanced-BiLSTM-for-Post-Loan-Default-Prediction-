from Models.ModelInterface import ModelInterface
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Flatten, Bidirectional
import numpy as np
import os


class BiLSTMModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self.model_name = "BiLSTMModel"

    def getModel(self, train_data, train_labels):
        inputs = Input(shape=(train_data.shape[1], train_data.shape[-1]))
        bilstm1 = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        bilstm2 = Bidirectional(LSTM(64, return_sequences=True))(bilstm1)
        flattened_out = Flatten()(bilstm2)  # Flatten to 1D
        output2 = Dense(25, activation='sigmoid')(flattened_out)
        output = Dense(1, activation='sigmoid')(output2)

        model = Model(inputs=inputs, outputs=output)
        return model
