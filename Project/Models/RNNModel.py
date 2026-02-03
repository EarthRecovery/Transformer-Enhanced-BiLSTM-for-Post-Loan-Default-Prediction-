from Models.ModelInterface import ModelInterface
from keras.models import Model
from keras.layers import Input, SimpleRNN, Dense, Flatten
import numpy as np
import os


class RNNModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self.model_name = "RNNModel"

    def getModel(self, train_data, train_labels):
        inputs = Input(shape=(train_data.shape[1], train_data.shape[-1]))
        rnn1 = SimpleRNN(64, return_sequences=True)(inputs)
        rnn2 = SimpleRNN(32, return_sequences=True)(rnn1)  # Use SimpleRNN instead of GRU
        flattened_out = Flatten()(rnn2)  # Flatten to 1D
        output2 = Dense(25, activation='sigmoid')(flattened_out)
        output = Dense(1, activation='sigmoid')(output2)

        model = Model(inputs=inputs, outputs=output)
        return model
