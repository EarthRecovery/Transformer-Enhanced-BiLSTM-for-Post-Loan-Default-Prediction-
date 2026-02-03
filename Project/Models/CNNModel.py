from Models.ModelInterface import ModelInterface
from keras.models import Model
from keras.layers import Input, SimpleRNN, Dense, Flatten, Normalization
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import GlobalMaxPooling1D
import numpy as np
import os

class CNNModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self.model_name = "CNNModel"

    def getModel(self,train_data, train_labels):
        inputs = Input(shape=(train_data.shape[1],train_data.shape[-1]))
        
        conv2 = Conv1D(32, kernel_size=3, activation='relu')(inputs)
        maxpool2 = GlobalMaxPooling1D()(conv2)
        
        flattened_out = Flatten()(maxpool2)
        
        output2 = Dense(25, activation='sigmoid')(flattened_out)
        output = Dense(1, activation='sigmoid')(output2)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model
