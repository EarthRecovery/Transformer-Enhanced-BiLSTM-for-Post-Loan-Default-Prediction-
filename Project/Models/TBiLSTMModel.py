from Models.ModelInterface import ModelInterface
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Flatten, Bidirectional, MultiHeadAttention, LayerNormalization
import numpy as np
import os

class TBiLSTMModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self.model_name = "TBiLSTMModel"

    def getModel(self,train_data, train_labels):
        # Build the BiLSTM model here
        inputs = Input(shape=(train_data.shape[1],train_data.shape[-1]))
        # Multi-Head Self-Attention
        attention_out = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
        attention_out = LayerNormalization(epsilon=1e-6)(attention_out + inputs)
        
        # Feed-Forward Neural Network
        ff_out = Dense(units=256,  activation='relu')(attention_out)
        ff_out = Dense(units=train_data.shape[-1])(ff_out)
        transformer_out = LayerNormalization(epsilon=1e-6)(ff_out + attention_out)
        
        # LSTM layer
        lstm_out = Bidirectional(LSTM(64, return_sequences=True))(transformer_out)
        
        flattened_out = Flatten()(lstm_out)
        output2 = Dense(25, activation='sigmoid')(flattened_out)
        output = Dense(1, activation='sigmoid')(output2)
        
        model = Model(inputs=inputs, outputs=output)
        return model

    
