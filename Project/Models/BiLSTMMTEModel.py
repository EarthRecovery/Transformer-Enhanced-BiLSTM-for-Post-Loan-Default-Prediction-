from Models.ModelInterface import ModelInterface
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Flatten, Bidirectional, MultiHeadAttention, LayerNormalization
import numpy as np
import os


class BiLSTMMTEModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self.model_name = "BiLSTMMTEModel"

    def getModel(self, train_data, train_labels):
        inputs = Input(shape=(train_data.shape[1], train_data.shape[-1]))  # (time_steps, feature_dim)

        # Transformer Encoder layers
        num_transformer_layers = 4  # Stack 4 Transformer Encoder layers
        x = inputs
        for _ in range(num_transformer_layers):
            attention_out = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
            attention_out = LayerNormalization(epsilon=1e-6)(attention_out + x)

            # Feed Forward Network (simple 2-layer Dense)
            ffn_out = Dense(units=256, activation='relu')(attention_out)
            ffn_out = Dense(x.shape[-1])(ffn_out)
            x = LayerNormalization(epsilon=1e-6)(ffn_out + attention_out)

        # BiLSTM layer
        lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)

        # Pooling + Dense for output
        pooled_out = Flatten()(lstm_out)
        output = Dense(1, activation='sigmoid')(pooled_out)  # Binary classification output

        model = Model(inputs=inputs, outputs=output)
        return model
