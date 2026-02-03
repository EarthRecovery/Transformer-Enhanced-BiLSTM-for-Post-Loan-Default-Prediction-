from Models.ModelInterface import ModelInterface
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Flatten, Bidirectional, MultiHeadAttention, LayerNormalization, Add
import numpy as np
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dropout
import tensorflow as tf
import os


class BERTModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self.model_name = "BERTModel"
        self.num_layers = 12  # BERT-base typically uses 12 layers
        self.d_model = 64
        self.num_heads = 4
        self.dff = 256
        self.dropout_rate = 0.1

    def transformer_encoder_layer(self, x, num_heads, d_model, dff, dropout_rate):
        # Multi-Head Attention block
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        out1 = Add()([x, attn_output])
        out1 = LayerNormalization(epsilon=1e-6)(out1)

        # Feed Forward Network block
        ffn_output = Dense(dff, activation='relu')(out1)
        ffn_output = Dense(d_model)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        out2 = Add()([out1, ffn_output])
        out2 = LayerNormalization(epsilon=1e-6)(out2)

        return out2

    def build_bert_encoder(self, num_layers, d_model, num_heads, dff, input_shape, dropout_rate):
        inputs = Input(shape=input_shape)  # (batch_size, seq_len, feature_dim)
        x = inputs
        x = Dense(d_model)(x)
        for _ in range(num_layers):
            x = self.transformer_encoder_layer(x, num_heads, d_model, dff, dropout_rate)
        return Model(inputs=inputs, outputs=x)

    def getModel(self, train_data, train_labels):
        inputs = Input(shape=(train_data.shape[1], train_data.shape[-1]))  # (time_steps, feature_dim)

        bert_encoder = self.build_bert_encoder(
            self.num_layers, self.d_model, self.num_heads, self.dff,
            (train_data.shape[1], train_data.shape[-1]), self.dropout_rate
        )

        bert_output = bert_encoder(inputs)

        lstm_output = LSTM(64, return_sequences=False)(bert_output)  # Alternative: GlobalAveragePooling1D()
        outputs = Dense(1, activation='sigmoid')(lstm_output)  # Binary classification output

        model = Model(inputs=inputs, outputs=outputs)
        return model
