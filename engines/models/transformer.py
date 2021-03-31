# -*- coding: utf-8 -*-
# @Time : 2021/3/30 22:56 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : transformer.py
# @Software: PyCharm
from abc import ABC
import tensorflow as tf
import numpy as np
from config import classifier_config


class PositionalEncoding:

    def __init__(self, embedding_dim, seq_length):
        super(PositionalEncoding, self).__init__()
        self.pe = np.array([[pos / np.power(10000, 2 * (i // 2) / embedding_dim)
                             for i in range(embedding_dim)] for pos in range(seq_length)])
        self.pe[1:, 0::2] = np.sin(self.pe[1:, 0::2])
        self.pe[1:, 1::2] = np.cos(self.pe[1:, 1::2])

    @tf.function
    def call(self, inputs):
        position_embed = inputs + self.pe
        return position_embed


class MultiHeadAttention:
    def __init__(self, embedding_dim, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.head_num = classifier_config['head_num']
        self.batch_size = classifier_config['batch_size']
        self.head_dim = embedding_dim // self.head_num
        self.W_Q = tf.keras.layers.Dense(self.head_dim * self.head_num, use_bias=False)
        self.W_K = tf.keras.layers.Dense(self.head_dim * self.head_num, use_bias=False)
        self.W_V = tf.keras.layers.Dense(self.head_dim * self.head_num, use_bias=False)
        self.W_O = tf.keras.layers.Dense(self.head_dim * self.head_num, use_bias=False)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def scaled_dot_product_attention(self, query, key, value):
        scale = self.head_dim ** -0.5
        key = tf.transpose(key, [0, 2, 1])
        attention = tf.matmul(query, key)
        attention = scale * attention
        attention = tf.nn.softmax(attention)
        attention = tf.matmul(attention, value)
        return attention

    # @tf.function
    def call(self, inputs):
        query = self.W_Q(inputs)
        key = self.W_K(inputs)
        value = self.W_V(inputs)

        query = tf.reshape(query, [self.batch_size * self.head_num, -1, self.head_dim])
        key = tf.reshape(key, [self.batch_size * self.head_num, -1, self.head_dim])
        value = tf.reshape(value, [self.batch_size * self.head_num, -1, self.head_dim])
        z = self.scaled_dot_product_attention(query, key, value)
        z = tf.reshape(z, [self.batch_size, -1, self.head_dim * self.head_num])
        z = self.W_O(z)
        dropout_output = self.dropout(z)
        output = dropout_output + inputs
        output = self.layer_norm(output)
        return output


class FeedForward:
    def __init__(self, embedding_dim, dropout_rate):
        super(FeedForward, self).__init__()
        hidden_dim = classifier_config['hidden_dim']
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu', use_bias=False)
        self.dense2 = tf.keras.layers.Dense(embedding_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    # @tf.function
    def call(self, inputs):
        output = self.dense1(inputs)
        output = self.dense2(output)
        dropout_output = self.dropout(output)
        output = dropout_output + inputs
        output = self.layer_norm(output)
        return output


class Encoder:

    def __init__(self, embedding_dim, dropout_rate):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(embedding_dim, dropout_rate)
        self.feed_forward = FeedForward(embedding_dim, dropout_rate)

    # @tf.function
    def call(self, inputs):
        attention_outputs = self.attention.call(inputs)
        output = self.feed_forward.call(attention_outputs)
        return output


class Transformer(tf.keras.Model, ABC):
    """
    Transformer模型
    """

    def __init__(self, vocab_size, embedding_dim, seq_length):
        super(Transformer, self).__init__()
        dropout_rate = classifier_config['dropout_rate']
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.positional_encoder = PositionalEncoding(embedding_dim, seq_length)
        self.encoder = Encoder(embedding_dim, dropout_rate)

    # @tf.function
    def call(self, inputs, training=None):
        embed_inputs = self.embedding(inputs)
        position_embed = self.positional_encoder.call(embed_inputs)
        attention_output = self.encoder.call(position_embed)
