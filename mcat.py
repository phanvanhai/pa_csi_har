import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import add, concatenate, TimeDistributed, Concatenate
import numpy as np

import math, copy, time

class Encoder(layers.Layer):
    def __init__(self, layer, N, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.layers_list = [] 
        self.N = N
        for i in range(N):
            self.layers_list.append(layer)
        # [SỬA ĐỔI]: Để trống LayerNorm để tự động nhận kích thước đầu ra thay vì truyền layer.size cố định
        self.norm = LayerNorm()
    
    def call(self, x, mask=None):
        for layer in self.layers_list:
            x = layer(x)
        return self.norm(x)

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"N": self.N})
        return config

class LayerNorm(layers.Layer):
    def __init__(self, features=None, eps=1e-6, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.features = features
        self.eps = eps
        
    def build(self, input_shape):
        # [SỬA ĐỔI]: Tự động lấy chiều cuối cùng của input_shape nếu features=None
        if self.features is None:
            self.features = input_shape[-1]
        self.a_2 = self.add_weight(shape=(self.features,), initializer='ones', trainable=True, name="gamma")
        self.b_2 = self.add_weight(shape=(self.features,), initializer='zeros', trainable=True, name="beta")
        super(LayerNorm, self).build(input_shape)
    
    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        out = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return out

    def get_config(self):
        config = super(LayerNorm, self).get_config()
        config.update({"features": self.features, "eps": self.eps})
        return config

class EncoderLayer(layers.Layer):
    def __init__(self, size, self_attt, feed_forward, dropout, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.self_attt = self_attt
        self.feed_forward = feed_forward
        self.size = size
        self.dropout_rate = dropout
        
        self.norm1 = LayerNorm(size)
        self.drop1 = layers.Dropout(dropout)
        self.norm2 = LayerNorm(size)
        self.drop2 = layers.Dropout(dropout)
        
    def call(self, x, mask=None):
        # 1. Sublayer: Self-Attention
        nx = self.norm1(x)
        attn_out = self.self_attt(nx, nx, nx, mask=mask)
        x = x + self.drop1(attn_out)

        # 2. Sublayer: Feed Forward
        nx2 = self.norm2(x)
        ff_out = self.feed_forward(nx2)
        x = x + self.drop2(ff_out)
        
        return x

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({"size": self.size, "dropout": self.dropout_rate})
        return config

class HAR_CNN(layers.Layer):
    def __init__(self, d_model, d_ff, filters, dropout=0.2, **kwargs):
        super(HAR_CNN, self).__init__(**kwargs)
        self.kernel_num = int(d_ff)
        self.filter_sizes = filters
        self.dropout_rate = dropout
        self.dropout = layers.Dropout(dropout)
        self.bn = layers.BatchNormalization(axis=1)
        self.relu = layers.Activation('relu')
        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            encoder = layers.Conv1D(filters=self.kernel_num, 
                                    kernel_size=filter_size,
                                    data_format='channels_first',
                                    padding='same'
                                   )
            self.encoders.append(encoder)

    def call(self, data):
        data = tf.cast(data, dtype=tf.float32)
        enc_outs = []
        for encoder in self.encoders:
            f_map = encoder(tf.transpose(data, perm=[0, 2, 1]))
            enc_ = f_map
            enc_ = self.relu(self.dropout(self.bn(enc_)))
            enc_outs.append(tf.expand_dims(enc_, axis=1))

        re = tf.divide(tf.reduce_sum(tf.concat(enc_outs, axis=1), axis=1), 3)
        return tf.transpose(re, perm=[0, 2, 1])

    def get_config(self):
        config = super(HAR_CNN, self).get_config()
        config.update({
            "d_model": self.kernel_num,
            "d_ff": self.kernel_num,
            "filters": self.filter_sizes,
            "dropout": self.dropout_rate
        })
        return config

def attention(query, key, value, mask=None, dropout=None):
    d_k = tf.cast(tf.shape(query)[-1], tf.float32)
    scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(d_k)

    if mask is not None:
        scores += (mask * -1e9)

    p_attn = tf.keras.activations.softmax(scores, axis=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return tf.matmul(p_attn, value), p_attn

class MultiHeadAttention(layers.Layer):
    def __init__(self, h, d_model, dropout=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.d_k = d_model // h
        self.h = h
        self.d_model = d_model
        self.dropout_rate = dropout
        self.linears = [layers.Dense(d_model) for _ in range(4)]
        self.att = None
        self.dropout = layers.Dropout(dropout)

    def call(self, query, key, value, mask=None):
        if mask is not None:
            mask = tf.expand_dims(mask, 1)
        nbatches = tf.shape(query)[0]

        query, key, value = [
            tf.transpose(tf.reshape(l(x), (nbatches, -1, self.h, self.d_k)), perm=[0, 2, 1, 3])
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask)

        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), (nbatches, -1, self.h * self.d_k))
        return self.linears[-1](x)

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            "h": self.h,
            "d_model": self.d_model,
            "dropout": self.dropout_rate
        })
        return config

class MCAT(layers.Layer):
    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5], **kwargs):
        super(MCAT, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.N = N
        self.H = H
        self.total_size = total_size
        self.filters = filters

        self.model = Encoder(
            EncoderLayer(hidden_dim, MultiHeadAttention(H, hidden_dim),
                         HAR_CNN(hidden_dim, hidden_dim, filters)
                         , 0.1),
            N
        )

    def call(self, x):
        return self.model(x)

    def get_config(self):
        config = super(MCAT, self).get_config()
        config.update({
            "hidden_dim": self.hidden_dim, "N": self.N, 
            "H": self.H, "total_size": self.total_size, "filters": self.filters
        })
        return config