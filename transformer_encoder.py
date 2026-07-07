import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import GlorotUniform
import numpy as np

class LayerNorm(layers.Layer):
    def __init__(self, features, eps=1e-6, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.features = features
        self.eps = eps
    
    def build(self, input_shape):
        # [SỬA ĐỔI]: np.ones tĩnh sẽ không thể huấn luyện. Phải dùng add_weight để mô hình học được tham số.
        self.a_2 = self.add_weight(shape=(self.features,), initializer='ones', trainable=True)
        self.b_2 = self.add_weight(shape=(self.features,), initializer='zeros', trainable=True)
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

class Encoder(layers.Layer):
    def __init__(self, layer, N, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        # [SỬA ĐỔI]: Tránh đặt tên self.layers trùng với thuộc tính nội bộ của Keras
        self.layers_list = [layer for _ in range(N)]
        self.N = N
        self.norm = LayerNorm(500)
        
    def call(self, x, mask=None):
        for layer in self.layers_list:
            x = layer(x, mask)
        return self.norm(x)

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"N": self.N})
        return config

class SublayerConnection(layers.Layer):
    def __init__(self, size, dropout, **kwargs):
        super(SublayerConnection, self).__init__(**kwargs)
        self.size = size
        self.dropout_rate = dropout
        self.norm = LayerNorm(size)
        self.dropout = layers.Dropout(dropout)

    # [SỬA ĐỔI]: Đổi tên hàm từ forward (chuẩn PyTorch) sang call (chuẩn Keras)
    def call(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

    def get_config(self):
        config = super(SublayerConnection, self).get_config()
        config.update({"size": self.size, "dropout": self.dropout_rate})
        return config

class EncoderLayer(layers.Layer):
    def __init__(self, size, self_attn, feed_forward, dropout, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size
        self.dropout_rate = dropout
        self.sublayer1 = SublayerConnection(size, dropout)
        self.sublayer2 = SublayerConnection(size, dropout)

    # [SỬA ĐỔI]: Sửa lỗi 'forward' thành 'call'. Sửa luôn lỗi tác giả gọi nhầm self.sublayer[0] vốn không tồn tại.
    def call(self, x, mask=None):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer2(x, self.feed_forward)

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({"size": self.size, "dropout": self.dropout_rate})
        return config

def attention(query, key, value, mask=None, dropout=None):
    d_k = tf.shape(query)[-1]
    scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.cast(d_k, tf.float32))
    if mask is not None:
        scores = tf.where(tf.equal(mask, 0), tf.fill(tf.shape(scores), -1e9), scores)
    p_attn = tf.nn.softmax(scores, axis=-1)
    if dropout is not None:
        # [SỬA ĐỔI]: Sử dụng layer dropout được truyền vào thay vì khởi tạo ngay trong hàm
        p_attn = dropout(p_attn)
    return tf.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(layers.Layer):
    def __init__(self, d_model, d_ff, dropout=0.1, **kwargs):
        super(PositionwiseFeedForward, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.w_1 = layers.Dense(d_ff, activation='relu', kernel_initializer=GlorotUniform())
        self.w_2 = layers.Dense(d_model, kernel_initializer=GlorotUniform())
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        x = self.w_1(x)
        x = self.dropout(x, training=training)
        return self.w_2(x)

    def get_config(self):
        config = super(PositionwiseFeedForward, self).get_config()
        config.update({"d_model": self.d_model, "d_ff": self.d_ff, "dropout": self.dropout_rate})
        return config

class MultiHeadedAttention(layers.Layer):
    def __init__(self, h, d_model, dropout=0.1, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.d_model = d_model
        self.dropout_rate = dropout
        self.linears = [layers.Dense(d_model) for _ in range(4)]
        self.attn = None
        self.dropout = layers.Dropout(dropout)

    # [SỬA ĐỔI]: Đổi 'forward' thành 'call'
    def call(self, query, key, value, mask=None):
        if mask is not None:
            mask = tf.expand_dims(mask, axis=1)
        nbatches = tf.shape(query)[0]

        query, key, value = [
            tf.transpose(tf.reshape(l(x), (nbatches, -1, self.h, self.d_k)), perm=[0, 2, 1, 3])
            for l, x in zip(self.linears, (query, key, value))
        ]

        # [SỬA ĐỔI]: Sửa lỗi self.attention (vì attention là hàm cục bộ bên ngoài lớp)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), (nbatches, -1, self.h * self.d_k))
        return self.linears[-1](x)

    def get_config(self):
        config = super(MultiHeadedAttention, self).get_config()
        config.update({"h": self.h, "d_model": self.d_model, "dropout": self.dropout_rate})
        return config

class Transfomer(layers.Layer):
    def __init__(self, hidden_dim, N, H, **kwargs):
        super(Transfomer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.N = N
        self.H = H
        self.model = Encoder(
            EncoderLayer(hidden_dim, MultiHeadedAttention(H, hidden_dim),
            PositionwiseFeedForward(hidden_dim, hidden_dim*4),
            0.1
            ),
            N
        )
        
    # [SỬA ĐỔI]: Đổi 'forward' thành 'call' và bổ sung return
    def call(self, x, mask=None):
        return self.model(x, mask)

    def get_config(self):
        config = super(Transfomer, self).get_config()
        config.update({"hidden_dim": self.hidden_dim, "N": self.N, "H": self.H})
        return config