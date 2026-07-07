# [SỬA ĐỔI]: Xóa bỏ toàn bộ các import của thư viện torch do xung đột hệ thống
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class PE(layers.Layer):
    # [SỬA ĐỔI]: Thêm **kwargs để Keras có thể lưu cấu hình Layer
    def __init__(self, d_model, max_seq_length=500, dropout=0.1, **kwargs):
        super(PE, self).__init__(**kwargs)
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout
        self.dropout = layers.Dropout(dropout)
        
        # [SỬA ĐỔI]: Sử dụng np thay vì mặc định, vì np đã được import ở trên
        pe = np.zeros((max_seq_length, d_model), dtype=float)

        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos/(10000**(2*i/d_model)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/d_model)))
        pe = pe.reshape(1, pe.shape[0], pe.shape[1])      
        
        # [SỬA ĐỔI]: Chuyển numpy array thành hằng số TensorFlow để tối ưu tốc độ Graph
        self.pe = tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        x = x * math.sqrt(self.d_model)
        seq_length = tf.shape(x)[1] 
        
        pe = self.pe[:, :seq_length] 
        x = x + pe
        x = self.dropout(x)
        return x

    def get_config(self):
        config = super(PE, self).get_config()
        config.update({
            "d_model": self.d_model,
            "max_seq_length": self.max_seq_length,
            "dropout": self.dropout_rate
        })
        return config


# [SỬA ĐỔI]: Chuyển đổi toàn bộ lớp GRE từ PyTorch (nn.Module) sang TensorFlow (layers.Layer)
class GRE(layers.Layer):
    def __init__(self, d_model, total_size, K=10, **kwargs):
        super(GRE, self).__init__(**kwargs)
        self.d_model = d_model
        self.total_size = total_size
        self.K = K

    def build(self, input_shape):
        # Khởi tạo ma trận Embedding có thể huấn luyện (thay cho nn.Parameter của Torch)
        self.embedding = self.add_weight(
            shape=(self.K, self.d_model),
            initializer='glorot_uniform',
            trainable=True,
            name="gre_embedding"
        )
        
        # Tạo vị trí cố định
        positions = tf.range(self.total_size, dtype=tf.float32)
        positions = tf.expand_dims(positions, 1)
        self.positions = tf.tile(positions, [1, self.K]) # [total_size, K]

        # Khởi tạo mu và sigma
        interval = self.total_size / self.K
        mu_init = [i * interval for i in range(self.K)]
        
        self.mu = self.add_weight(
            shape=(1, self.K),
            initializer=tf.constant_initializer(mu_init),
            trainable=True,
            name="gre_mu"
        )
        
        self.sigma = self.add_weight(
            shape=(1, self.K),
            initializer=tf.constant_initializer(50.0),
            trainable=True,
            name="gre_sigma"
        )
        super(GRE, self).build(input_shape)

    def call(self, x):
        # Tính hàm mật độ xác suất phân phối chuẩn (Normal PDF)
        a = self.positions - self.mu
        log_p = -1 * tf.math.multiply(a, a) / (2 * self.sigma) - tf.math.log(self.sigma) / 2
        M = tf.nn.softmax(log_p, axis=1)
        
        # Tính Position Encoding
        pos_enc = tf.matmul(M, self.embedding)
        temp = tf.expand_dims(pos_enc, 0)
        return x + temp

    def get_config(self):
        config = super(GRE, self).get_config()
        config.update({
            "d_model": self.d_model,
            "total_size": self.total_size,
            "K": self.K
        })
        return config