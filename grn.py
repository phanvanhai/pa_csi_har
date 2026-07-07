import tensorflow as tf
from tensorflow.keras import layers

class GatesResidualNetwork(layers.Layer):
    def __init__(self, units, dropout=0.2, **kwargs):
        super(GatesResidualNetwork, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout
        self.dense_1 = layers.Dense(units, activation=None)
        self.dense_2 = layers.Dense(units, activation=None)
        self.elu = layers.Activation('elu')
        self.gate_dense = layers.Dense(units, activation='sigmoid')
        self.add = layers.Add()
        self.dropout = layers.Dropout(dropout)
        self.layer_norm = layers.LayerNormalization()
        
    def call(self, inputs, context=None):
        x = self.dense_1(inputs)
        x = self.elu(x)
        if context is not None:
            context_transform = self.dense_2(context)
            context_transform = self.elu(context_transform)
            x = self.add([x, context_transform])
            x = self.layer_norm(x)
            x = self.gate_dense(x)
        return x 
        
    # [SỬA ĐỔI]: Thêm hàm get_config để lớp có thể tuần tự hóa
    def get_config(self):
        config = super(GatesResidualNetwork, self).get_config()
        config.update({
            "units": self.units,
            "dropout": self.dropout_rate
        })
        return config