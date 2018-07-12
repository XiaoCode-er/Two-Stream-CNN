from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class Transformer(Layer):

    def __init__(self, d_k, frames, **kwargs):
        # d_k是经过transformer的输出维度
        self.d_k = d_k
        # self.scalar = np.sqrt(self.d_k)
        self.frames = frames
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='transformer',
                                 shape=[input_shape[2], self.d_k],
                                 initializer='glorot_uniform',
                                 trainable=True)
        super().build(input_shape)

    def call(self, inputs):

        inputs = K.permute_dimensions(inputs, (0, 3, 1, 2))
        inputs = K.reshape(inputs, (-1, self.frames*3, int(inputs.shape[3])))
        inputs = K.dot(inputs, self.W)
        inputs = K.reshape(inputs, (-1, 3, self.frames, self.d_k))
        inputs = K.permute_dimensions(inputs, (0, 2, 3, 1))
        # q, k, v = inputs
        # A = K.batch_dot(inputs, inputs, axes=[3, 3])/self.scalar
        # A = K.softmax(A)
        # A = K.batch_dot(A, inputs, axes=[3, 3])
        # A = K.permute_dimensions(A, (0, 2, 3, 1))
        return inputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_k, input_shape[3])