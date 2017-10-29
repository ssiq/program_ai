import tensorflow as tf
from tensorflow.contrib.framework import nest

import common.rnn_util
from . import tf_util, util


class RNNWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 cell,
                 reuse=False):
        super().__init__(_reuse=reuse)
        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size


class GatedAttentionWrapper(RNNWrapper):
    def __init__(self,
                 cell: tf.nn.rnn_cell.RNNCell,
                 memory,
                 memory_length,
                 attention_size: int,
                 reuse=False):
        super().__init__(cell, reuse)
        self._memory = memory
        self._memory_length = memory_length
        self._attention_size = attention_size

    def call(self, inputs, state):
        with tf.variable_scope("gated_attention"):
            atten = common.rnn_util.soft_attention_reduce_sum(self._memory, [state, inputs], self._attention_size, self._memory_length)
            atten = nest.flatten(atten)
            inputs = nest.flatten(atten)
            print("atten:{}, input:{}".format(atten, inputs))
            inputs = tf.concat(inputs + atten, axis=1)
            gate_weight = tf.get_variable("gate_weight",
                                          shape=(tf_util.get_shape(inputs)[1], tf_util.get_shape(inputs)[1]),
                                          dtype=tf.float32)
            inputs = inputs * tf.sigmoid(tf.matmul(inputs, gate_weight))
            print("gated inputs:{}".format(inputs))
        return self._cell(inputs, state)


class SelfMatchAttentionWrapper(RNNWrapper):
    def __init__(self,
                 cell: tf.nn.rnn_cell.RNNCell,
                 memory,
                 memory_length,
                 attention_size,
                 reuse=False):
        super().__init__(cell, reuse)
        self._memory = memory
        self._memory_length = memory_length
        self._attention_size = attention_size

    def call(self, inputs, state):
        with tf.variable_scope("self_match_attention"):
            inputs = util.convert_to_list(inputs)
            atten = common.rnn_util.soft_attention_reduce_sum(self._memory,
                                                              [inputs],
                                                              self._attention_size,
                                                              self._memory_length)
            atten = util.convert_to_list(atten)

            print("Self match, input:{}, atten:{}\n, added:{}".format(inputs, atten, inputs+atten))
            inputs = tf.concat(inputs + atten, axis=1)
            inputs = tf_util.weight_multiply("gate_weight", inputs, tf_util.get_shape(inputs)[1])
            return self._cell(inputs, state)