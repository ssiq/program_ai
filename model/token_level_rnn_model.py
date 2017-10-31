import tensorflow as tf
import more_itertools
import cytoolz as toolz
from tensorflow.python.util import nest

from common import tf_util, rnn_cell, rnn_util

def _rnn_cell(hidden_size):
    return tf.nn.rnn_cell.GRUCell(hidden_size)


def _multi_rnn_cell(hidden_size, layer_number):
    return tf.nn.rnn_cell.MultiRNNCell([_rnn_cell(hidden_size) for _ in range(layer_number)])


class OutputAttentionWrapper(rnn_cell.RNNWrapper):
    def __init__(self,
                 cell: tf.nn.rnn_cell.RNNCell,
                 memory,
                 memory_length,
                 projection_size,
                 attention_size: int,
                 reuse=False):
        """
        :param cell: a tensorflow cell object
        :param memory: a tensor [batch, time, dim]
        :param memory_length: [batch, ]
        :param attention_size: int number
        :param projection_size: map the output shape to the embedding_size
        :param reuse:
        """
        super().__init__(cell, reuse)
        self._memory = memory
        self._memory_length = memory_length
        self._attention_size = attention_size
        self._projection_size = projection_size

    def call(self, inputs, state):
        attention_logit = rnn_util.soft_attention_logit(self._attention_size, inputs, self._memory, self._memory_length)
        attention_softmax = tf.nn.softmax(attention_logit)
        atten = rnn_util.reduce_sum_with_attention_softmax(self._memory, attention_softmax)
        atten = nest.flatten(atten)
        inputs = nest.flatten(atten)
        print("atten:{}, input:{}".format(atten, inputs))
        inputs = tf.concat(inputs + atten, axis=1)
        gate_weight = tf.get_variable("gate_weight",
                                      shape=(tf_util.get_shape(inputs)[1], tf_util.get_shape(inputs)[1]),
                                      dtype=tf.float32)
        cell_inputs = inputs * tf.sigmoid(tf.matmul(inputs, gate_weight))
        outputs, next_hidden_state = self._cell(cell_inputs, state)
        # The next line will concat all useful things to use in the nested character-level decoder
        attentioned_memory = tf.gather_nd(self._memory,
                                          tf.concat((tf.expand_dims(tf.range(0, limit=tf_util.get_shape(self._memory)[0]), axis=1),
                                                     tf.expand_dims(tf.argmax(attention_softmax, axis=1), axis=1)), axis=1),
                                          )
        outputs = tf.concat((
            attentioned_memory,
            cell_inputs,
            tf_util.weight_multiply("embedding_map", outputs, self._projection_size)
        ), axis=1)

        return outputs, next_hidden_state



class TokenLevelRnnModel(tf_util.Summary):
    def __init__(self,
                 word_embedding_layer_fn,
                 character_embedding_layer_fn,
                 hidden_size,
                 rnn_layer_number,
                 ):
        super().__init__()
        self.word_embedding_layer_fn = word_embedding_layer_fn
        self.character_embedding_layer_fn = character_embedding_layer_fn
        self.hidden_state_size = hidden_size
        self.rnn_layer_number = rnn_layer_number
        self.token_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name="token_input")
        self.token_input_length = tf.placeholder(dtype=tf.int32, shape=(None,), name="token_input_length")
        self.character_input = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name="character_input")
        self.character_input_length = tf.placeholder(dtype=tf.int32, shape=(None, None), name="character_input_length")

    @tf_util.define_scope("batch_size")
    def batch_size_op(self):
        return tf_util.get_shape(self.token_input)[0]

    @tf_util.define_scope("word_embedding_op")
    def word_embedding_op(self):
        return self.word_embedding_layer_fn(self.token_input)

    @tf_util.define_scope("character_embedding_op")
    def character_embedding_op(self):
        return self.character_embedding_layer_fn(self.character_input, self.character_input_length)

    @tf_util.define_scope("code_embedding_op")
    def code_embedding_op(self):
        return tf.concat([self.word_embedding_op, self.character_embedding_op], axis=2)

    @tf_util.define_scope("bi_gru_encode_op")
    def bi_gru_encode_op(self):
        return rnn_util.concat_bi_rnn_output(rnn_util.bi_rnn(lambda: _multi_rnn_cell(self.hidden_state_size,
                                                                                     self.rnn_layer_number),
                                                             self.code_embedding_op,
                                                             self.token_input_length))

    @tf_util.define_scope("result_initial_state")
    def result_initial_state_op(self):
        cell = self.decode_cell_op
        return nest.map_structure(lambda x: tf.Variable(initial_value=x, ),
                                  cell.zero_state(self.batch_size_op, tf.float32))

    @tf_util.define_scope("decode_cell")
    def decode_cell_op(self):
        pass

    @tf_util.define_scope("gru_decode_op")
    def gru_decode(self):
        pass

    @tf_util.define_scope("loss_op")
    def loss_op(self):
        pass

    @tf_util.define_scope("train_op")
    def train_op(self):
        pass
