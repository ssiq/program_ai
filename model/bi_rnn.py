import math
import numpy as np

import tensorflow as tf

from . import util
from common import tf_util


def build_bi_rnn(x, state_size, embedding_matrix, action_num):
    length_of_x, output = bi_rnn(action_num, embedding_matrix, state_size, x)

    output = fill_out_of_length_sequence(length_of_x, output, -math.inf)

    return output


class EmbeddingWrapper(tf.contrib.rnn.EmbeddingWrapper):
    def __init__(self,
                 cell,
                 embedding_classes,
                 embedding_size,
                 output_shape=None,
                 initializer=None,
                 reuse=None):
        super(EmbeddingWrapper, self).__init__(cell,
                 embedding_classes,
                 embedding_size,
                 initializer,
                 reuse)
        self.output_shape = output_shape

    @property
    def output_size(self):
        if self.output_shape is None:
            return self._cell.output_size
        else:
            return self.output_shape

    def call(self, inputs, state):
        """Run the cell on embedded inputs."""

        if self._initializer:
            initializer = self._initializer
        elif tf.get_variable_scope().initializer:
            initializer = tf.get_variable_scope().initializer
        else:
            # Default initializer for embeddings should have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

        if isinstance(state, tuple):
            data_type = state[0].dtype
        else:
            data_type = state.dtype

        embedding = tf.get_variable(
            "embedding", [self._embedding_classes, self._embedding_size],
            initializer=initializer,
            dtype=data_type)
        embedded = tf.nn.embedding_lookup(embedding, tf.reshape(inputs, [-1]))

        output = self._cell(embedded, state)
        if self.output_shape is None:
            return output
        else:
            return tf.contrib.layers.fully_connected(output[0], num_outputs=self.output_shape, activation_fn=None), \
                   output[1]



def bi_rnn(action_num, embedding_matrix, state_size, x):
    embedding_matrix = tf.Variable(initial_value=embedding_matrix, name='embedding', dtype=tf.float32)
    embedding_m = tf.nn.embedding_lookup(embedding_matrix, x)
    # embedding_m = tf.expand_dims(x, axis=2)
    length_of_x = util.length(tf.one_hot(x, embedding_matrix.shape[0], dtype=tf.int32))
    cell_bw = tf.nn.rnn_cell.GRUCell(state_size)
    cell_fw = tf.nn.rnn_cell.GRUCell(state_size)
    # cell_fw = EmbeddingWrapper(tf.nn.rnn_cell.GRUCell(state_size), embedding_classes=embedding_matrix.shape[0],
    #                            embedding_size=embedding_matrix.shape[1],)
    # cell_bw = EmbeddingWrapper(tf.nn.rnn_cell.GRUCell(state_size), embedding_classes=embedding_matrix.shape[0],
    #                            embedding_size=embedding_matrix.shape[1],)
    fw_init_state = cell_fw.zero_state(tf.shape(x)[0], tf.float32)
    bw_init_state = cell_bw.zero_state(tf.shape(x)[0], tf.float32)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                 cell_bw=cell_bw,
                                                 inputs=embedding_m,
                                                 sequence_length=length_of_x,
                                                 initial_state_fw=fw_init_state,
                                                 initial_state_bw=bw_init_state,
                                                 dtype=tf.float32,
                                                 swap_memory=True)
    print(outputs)
    output_fw, output_bw = outputs
    reshape_init_state = lambda t: tf.reshape(t, [util.get_shape(t)[0], 1, util.get_shape(t)[1]])
    output_fw = tf.concat([reshape_init_state(fw_init_state), output_fw], axis=1)
    # output_fw = tf.concat([tf.zeros((util.get_shape(x)[0], action_num), dtype=tf.float32), output_fw], axis=1)
    output_bw = tf.reverse_sequence(output_bw, seq_lengths=length_of_x, seq_axis=1, batch_axis=0)
    output_bw = tf.concat([reshape_init_state(bw_init_state), output_bw], axis=1)
    # output_bw = tf.concat([tf.zeros((util.get_shape(x)[0], action_num), dtype=tf.float32), output_bw], axis=1)
    output_bw = tf.reverse_sequence(output_bw, seq_lengths=tf.add(length_of_x, tf.constant(1, dtype=tf.int32)),
                                    seq_axis=1,
                                    batch_axis=0)
    # embedding_code = tf.concat([reshape_init_state(output_fw[:, 0, :]), reshape_init_state(output_bw[:, 0, :])], axis=2)
    # output_array = tf.TensorArray(dtype=tf.float32, size=util.get_shape(x)[1]*2+1)
    # output_array.write(0, tf.concat([reshape_init_state(output_fw[:, 0, :]), reshape_init_state(output_bw[:, 0, :])], axis=2))
    # i0 = tf.constant(1, dtype=tf.int32, name="i")
    #
    # def cond(i, _):
    #     return tf.less(i, tf.shape(output_bw)[1])
    #
    # def body(i, output_array_t):
    #     h0_f = reshape_init_state(output_fw[:, i, :])
    #     h0_b = reshape_init_state(output_bw[:, i, :])
    #     hm1_b = reshape_init_state(output_bw[:, i - 1, :])
    #     # code = tf.concat([code, tf.concat([h0_f, hm1_b], axis=2)], axis=1)
    #     # code = tf.concat([code, tf.concat([h0_f, h0_b], axis=2)], axis=1)
    #     output_array_t.write(2*i-1, tf.concat([h0_f, hm1_b], axis=2))
    #     output_array_t.write(2*i, tf.concat([h0_f, h0_b], axis=2))
    #     return tf.add(i, 1), output_array_t
    #
    # _, output_array = tf.while_loop(
    #     cond=cond,
    #     body=body,
    #     loop_vars=[i0, output_array],
    #     swap_memory=True,)
    #
    # output_array = output_array.stack()
    # print(util.get_shape(output_array))

    output = tf.concat((output_fw, output_bw), axis=2)
    output_in = tf.concat((output_bw[:, :-1, :], output_fw[:, 1:, :]), axis=2)
    output_in_shape = util.get_shape(output_in)
    output_in = tf.concat((output_in, tf.zeros((output_in_shape[0], 1, output_in_shape[2]), dtype=tf.float32)), axis=1)
    output = tf.concat((output, output_in), axis=2)
    output = tf.reshape(output, (output_in_shape[0], -1, output_in_shape[2]))

    output = tf.contrib.layers.fully_connected(output, num_outputs=action_num, activation_fn=None)
    return length_of_x, output


def fill_out_of_length_sequence(length_of_x, output, fill_number):
    """
    :param length_of_x: [batch] consists of each batch length
    :param output: [batch, time, some one]
    :param fill_number: a scalar
    :return:
    """
    output_shape = util.get_shape(output)
    indices = tf.reshape(tf.range(0, output_shape[1], dtype=tf.int32), (1, -1, 1))
    indices = tf.tile(indices, [output_shape[0], 1, output_shape[2]])
    length_indices = tf.tile(tf.reshape(2 * length_of_x + 1, (-1, 1, 1)), [1, output_shape[1], output_shape[2]])
    output = tf.where(indices < length_indices, output, tf.fill(util.get_shape(output), fill_number))
    output = tf.reshape(output, (output_shape[0], -1))
    return output



class BiRnnClassifyModel(util.Summary):
    def __init__(self,
                 state_size,
                 action_number,
                 character_number,
                 learning_rate):
        super().__init__()
        self.state_size = state_size
        self.action_number = action_number
        self.learning_rate = learning_rate
        self._X_label = tf.placeholder(dtype=tf.int32, shape=(None, None), name="X_label")
        self._Y_label = tf.placeholder(dtype=tf.int32, shape=(None, ), name="y_label")
        self.embedding_matrix = np.random.randn(character_number, 500)
        print("embedding_matrix.shape:{}".format(self.embedding_matrix.shape))
        self._global_step_variable = tf.Variable(0, trainable=False, dtype=tf.int32)
        util.init_all_op(self)
        self._add_summary_scalar("loss", self.loss_op)
        self._add_summary_scalar("accuracy", self.accuracy_op)
        self._add_summary_histogram("softmax", self.softmax_op[0, :])
        self._add_summary_scalar("predict", self.predict_op[0])
        self._add_summary_scalar("label", self._Y_label[0])
        self._add_summary_scalar("predict_and_label_distance",
                                 tf.abs(tf.cast(self.predict_op[0], tf.int32)-self._Y_label[0]))
        self._merge_all()
        init_op = tf.global_variables_initializer()
        tf_util.get_session().run(init_op)
        setattr(self, "train", tf_util.function([self._X_label, self._Y_label], [self.loss_op, self.accuracy_op,
                                                                                 self.train_op]))
        print("summary_op:{}".format(self.summary_op))
        setattr(self, "summary", tf_util.function([self._X_label, self._Y_label], self.summary_op))
        setattr(self, "rnn", tf_util.function([self._X_label], self.rnn_op))
        setattr(self, "accuracy", tf_util.function([self._X_label, self._Y_label], self.accuracy_op))

    @util.define_scope(scope="bi_rnn")
    def rnn_op(self):
        length, output = bi_rnn(self.action_number, self.embedding_matrix, self.state_size, self._X_label)
        return length, tf.reshape(output, (util.get_shape(output)[0], -1))

    @util.define_scope(scope="logit")
    def logit_op(self):
        return tf.reshape(self.rnn_op[1], (util.get_shape(self._X_label)[0], -1))

    @util.define_scope(scope="softmax")
    def softmax_op(self):
        return tf.nn.softmax(self.logit_op)

    @util.define_scope(scope="loss")
    def loss_op(self):
        return tf.losses.sparse_softmax_cross_entropy(self._Y_label, self.logit_op)

    @util.define_scope(scope="predict")
    def predict_op(self):
        return tf.argmax(self.softmax_op, axis=1)

    @util.define_scope(scope="metrics")
    def accuracy_op(self):
        return tf.reduce_mean(tf.cast(tf.equal(self._Y_label, tf.cast(self.predict_op, tf.int32)), tf.float32))

    @util.define_scope(scope="train")
    def train_op(self):
        train = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return tf_util.minimize_and_clip(train, self.loss_op, tf.trainable_variables(), self._global_step_variable)

    @property
    def global_step(self):
        return self._global_step_variable.eval(tf_util.get_session())
