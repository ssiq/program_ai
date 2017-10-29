import tensorflow as tf
import numpy as np
import typing

import common.tf_util


class CharacterLanguageModel(object):
    def __init__(self,
                 X_label,
                 rnn_cell,
                 embedding_matrix):
        self.X_label = X_label
        self.rnn_cell = rnn_cell
        self.one_hot_X = tf.one_hot(X_label, embedding_matrix.shape[0], dtype=tf.float32)
        self.embedding_matrix = embedding_matrix
        common.tf_util.init_all_op(self)

    @common.tf_util.define_scope(scope="loss")
    def loss_op(self):
        logits = self.logit_op[0][:, :-1]
        labels = self.one_hot_X[:, 1:]
        print(common.tf_util.get_shape(logits))
        print(common.tf_util.get_shape(labels))
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=labels)
        loss = tf.reduce_mean(loss)
        return loss

    @common.tf_util.define_scope(scope="sample")
    def sample_op(self):
        logits, init_state, out_state = self.logit_op
        sample = tf.multinomial(tf.exp(logits)[:, 0, :], 1)[:, 0]
        return sample, init_state, out_state

    @common.tf_util.define_scope(scope='logit')
    def logit_op(self):
        default_init_state = self.rnn_cell.zero_state(tf.shape(self.X_label)[0], tf.float32)
        # init_state = [tf.placeholder_with_default(t, shape=(None, None)) for t in default_init_state]
        init_state = tf.placeholder_with_default(default_init_state, shape=(None, common.tf_util.get_shape(default_init_state)[1]))
        sequence_length = common.tf_util.length(self.one_hot_X)
        outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.embedding_op, sequence_length=sequence_length,
                                           initial_state=init_state,
                                           swap_memory=True)
        logits = tf.contrib.layers.fully_connected(outputs, self.embedding_matrix.shape[0], None)
        return logits, init_state, state

    @common.tf_util.define_scope(scope='embedding')
    def embedding_op(self):
        character_embedding = tf.Variable(name='embedding', initial_value=self.embedding_matrix, dtype=tf.float32)
        embededing = tf.nn.embedding_lookup(character_embedding, self.X_label)
        return embededing

    @common.tf_util.define_scope(scope='perplexity')
    def perplexity_op(self):
        return common.tf_util.perplexity(self.logit_op[0], self.one_hot_X)

    @common.tf_util.define_scope(scope='summary')
    def summary_op(self):
        tf.summary.scalar('perplexity',
                          self.perplexity_op)
        tf.summary.scalar('loss', self.loss_op)
        return tf.summary.merge_all()


def build_model(learning_rate: float, hidden_size: int, embedding_size: int, character_size: int, Model: typing.Type,
                max_grad_norm: float):
    X_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name='X')
    rnn_cell = tf.nn.rnn_cell.GRUCell(hidden_size)

    model = Model(X_input, rnn_cell, np.random.randn(character_size, embedding_size))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients = tf.gradients(model.loss_op, tf.trainable_variables())
    global_step = tf.Variable(initial_value=0, name='global_step', dtype=tf.int32, trainable=False)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
    train_op = optimizer.apply_gradients(zip(clipped_gradients, tf.trainable_variables()), global_step=global_step)
    return model, train_op, X_input, global_step
