import tensorflow as tf
import numpy as np

from model import util


class CharacterLanguageModel(object):
    def __init__(self,
                 X_label,
                 rnn_cell,
                 embedding_matrix):
        self.X_label = X_label
        self.rnn_cell = rnn_cell
        self.embedding_matrix = embedding_matrix
        util.init_all_op(self)

    @util.define_scope("loss")
    def loss_op(self):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit_op[:, -1],
                                                       labels=self.X_label[:, 1:])
        return loss

    @util.define_scope("sample")
    def sample_op(self):
        logits, init_state, out_state = self.logit_op
        sample = tf.multinomial(tf.exp(logits)[:, 0, :], 1)[:, 0]
        return sample, init_state, out_state

    @util.define_scope('logit')
    def logit_op(self):
        init_state = tf.placeholder_with_default(self.rnn_cell.zero_state(tf.shape(self.X_label)[0], tf.float32),
                                               shape=[None, self.rnn_cell.state_size], name='input state')
        sequence_length = util.length(self.X_label)
        outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.embedding_op, sequence_length=sequence_length,
                                           initial_state=init_state,
                                           swap_memory=True)
        logits = tf.contrib.layers.fully_connected(outputs, tf.shape(self.embedding_op)[0], None)
        return logits, init_state, state

    @util.define_scope('embedding')
    def embedding_op(self):
        character_embedding = tf.get_variable('embedding', initializer=self.embedding_matrix, dtype=tf.float32)
        embededing = tf.nn.embedding_lookup(character_embedding, self.X_label)
        return embededing


def build_model(learning_rate, hidden_size, embedding_size, character_size, Model, max_grad_norm):
    X = tf.placeholder(dtype=tf.int32, shape=(None, None), name='X')
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)

    model = Model(X, rnn_cell, np.random.randn(character_size, embedding_size))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients = optimizer.compute_gradients(model.loss_op)
    clipped_gradients = tf.clip_by_global_norm(gradients, max_grad_norm)
    train_op = optimizer.apply_gradients(clipped_gradients)
    return model, train_op
