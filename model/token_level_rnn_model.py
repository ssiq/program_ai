import tensorflow as tf
from tensorflow.python.util import nest

from common import tf_util, rnn_cell, rnn_util, code_util
from common.rnn_util import create_decoder_initialize_fn, create_decode


def _rnn_cell(hidden_size):
    return tf.nn.rnn_cell.GRUCell(hidden_size)


def _multi_rnn_cell(hidden_size, layer_number):
    return tf.nn.rnn_cell.MultiRNNCell([_rnn_cell(hidden_size) for _ in range(layer_number)])


class OutputAttentionWrapper(rnn_cell.RNNWrapper):
    def __init__(self,
                 cell: tf.nn.rnn_cell.RNNCell,
                 memory,
                 memory_length,
                 position_embedding,
                 position_length,
                 projection_size,
                 attention_size: int,
                 reuse=False):
        """
        :param cell: a tensorflow cell object
        :param memory: a tuple with length 2 containing tensor [batch, time, dim]
        :param memory_length: [batch, ]
        :param attention_size: int number
        :param projection_size: map the output shape to the embedding_size
        :param reuse:
        """
        super().__init__(cell, reuse)
        self._memory = memory
        self._memory_length = memory_length
        self._position_embedding = position_embedding
        self._position_length = position_length
        self._attention_size = attention_size
        self._projection_size = projection_size

    def build(self, _):
        self.built = True
        self._cell.built = True

    @property
    def output_size(self):
        """
        :return: (whether copy, projection_size, position logit size, copy logit size)
        """
        return tuple([
            tf_util.get_shape(self._position_embedding)[1],
            1,
            self._projection_size,
            tf_util.get_shape(self._memory)[1]
        ])

    def call(self, inputs, state):
        with tf.variable_scope("input_attention"):
            atten = rnn_util.soft_attention_reduce_sum(self._memory,
                                                       inputs,
                                                       self._attention_size,
                                                       self._memory_length)
        atten = nest.flatten(atten)
        inputs = nest.flatten(atten)
        print("atten:{}, input:{}".format(atten, inputs))
        inputs = tf.concat(inputs + atten, axis=1)
        gate_weight = tf.get_variable("gate_weight",
                                      shape=(tf_util.get_shape(inputs)[1], tf_util.get_shape(inputs)[1]),
                                      dtype=tf.float32)
        cell_inputs = inputs * tf.sigmoid(tf.matmul(inputs, gate_weight))
        outputs, next_hidden_state = self._cell(cell_inputs, state)
        #position_logit
        with tf.variable_scope("poistion_logit"):
            position_logit = rnn_util.soft_attention_logit(self._attention_size,
                                                           outputs,
                                                           self._position_embedding,
                                                           self._position_length)
        position_softmax = tf.nn.softmax(position_logit)
        replace_input = rnn_util.reduce_sum_with_attention_softmax(self._position_embedding,
                                                                   position_softmax)
        replace_ouput = tf_util.weight_multiply("replace_output_weight", replace_input, self._attention_size)
        #a scalar indicating whether copies from the code
        is_copy_logit = tf_util.weight_multiply("copy_weight", replace_ouput, 1)
        #key_word_logit
        key_word_logit = tf_util.weight_multiply("key_word_logit_weight", replace_ouput, self._projection_size)
        #copy_word_logit
        with tf.variable_scope("copy_word_logit"):
            copy_word_logit = rnn_util.soft_attention_logit(self._attention_size, replace_ouput, self._memory, self._memory_length)

        return (position_logit, is_copy_logit, key_word_logit, copy_word_logit,), next_hidden_state


def create_sample_fn():
    def sample_fn(time, outputs, state):
        """Returns `sample_ids`."""
        position_logit, is_copy_logit, key_word_logit, copy_word_logit = outputs
        position = tf.argmax(position_logit, axis=1)
        is_copy = tf.greater(tf.nn.sigmoid(is_copy_logit), 0.5)
        keyword_id = tf.argmax(key_word_logit, axis=1)
        copy_word_id = tf.argmax(key_word_logit, axis=1)
        return position, is_copy, keyword_id, copy_word_id
    return sample_fn


def compute_next_input(copy_word_id,
                       is_copy,
                       keyword_id,
                       position,
                       position_embedding,
                       embedding_fn,
                       copy_word_embedding,
                       batch_size):
    def gather_postion(param, indices):
        return tf.gather_nd(param,
                            tf.concat(
                                (tf.expand_dims(tf.range(0, batch_size), axis=1),
                                 tf.expand_dims(indices, axis=1)),
                                axis=1)
                            )

    is_copy = tf.cast(is_copy, dtype=tf.bool)
    next_position_embedding = gather_postion(position_embedding, position)
    next_keyword_embedding = embedding_fn(keyword_id)
    next_copy_word_embedding = gather_postion(copy_word_embedding, copy_word_id)
    next_input = tf.concat(
        (next_position_embedding,
         tf.where(is_copy, x=next_copy_word_embedding, y=next_keyword_embedding)),
        axis=1
    )
    return next_input

def create_next_input_fn(position_embedding,
                         embedding_fn,
                         copy_word_embedding,
                         endtoken_id,
                         batch_size):
    def next_input_fn(time, outputs, state, sample_ids):
        """Returns `(finished, next_inputs, next_state)`."""
        position, is_copy, keyword_id, copy_word_id = sample_ids
        finished = is_copy
        finished = tf.logical_and(finished, tf.equal(keyword_id, endtoken_id))
        next_input = compute_next_input(copy_word_id, is_copy, keyword_id, position, position_embedding,
                                        embedding_fn, copy_word_embedding, batch_size)
        return finished, next_input, state



    return next_input_fn

def create_train_next_input_fn(output_label, #a tuple (position, is_copy, key_word_index, copy_word_index)
                               output_length,
                               embedding_fn,
                               position_embedding,
                               copy_word_embedding,
                               batch_size):
    def next_input_fn(time, outputs, state, sample_ids):
        """Returns `(finished, next_inputs, next_state)`."""
        finished = tf.logical_not(tf.less(time, output_length))
        position, is_copy, keyword_id, copy_word_id = map(lambda x:x[time], output_label)
        next_input = compute_next_input(copy_word_id, is_copy, keyword_id, position, position_embedding,
                                        embedding_fn, copy_word_embedding, batch_size)
        return finished, next_input, state


    return next_input_fn


class TokenLevelRnnModel(tf_util.Summary):
    def __init__(self,
                 word_embedding_layer_fn,
                 character_embedding_layer_fn,
                 hidden_size,
                 rnn_layer_number,
                 keyword_number,
                 end_token_id,
                 learning_rate,
                 max_decode_iterator_num,
                 ):
        super().__init__()
        self.word_embedding_layer_fn = word_embedding_layer_fn
        self.character_embedding_layer_fn = character_embedding_layer_fn
        self.hidden_state_size = hidden_size
        self.rnn_layer_number = rnn_layer_number
        self.keyword_number = keyword_number
        self.end_token_id = end_token_id
        self.max_decode_iterator_num = max_decode_iterator_num
        self.learning_rate = learning_rate
        self.token_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name="token_input")
        self.token_input_length = tf.placeholder(dtype=tf.int32, shape=(None,), name="token_input_length")
        self.character_input = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name="character_input")
        self.character_input_length = tf.placeholder(dtype=tf.int32, shape=(None, None), name="character_input_length")
        self.output_length = tf.placeholder(dtype=tf.int32, shape=(None, ), name="output_length")
        self.output_position_label = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_position")
        self.output_is_copy = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_is_copy") #1 means using copy
        self.output_keyword_id = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_keyword_id")
        self.output_copy_word_id = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_copy_word_id")

        #the function
        self.train = tf_util.function(
            [self.token_input,
             self.token_input_length,
             self.character_input,
             self.character_input_length,
             self.output_length,
             self.output_position_label,
             self.output_is_copy,
             self.output_keyword_id,
             self.output_copy_word_id],
            [self.loss_op, self.loss_op, self.train_op])


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
        return rnn_util.bi_rnn(lambda: _multi_rnn_cell(self.hidden_state_size,
                                                       self.rnn_layer_number),
                               self.code_embedding_op,
                               self.token_input_length)[0]

    @tf_util.define_scope("position_embedding")
    def position_embedding_op(self):
        """
        :return: (position_embedding tensor)
        """
        return code_util.position_embedding(self.bi_gru_encode_op[0], self.bi_gru_encode_op[1], self.token_input_length)

    @tf_util.define_scope("position_length")
    def position_length_op(self):
        return self.token_input_length * 2 + 1

    @tf_util.define_scope("result_initial_state")
    def result_initial_state_op(self):
        cell = self.decode_cell_op
        return nest.map_structure(lambda x: tf.Variable(initial_value=x, ),
                                  cell.zero_state(self.batch_size_op, tf.float32))

    @tf_util.define_scope("start_label")
    def start_label_op(self):
        return tf.zeros(
            shape=(tf_util.get_shape(self.code_embedding_op)[-1] + tf_util.get_shape(self.position_embedding_op)[-1]),
            dtype=tf.float32)

    @tf_util.define_scope("decode_cell")
    def decode_cell_op(self):
        return OutputAttentionWrapper(_rnn_cell(self.hidden_state_size),
                                      self.code_embedding_op,
                                      self.token_input_length,
                                      self.position_embedding_op,
                                      self.position_length_op,
                                      self.keyword_number,
                                      self.hidden_state_size)

    @tf_util.define_scope("decode")
    def decode_op(self):
        initialize_fn = create_decoder_initialize_fn(self.start_label_op, self.batch_size_op)
        sample_fn = create_sample_fn()
        sample_next_input_fn = create_next_input_fn(self.position_embedding_op,
                                                      self.word_embedding_layer_fn,
                                                      self.code_embedding_op,
                                                      self.end_token_id,
                                                      self.batch_size_op)

        training_next_input_fn  = create_train_next_input_fn((self.output_position_label,
                                                              self.output_is_copy,
                                                              self.output_keyword_id,
                                                              self.output_copy_word_id), #(position, is_copy, key_word_index, copy_word_index)
                                                             self.output_length,
                                                             self.word_embedding_layer_fn,
                                                             self.position_embedding_op,
                                                             self.code_embedding_op,
                                                             self.batch_size_op)
        return create_decode((initialize_fn, sample_fn, sample_next_input_fn),
                             (initialize_fn, sample_fn, training_next_input_fn), self.decode_cell_op,
                             self.result_initial_state_op, self.max_decode_iterator_num)

    @tf_util.define_scope("gru_decode_op")
    def gru_decode_op(self):
        return self.decode_op[0]

    @tf_util.define_scope("gru_sample_op")
    def gru_sample_op(self):
        return self.decode_op[1]

    @tf_util.define_scope("loss_op")
    def loss_op(self):
        output_logit, _ = self.gru_decode_op
        output_logit, _, _ = output_logit
        position_logit, is_copy_logit, key_word_logit, copy_word_logit = output_logit
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=is_copy_logit,
                                                       labels=self.output_is_copy))
        sparse_softmax_loss = lambda x, y: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x,logits=y))
        loss += sparse_softmax_loss(self.output_position_label, position_logit)
        loss += sparse_softmax_loss(self.output_keyword_id, key_word_logit)
        loss += sparse_softmax_loss(self.output_copy_word_id, copy_word_logit)
        return loss

    @tf_util.define_scope("train_op")
    def train_op(self):
        optimiizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return tf_util.minimize_and_clip(optimizer=optimiizer,
                                         objective=self.loss_op,
                                         var_list=tf.trainable_variables(),
                                         global_step=self.global_step)


