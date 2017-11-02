import tensorflow as tf

from common import code_util, tf_util, util, rnn_util, rnn_cell

def _rnn_cell(hidden_size):
    return tf.nn.rnn_cell.GRUCell(hidden_size)


def _multi_rnn_cell(hidden_size, layer_number):
    return tf.nn.rnn_cell.MultiRNNCell([_rnn_cell(hidden_size) for _ in range(layer_number)])


class OutputAttentionWrapper(rnn_cell.GatedAttentionWrapper):

    def __init__(self,
                 cell: tf.nn.rnn_cell.RNNCell,
                 memory,
                 memory_length,
                 attention_size: int,
                 keyword_num,
                 reuse=False):
        super().__init__(cell, memory, memory_length, attention_size, reuse)
        self._keyword_num = keyword_num

    @property
    def output_size(self):
        return tuple([tf.TensorShape([]), self._keyword_num, tf_util.get_shape(self._memory)[2]])

    def call(self, inputs, state):
        output, next_state = super().call(inputs, state)
        is_copy = tf.sigmoid(tf_util.weight_multiply("is_copy_weight", output, 1))
        keyword_logit = tf_util.weight_multiply("keyword_weight", output, self._keyword_num)
        with tf.variable_scope("copy_word_logit"):
            copy_word_logit = rnn_util.soft_attention_logit(
                self._attention_size,
                output,
                self._memory,
                self._memory_length
            )
        return (is_copy, keyword_logit, copy_word_logit), next_state


def create_sample_fn():
    def sample_fn(time, outputs, state):
        """Returns `sample_ids`."""
        is_copy_logit, key_word_logit, copy_word_logit = outputs
        is_copy = tf.greater(tf.nn.sigmoid(is_copy_logit), tf.constant(0.5, dtype=tf.float32))
        keyword_id = tf.argmax(key_word_logit, axis=1)
        copy_word_id = tf.argmax(key_word_logit, axis=1)
        return is_copy, keyword_id, copy_word_id
    return sample_fn


def create_train_helper_function(sample_fn,
                                 output_length,
                                 output_embedding,
                                 batch_size):
    initialize_fn = rnn_util.create_decoder_initialize_fn(output_embedding[0, 0, :], batch_size)
    def next_input_fn(time, outputs, state, sample_ids):
        """Returns `(finished, next_inputs, next_state)`."""
        finished = tf.greater_equal(time+2, output_length)
        next_inputs = output_embedding[:, time+1, :]
        return finished, next_inputs, state
    return initialize_fn, sample_fn, next_input_fn


def create_sample_helper_function(sample_fn,
                                  start_label,
                                  end_label,
                                  batch_size,
                                  intput_embedding_seq,
                                  token_embedding_fn):
    initialize_fn = rnn_util.create_decoder_initialize_fn(start_label, batch_size)
    def next_input_fn(time, outputs, state, sample_ids):
        """Returns `(finished, next_inputs, next_state)`."""
        is_copy, keyword_id, copy_word_id = sample_ids
        finished = tf.logical_not(is_copy)
        finished = tf.logical_and(finished, tf.equal(keyword_id, end_label))
        keyword_embedding = token_embedding_fn(keyword_id)
        copy_word_embedding = rnn_util.gather_sequence(intput_embedding_seq, copy_word_id)
        next_inputs = tf.where(is_copy, copy_word_embedding, keyword_embedding)
        return finished, next_inputs, state
    return initialize_fn, sample_fn, next_input_fn



class Seq2SeqModel(tf_util.Summary):
    def __init__(self,
                 word_embedding_layer_fn,
                 character_embedding_layer_fn,
                 hidden_size,
                 rnn_layer_number,
                 keyword_number,
                 end_token_id,
                 learning_rate,
                 max_decode_iterator_num,
                 identifier_token,
                 ):
        super().__init__()
        self.word_embedding_layer_fn = word_embedding_layer_fn
        self.character_embedding_layer_fn = character_embedding_layer_fn
        self.hidden_state_size = hidden_size
        self.rnn_layer_number = rnn_layer_number
        self.keyword_number = keyword_number
        self.end_token_id = end_token_id
        self.max_decode_iterator_num = max_decode_iterator_num
        self.identifier_token = identifier_token
        self.learning_rate = learning_rate
        self.token_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name="token_input")
        self.token_input_length = tf.placeholder(dtype=tf.int32, shape=(None,), name="token_input_length")
        self.character_input = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name="character_input")
        self.character_input_length = tf.placeholder(dtype=tf.int32, shape=(None, None), name="character_input_length")
        self.output_length = tf.placeholder(dtype=tf.int32, shape=(None,), name="output_length")
        self.output_is_copy = tf.placeholder(dtype=tf.int32, shape=(None, None),
                                             name="output_is_copy")  # 1 means using copy
        self.output_keyword_id = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_keyword_id")
        self.output_copy_word_id = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_copy_word_id")

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
        return code_util.code_embedding(self.word_embedding_op,
                                        self.character_embedding_op,
                                        self.token_input,
                                        self.identifier_token)

    @tf_util.define_scope("bi_gru_encode_op")
    def bi_gru_encode_op(self):
        return rnn_util.bi_rnn(lambda: _multi_rnn_cell(self.hidden_state_size,
                                                       self.rnn_layer_number),
                               self.code_embedding_op,
                               self.token_input_length)[0]

