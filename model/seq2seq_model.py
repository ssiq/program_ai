import tensorflow as tf
from tensorflow.python.util import nest

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
        if isinstance(self._memory, tuple):
            m = self._memory[0]
        else:
            m = self._memory
        return tuple([tf.TensorShape([]), self._keyword_num, tf.expand_dims(tf_util.get_shape(m)[1], axis=0)])

    def call(self, inputs, state):
        output, next_state = super().call(inputs, state)
        is_copy = tf.sigmoid(tf_util.weight_multiply("is_copy_weight", output, 1))
        is_copy = is_copy[:, 0]
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
        keyword_id =  tf.cast(tf.argmax(key_word_logit, axis=1), dtype=tf.int32)
        copy_word_id = tf.cast(tf.argmax(key_word_logit, axis=1), dtype=tf.int32)
        zeros_id = tf.zeros_like(keyword_id)
        keyword_id, copy_word_id = tf.where(is_copy, zeros_id, keyword_id), tf.where(is_copy, copy_word_id, zeros_id)
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
    return (initialize_fn, sample_fn, next_input_fn), (tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])), (tf.bool, tf.int32, tf.int32)


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
        print("is_copy:{}".format(is_copy))
        print("sample_finished:{}".format(finished))
        return finished, next_inputs, state
    return (initialize_fn, sample_fn, next_input_fn), (tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])), (tf.bool, tf.int32, tf.int32)



class Seq2SeqModel(tf_util.Summary):
    def __init__(self,
                 word_embedding_layer_fn,
                 character_embedding_layer_fn,
                 hidden_size,
                 rnn_layer_number,
                 keyword_number,
                 start_id,
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
        self.start_id = start_id
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

        input_placeholders = [self.token_input,
                              self.token_input_length,
                              self.character_input,
                              self.character_input_length,
                              ]

        output_placeholders = [self.output_length,
                               self.output_is_copy,
                               self.output_keyword_id,
                               self.output_copy_word_id]

        self._add_summary_scalar("loss", self.loss_op)
        self._add_summary_scalar("metrics", self.metrics_op)
        self._merge_all()
        tf_util.init_all_op(self)
        init = tf.global_variables_initializer()
        sess = tf_util.get_session()
        sess.run(init)

        self.train = tf_util.function(
            input_placeholders + output_placeholders,
            [self.loss_op, self.metrics_op, self.train_op]
        )

        self.metrics = tf_util.function( input_placeholders + output_placeholders, self.metrics_op)
        self.predict = tf_util.function(input_placeholders, self.predict_op)
        self.summary = tf_util.function(input_placeholders + output_placeholders, self.summary_op)

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

    @tf_util.define_scope("decode_cell")
    def decode_cell_op(self):
        return OutputAttentionWrapper(
            cell=_rnn_cell(self.hidden_state_size),
            memory=self.bi_gru_encode_op,
            memory_length=self.token_input_length,
            attention_size=self.hidden_state_size,
            keyword_num=self.keyword_number
        )

    @tf_util.define_scope("start_label")
    def start_label_op(self):
        return self.word_embedding_layer_fn(self.start_id)

    @tf_util.define_scope("output_embedding")
    def output_embedding_op(self):
        keyword_embedding = self.word_embedding_layer_fn(self.output_keyword_id)
        copyword_embedding = rnn_util.gather_sequence(self.code_embedding_op, self.output_copy_word_id)
        mask = tf.cast(self.output_is_copy, tf.bool)
        mask = tf.expand_dims(mask, axis=2)
        mask = tf.tile(mask, multiples=[1, 1, tf_util.get_shape(copyword_embedding)[2]])
        return tf.where(mask, copyword_embedding, keyword_embedding)

    @tf_util.define_scope("result_initial_state")
    def result_initial_state_op(self):
        cell = self.decode_cell_op
        return tf.tile(tf.Variable(cell.zero_state(1, tf.float32)), [self.batch_size_op, 1])

    @tf_util.define_scope("decode_op")
    def decode_op(self):
        sample_helper_fn, sample_sample_output_shape, sample_sample_dtype = create_sample_helper_function(create_sample_fn(),
                                                                                     self.start_label_op,
                                                                                     self.end_token_id,
                                                                                     self.batch_size_op,
                                                                                     self.code_embedding_op,
                                                                                     self.word_embedding_layer_fn)
        training_helper_fn, training_sample_output_shape, training_sample_dtype= create_train_helper_function(create_sample_fn(),
                                                                                        self.output_length,
                                                                                        self.output_embedding_op,
                                                                                        self.batch_size_op)
        return rnn_util.create_decode(
            sample_helper_fn,
            sample_sample_output_shape,
            sample_sample_dtype,
            training_helper_fn,
            training_sample_output_shape,
            training_sample_dtype,
            self.decode_cell_op,
            self.result_initial_state_op,
            max_decode_iterator_num=self.max_decode_iterator_num
        )

    @tf_util.define_scope("gru_decode_op")
    def gru_decode_op(self):
        return self.decode_op[0]

    @tf_util.define_scope("gru_sample_op")
    def gru_sample_op(self):
        return self.decode_op[1]

    @tf_util.define_scope("loss_op")
    def loss_op(self):
        output_logit, _, _ = self.gru_decode_op
        output_logit, _ = output_logit
        is_copy_logit, key_word_logit, copy_word_logit = output_logit
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=is_copy_logit,
                                                       labels=tf.cast(self.output_is_copy[:, 1:], tf.float32)))
        sparse_softmax_loss = lambda x, y: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x,logits=y))
        loss += sparse_softmax_loss(self.output_keyword_id[:, 1:], key_word_logit)
        loss += sparse_softmax_loss(self.output_copy_word_id[:, 1:], copy_word_logit)
        return loss

    @tf_util.define_scope("train_op")
    def train_op(self):
        optimiizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return tf_util.minimize_and_clip(optimizer=optimiizer,
                                         objective=self.loss_op,
                                         var_list=tf.trainable_variables(),
                                         global_step=self.global_step_variable)

    @tf_util.define_scope("predict_op")
    def predict_op(self):
        output_logit, _, output_length = self.gru_sample_op
        _, sample_ids = output_logit
        is_copy, keyword_id, copy_word_id = sample_ids
        return output_length, is_copy, keyword_id, copy_word_id

    @tf_util.define_scope("metrics_op")
    def metrics_op(self):
        # output_length, is_copy, keyword_id, copy_word_id = self.predict_op
        # true_mask = tf.equal(output_length, self.output_length-1)
        # new_maks_fn = lambda x, y: tf.logical_and(true_mask, tf.reduce_all(tf.equal(x, y), axis=1))
        # is_copy = tf.cast(is_copy, tf.int32)
        # true_mask = new_maks_fn(is_copy, self.output_is_copy)
        # true_mask = new_maks_fn(keyword_id, self.output_keyword_id)
        # true_mask = new_maks_fn(copy_word_id, self.output_copy_word_id)
        # return tf.reduce_mean(tf.cast(true_mask, tf.float32))
        return self.loss_op

    def train_model(self, *data):
        loss, _, train_op_res = self.train(*data)
        metrics_value = self.metrics_model(*data)
        return loss, metrics_value, train_op_res

    def metrics_model(self, *data):
        input_data = data[0:4]
        output_data = data[4:8]
        predict_data = self.predict(*input_data)
        metrics_value = self.cal_metrics(output_data, predict_data)
        return metrics_value

    def cal_metrics(self, output_data, predict_data):
        import numpy as np

        output_new_data = []
        output_new_data.append(list(map(lambda x: x - 1, output_data[0])))
        output_new_data.append(list(map(lambda x: x[1:], output_data[1])))
        output_new_data.append(list(map(lambda x: x[1:], output_data[2])))
        output_new_data.append(list(map(lambda x: x[1:], output_data[3])))

        res = []
        for i in range(0, len(predict_data)):
            true_mask = 0
            predict_idata = np.array(predict_data[i])
            output_idata = np.array(output_new_data[i])
            if predict_idata.shape == output_idata.shape and np.array_equal(predict_idata, output_idata):
                true_mask = 1
            res.append(true_mask)

        return np.mean(res)
