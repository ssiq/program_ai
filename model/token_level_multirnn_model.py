import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from common import tf_util, rnn_cell, rnn_util, code_util, util
from common.beam_search_util import beam_calculate_length_penalty, beam_calculate_score, beam_cal_top_k, flat_list, \
    beam_flat, select_max_output, beam_gather


class SequenceRNNCell(rnn_cell.RNNWrapper):
    def __init__(self,
                 cell:tf.nn.rnn_cell.RNNCell,
                 max_copy_length,
                 keyword_size,
                 attention_size,
                 reuse=False):
        super().__init__(cell=cell, reuse=reuse)
        self._max_copy_length = max_copy_length
        self._max_position_length = 2*max_copy_length + 1
        self._keyword_size = keyword_size
        self._attention_size = attention_size

    @property
    def output_size(self):
        return tuple([
            tf.TensorShape([]),
            tf.expand_dims(self._max_position_length, axis=0),
            tf.TensorShape([]),
            self._keyword_size,
            tf.expand_dims(self._max_copy_length, axis=0)])

    def call(self, inputs, state):
        _memory, _memory_length, _position_embedding, _position_length,  inputs = inputs
        with tf.variable_scope("input_attention"):
            atten = rnn_util.soft_attention_reduce_sum(_memory,
                                                       inputs,
                                                       self._attention_size,
                                                       _memory_length)
        atten = nest.flatten(atten)
        inputs = nest.flatten(atten)
        print("atten:{}, input:{}".format(atten, inputs))
        inputs = tf.concat(inputs + atten, axis=1)
        gate_weight = tf.get_variable("gate_weight",
                                      shape=(tf_util.get_shape(inputs)[1], tf_util.get_shape(inputs)[1]),
                                      dtype=tf.float32)
        cell_inputs = inputs * tf.sigmoid(tf.matmul(inputs, gate_weight))
        outputs, next_hidden_state = self._cell(cell_inputs, state)
        # a scalar indicating whether the code has been corrupted
        is_continue_logit = tf_util.weight_multiply("continue_weight", outputs, 1)[:, 0]
        #position_logit
        with tf.variable_scope("poistion_logit"):
            position_logit = rnn_util.soft_attention_logit(self._attention_size,
                                                           outputs,
                                                           _position_embedding,
                                                           _position_length)
        position_softmax = tf.nn.softmax(position_logit)
        replace_input = rnn_util.reduce_sum_with_attention_softmax(_position_embedding,
                                                                   position_softmax)[0]
        replace_ouput = tf_util.weight_multiply("replace_output_weight", replace_input, self._attention_size)
        #a scalar indicating whether copies from the code
        is_copy_logit = tf_util.weight_multiply("copy_weight", replace_ouput, 1)[:, 0]
        #key_word_logit
        key_word_logit = tf_util.weight_multiply("key_word_logit_weight", replace_ouput, self._keyword_size)
        #copy_word_logit
        with tf.variable_scope("copy_word_logit"):
            copy_word_logit = rnn_util.soft_attention_logit(self._attention_size, replace_ouput, _memory, _memory_length)

        return (is_continue_logit, position_logit, is_copy_logit, key_word_logit, copy_word_logit,), next_hidden_state

def create_sample_fn():
    def sample_fn(time, outputs, state):
        """Returns `sample_ids`."""
        is_continue_logit, position_logit, is_copy_logit, key_word_logit, copy_word_logit = outputs
        is_continue = tf.greater(tf.nn.sigmoid(is_continue_logit), tf.constant(0.5, dtype=tf.float32))
        position_id = tf.cast(tf.argmax(position_logit, axis=1), tf.int32)
        is_copy = tf.greater(tf.nn.sigmoid(is_copy_logit), tf.constant(0.5, dtype=tf.float32))
        keyword_id =  tf.cast(tf.argmax(key_word_logit, axis=1), dtype=tf.int32)
        copy_word_id = tf.cast(tf.argmax(key_word_logit, axis=1), dtype=tf.int32)
        zeros_id = tf.zeros_like(keyword_id)
        keyword_id, copy_word_id = tf.where(is_copy, zeros_id, keyword_id), tf.where(is_copy, copy_word_id, zeros_id)
        return is_continue, position_id, is_copy, keyword_id, copy_word_id
    return sample_fn

def create_train_helper_function(sample_fn,
                                 memory,
                                 memory_length,
                                 position_embedding,
                                 position_length,
                                 output_length,
                                 output_embedding,
                                 start_label,
                                 batch_size):

    output_length = tf.cast(output_length, tf.bool)

    def initialize_fn():
        is_finished, start_batch = rnn_util.create_decoder_initialize_fn(start_label, batch_size)()
        return is_finished, \
               (memory[:, 0, :, :],
               memory_length[:, 0],
               position_embedding[:, 0, :, :],
               position_length[:,0, ], start_batch)


    def next_input_fn(time, outputs, state, sample_ids):
        """Returns `(finished, next_inputs, next_state)`."""
        finished = tf.logical_not(output_length[:, time])
        def false_fn():
            next_inputs = memory[:, time + 1, :, :], \
                          memory_length[:, time + 1], \
                          position_embedding[:, time + 1, :,:], \
                          position_length[:, time+1],\
                          output_embedding[:, time + 1, :]
            return next_inputs

        def true_fn():
            next_inputs = tf.zeros_like(memory[:, 0, :, :]), \
                          tf.zeros_like(memory_length[:, 0]), \
                          tf.zeros_like(position_embedding[:, 0, :, :]), \
                          tf.zeros_like(position_length[:, 0]), \
                          tf.zeros_like(output_embedding[:, 0, :])
            return next_inputs

        next_inputs = tf.cond(tf.reduce_all(finished),
                              true_fn=true_fn,
                              false_fn=false_fn,
                              strict=True)
        return finished, next_inputs, state
    return (initialize_fn, sample_fn, next_input_fn), \
           (tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])), \
           (tf.bool, tf.int32, tf.bool, tf.int32, tf.int32)

class TokenLevelMultiRnnModel(tf_util.BaseModel):
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
                 placeholder_token,
                 id_to_word_fn,
                 parse_token_fn,
                 ):
        super().__init__(learning_rate=learning_rate)
        self.word_embedding_layer_fn = word_embedding_layer_fn
        self.character_embedding_layer_fn = character_embedding_layer_fn
        self.hidden_state_size = hidden_size
        self.rnn_layer_number = rnn_layer_number
        self.keyword_number = keyword_number
        self.end_token_id = end_token_id
        self.max_decode_iterator_num = max_decode_iterator_num
        self.identifier_token = identifier_token
        self.placeholder_token = placeholder_token
        self.id_to_word = id_to_word_fn
        self.parse_token = parse_token_fn
        self.learning_rate = learning_rate
        #input placeholder
        self.token_input = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name="token_input")
        self.token_input_length = tf.placeholder(dtype=tf.int32, shape=(None, None,), name="token_input_length")
        self.character_input = tf.placeholder(dtype=tf.int32, shape=(None, None, None, None), name="character_input")
        self.character_input_length = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name="character_input_length")
        #train output placeholder
        self.output_is_continue = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_length")
        self.output_position_label = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_position")
        self.output_is_copy = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_is_copy") #1 means using copy
        self.output_keyword_id = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_keyword_id")
        self.output_copy_word_id = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_copy_word_id")
        #predict hidden_state and input
        self.predict_hidden_state = tf.placeholder(dtype=tf.float32, shape=(None, self.hidden_state_size),
                                                   name="predict_hidden_state")
        self.predict_input = tf.placeholder(dtype=tf.float32, shape=(None, tf_util.get_shape(self.start_label_op)[0]),
                                            name="predict_input")
        #create new input placeholder
        self.position_embedding_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 1, None, None))
        self.code_embedding_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 1, None, None))

        #summary
        metrics_input_placeholder = tf.placeholder(tf.float32, shape=[], name="metrics")
        tf_util.add_summary_scalar("metrics", metrics_input_placeholder, is_placeholder=True)
        tf_util.add_summary_histogram("predict_is_continue",
                                      tf.placeholder(tf.float32, shape=(None, None, None), name="predict_is_continue"),
                                      is_placeholder=True)
        tf_util.add_summary_histogram("predict_position_softmax",
                                      tf.placeholder(tf.float32, shape=(None, None, None), name="predict_position_softmax"),
                                      is_placeholder=True)
        tf_util.add_summary_histogram("predict_is_copy",
                                      tf.placeholder(tf.float32, shape=(None, None, None), name="predict_is_copy"),
                                      is_placeholder=True)
        tf_util.add_summary_histogram("predict_key_word",
                                      tf.placeholder(tf.float32, shape=(None, None, None), name="predict_keyword"),
                                      is_placeholder=True)
        tf_util.add_summary_histogram("predict_copy_word",
                                      tf.placeholder(tf.float32, shape=(None, None, None), name="predict_copy_word"),
                                      is_placeholder=True)
        tf_util.add_summary_scalar("loss", self.loss_op, is_placeholder=False)
        tf_util.add_summary_histogram("is_continue", self.softmax_op[0], is_placeholder=False)
        tf_util.add_summary_histogram("position_softmax", self.softmax_op[1], is_placeholder=False)
        tf_util.add_summary_histogram("is_copy", self.softmax_op[2], is_placeholder=False)
        tf_util.add_summary_histogram("key_word", self.softmax_op[3], is_placeholder=False)
        # tf_util.add_summary_histogram("copy_word_logit", self.train_output_logit_op[4], is_placeholder=False)
        tf_util.add_summary_histogram("copy_word", self.softmax_op[4], is_placeholder=False)
        self._summary_fn = tf_util.placeholder_summary_merge()
        self._summary_merge_op = tf_util.merge_op()

        #graph init
        tf_util.init_all_op(self)
        #create new input
        new_input = self._create_output_embedding(self.code_embedding_placeholder,
                                                  self.position_embedding_placeholder)
        sess = tf_util.get_session()
        init = tf.global_variables_initializer()
        sess.run(init)

        # train the function
        self.train = tf_util.function(
            [self.token_input,
             self.token_input_length,
             self.character_input,
             self.character_input_length,
             self.output_is_continue,
             self.output_position_label,
             self.output_is_copy,
             self.output_keyword_id,
             self.output_copy_word_id],
            [self.loss_op, self.loss_op, self.train_op])

        self._loss_fn = tf_util.function(
            [self.token_input,
             self.token_input_length,
             self.character_input,
             self.character_input_length,
             self.output_is_continue,
             self.output_position_label,
             self.output_is_copy,
             self.output_keyword_id,
             self.output_copy_word_id],
            self.loss_op)

        self._one_predict_fn = tf_util.function(
            [self.token_input,
             self.token_input_length,
             self.character_input,
             self.character_input_length,
             self.predict_hidden_state,
             self.predict_input],
            self.one_predict_op)

        self._initial_state_and_initial_label_fn = tf_util.function(
            [self.token_input,
             self.token_input_length,
             self.character_input,
             self.character_input_length,],
            [self.start_label_op, self.result_initial_state_op])

        self._create_next_input_fn = tf_util.function(
            [self.output_position_label,
             self.output_is_copy,
             self.output_keyword_id,
             self.output_copy_word_id,
             self.position_embedding_placeholder,
             self.code_embedding_placeholder],
            new_input
        )

        self._train_summary_fn = tf_util.function(
            [self.token_input,
             self.token_input_length,
             self.character_input,
             self.character_input_length,
             self.output_is_continue,
             self.output_position_label,
             self.output_is_copy,
             self.output_keyword_id,
             self.output_copy_word_id],
            self._summary_merge_op
        )

        # self._test_train_output_fn = tf_util.function(
        #     [self.token_input,
        #      self.token_input_length,
        #      self.character_input,
        #      self.character_input_length,
        #      self.output_is_continue,
        #      self.output_position_label,
        #      self.output_is_copy,
        #      self.output_keyword_id,
        #      self.output_copy_word_id],
        #     [self.softmax_op, self.train_output_logit_op]
        # )


    def _rnn_cell(self, hidden_size):
        return tf.nn.rnn_cell.GRUCell(hidden_size)

    def _multi_rnn_cell(self, hidden_size, layer_number):
        return tf.nn.rnn_cell.MultiRNNCell([self._rnn_cell(hidden_size) for _ in range(layer_number)])


    @tf_util.define_scope("batch_size")
    def batch_size_op(self):
        return tf_util.get_shape(self.token_input)[0]

    @tf_util.define_scope("word_embedding_op")
    def word_embedding_op(self):
        # input_shape = tf_util.get_shape(self.token_input)
        # input_op = tf.reshape(self.token_input, (-1, input_shape[2]))
        input_embedding_op = self.word_embedding_layer_fn(self.token_input)
        # input_embedding_op = tf.reshape(input_embedding_op, (
        # input_shape[0], input_shape[1], input_shape[2], tf_util.get_shape(input_embedding_op[-1])))
        return input_embedding_op

    @tf_util.define_scope("character_embedding_op")
    def character_embedding_op(self):
        # input_shape = tf_util.get_shape(self.character_input)
        # input_length_shape = tf_util.get_shape(self.character_input_length)
        # input_op = tf.reshape(self.character_input, (-1, input_shape[2], input_shape[3]))
        # input_length_op = tf.reshape(self.character_input_length, (-1, input_length_shape[2]))
        input_embedding_op = self.character_embedding_layer_fn(self.character_input, self.character_input_length)
        # input_embedding_op = tf.reshape(input_embedding_op, (input_shape[0], input_shape[1], input_shape[2], -1))
        return input_embedding_op

    @tf_util.define_scope("code_embedding_op")
    def code_embedding_op(self):
        token_embedding = self.word_embedding_op
        character_embedding = self.character_embedding_op
        # mask = tf.equal(self.token_input, self.identifier_token)
        # mask = tf.expand_dims(mask, axis=-1)
        # mask = tf.tile(mask, multiples=[1, 1, 1, tf_util.get_shape(token_embedding)[-1]])
        # return tf.where(mask, character_embedding, token_embedding)
        return code_util.code_embedding(token_embedding, character_embedding, self.token_input, self.identifier_token)

    @tf_util.define_scope("bi_gru_encode_op")
    def bi_gru_encode_op(self):
        code_embedding = self.code_embedding_op
        code_embedding_shape = tf_util.get_shape(code_embedding)
        code_embedding = tf.reshape(code_embedding, (-1, code_embedding_shape[2], code_embedding_shape[3]))
        code_input_length = self.token_input_length
        code_input_length = tf.reshape(code_input_length, [-1])
        (encode_output_fw, encode_output_bw), _ = rnn_util.bi_rnn(lambda: self._multi_rnn_cell(self.hidden_state_size, self.rnn_layer_number),
                                        code_embedding, code_input_length)
        encode_output_fw = tf.reshape(encode_output_fw, (
        code_embedding_shape[0], code_embedding_shape[1], code_embedding_shape[2], self.hidden_state_size))
        encode_output_bw = tf.reshape(encode_output_bw, (
        code_embedding_shape[0], code_embedding_shape[1], code_embedding_shape[2], self.hidden_state_size))
        return encode_output_fw, encode_output_bw

    @tf_util.define_scope("position_embedding")
    def position_embedding_op(self):
        """
        :return: (position_embedding tensor)
        """
        output_fw = self.bi_gru_encode_op[0]
        output_bw = self.bi_gru_encode_op[1]
        token_length = self.position_length_op
        output_fw_shape = tf_util.get_shape(output_fw)
        output_bw_shape = tf_util.get_shape(output_bw)
        # token_length_shape = tf_util.get_shape(self.token_input_length)
        output_fw = tf.reshape(output_fw, (-1, output_fw_shape[2], output_fw_shape[3]))
        output_bw = tf.reshape(output_bw, (-1, output_bw_shape[2], output_bw_shape[3]))
        output_fw_shape_tmp = tf_util.get_shape(output_fw)
        output_bw_shape_tmp = tf_util.get_shape(output_bw)
        token_length = tf.reshape(token_length, (-1, ))

        output = tf.concat((output_fw, output_bw), axis=2)
        output_fw_in = tf.concat((output_fw, tf.zeros((output_fw_shape_tmp[0], 1, output_fw_shape_tmp[2]), dtype=tf.float32)), axis=1)
        output_bw_in = tf.concat((tf.zeros((output_bw_shape_tmp[0], 1, output_bw_shape_tmp[2]), dtype=tf.float32), output_bw), axis=1)
        output_in = tf.concat((output_bw_in, output_fw_in), axis=2)
        output = tf.concat((output, output_in[:, 1:, :]), axis=2)
        output = tf.reshape(output, (output_fw_shape_tmp[0], -1, output_fw_shape_tmp[2] * 2))
        output = tf.concat((output_in[:, :1, :], output), axis=1)

        output = tf_util.sequence_mask_with_length(output, token_length, score_mask_value=0)
        output = tf.reshape(output, (output_fw_shape[0], output_fw_shape[1], -1, 2*output_fw_shape[-1]))
        return output

    @tf_util.define_scope("position_length")
    def position_length_op(self):
        return self.token_input_length * 2 + 1

    @tf_util.define_scope("output_embedding")
    def output_embedding_op(self):
        code_embedding = self.code_embedding_op
        position_embedding = self.position_embedding_op
        return self._create_output_embedding(code_embedding, position_embedding)

    def _create_output_embedding(self, code_embedding, position_embedding):
        keyword_embedding = self.word_embedding_layer_fn(self.output_keyword_id)
        copyword_embedding = rnn_util.gather_sequence(code_embedding, self.output_copy_word_id)
        output_word_embedding = tf.where(
            tf_util.expand_dims_and_tile(tf.cast(self.output_is_copy, tf.bool),
                                         [-1],
                                         [1, 1, tf_util.get_shape(keyword_embedding)[-1]]),
            copyword_embedding,
            keyword_embedding
        )
        position_embedding = rnn_util.gather_sequence(position_embedding, self.output_position_label)
        return tf.concat((position_embedding, output_word_embedding), axis=2)

    @tf_util.define_scope("decode_cell")
    def decode_cell_op(self):
        return SequenceRNNCell(
            self._rnn_cell(self.hidden_state_size),
            tf_util.get_shape(self.code_embedding_op)[2],
            self.keyword_number,
            self.hidden_state_size
        )

    @tf_util.define_scope("start_label")
    def start_label_op(self):
        return tf.zeros(tf_util.get_shape(self.position_embedding_op)[-1]+tf_util.get_shape(self.code_embedding_op)[-1])

    @tf_util.define_scope("result_initial_state")
    def result_initial_state_op(self):
        cell = self.decode_cell_op
        return tf.tile(tf.Variable(cell.zero_state(1, tf.float32)), [self.batch_size_op, 1])

    @tf_util.define_scope("decode")
    def decode_op(self):
        train_helper_fn = create_train_helper_function(create_sample_fn(),
                                                       tf.concat(self.bi_gru_encode_op, axis=-1),
                                                       self.token_input_length,
                                                       self.position_embedding_op,
                                                       self.position_length_op,
                                                       self.output_is_continue,
                                                       self.output_embedding_op,
                                                       self.start_label_op,
                                                       self.batch_size_op)
        return rnn_util.create_train_decode(train_helper_fn[0],
                                            train_helper_fn[1],
                                            train_helper_fn[2],
                                            self.decode_cell_op,
                                            self.result_initial_state_op,
                                            self.max_decode_iterator_num)

    @tf_util.define_scope("loss_op")
    def loss_op(self):
        copy_word_logit, is_continue_logit, is_copy_logit, key_word_logit, position_logit = self.train_output_logit_op
        copy_length = tf.reduce_sum(self.output_is_continue, axis=1) + 1
        is_continue_mask = tf_util.lengths_to_mask(copy_length, tf_util.get_shape(self.output_is_continue)[1])
        loss = tf.reduce_mean(
            tf.multiply(tf.cast(is_continue_mask, tf.float32), tf.nn.sigmoid_cross_entropy_with_logits(logits=is_continue_logit,
                                                                                         labels=tf.cast(
                                                                                             self.output_is_continue,
                                                                                             tf.float32))))
        loss += tf.reduce_mean(
            tf.multiply(tf.cast(is_continue_mask, tf.float32), tf.nn.sigmoid_cross_entropy_with_logits(logits=is_copy_logit,
                                                                                         labels=tf.cast(
                                                                                             self.output_is_copy,
                                                                                             tf.float32))))
        sparse_softmax_loss = lambda x, y: tf.reduce_mean(tf.multiply(tf.cast(is_continue_mask, tf.float32),tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x,logits=y)))
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
                                         global_step=self.global_step_variable)

    @tf_util.define_scope("softmax_op")
    def softmax_op(self):
        copy_word_logit, is_continue_logit, is_copy_logit, key_word_logit, position_logit = self.train_output_logit_op
        is_continue = tf.nn.sigmoid(is_continue_logit)
        position_softmax = tf_util.variable_length_softmax(position_logit, self.position_length_op)
        is_copy = tf.nn.sigmoid(is_copy_logit)
        key_word_softmax = tf.nn.softmax(key_word_logit)
        copy_word_softmax = tf_util.variable_length_softmax(copy_word_logit, self.token_input_length)
        return is_continue, position_softmax, is_copy, key_word_softmax, copy_word_softmax

    @tf_util.define_scope("train_output_logit_op")
    def train_output_logit_op(self):
        output_logit, _, _ = self.decode_op
        output_logit, _ = output_logit
        print("output_logit:{}".format(output_logit))
        is_continue_logit, position_logit, is_copy_logit, key_word_logit, copy_word_logit = output_logit
        return copy_word_logit, is_continue_logit, is_copy_logit, key_word_logit, position_logit

    def train_model(self, *args):
        # print("train_model_input:")
        # for t in args:
        #     print(np.array(t).shape)
            # print("{}:{}".format(np.array(t).shape, np.array(t)))
        # print(np.array(args[4]))
        # l1, l2, _ = self.train(*args)
        # print(l1)
        # return l1, l2, None
        loss, _, train = self.train(*args)
        metrics = self.metrics_model(*args)
        # print('loss : {}. mertics: {}'.format(loss, metrics))
        return loss, metrics, train

    @tf_util.define_scope("one_predict")
    def one_predict_op(self):
        predict_cell_input = [
            tf.concat(self.bi_gru_encode_op, axis=-1)[:, 0, :, :],
            self.token_input_length[:, 0],
            self.position_embedding_op[:, 0, :, :],
            self.position_length_op[:, 0],
            self.predict_input,
        ]
        output, next_state = self.decode_cell_op(predict_cell_input, self.predict_hidden_state)
        is_continue_logit, position_logit, is_copy_logit, key_word_logit, copy_word_logit = output
        output = (tf.nn.sigmoid(is_continue_logit),
                  tf_util.variable_length_softmax(position_logit, self.position_length_op[: ,0]),
                  tf.nn.sigmoid(is_copy_logit),
                  tf.nn.softmax(key_word_logit),
                  tf_util.variable_length_softmax(copy_word_logit, self.token_input_length[:, 0]))
        return output, next_state, self.position_embedding_op, self.code_embedding_op

    def summary(self, *args):
        # print("summary_input:")
        # for t in args:
        #     print(np.array(t).shape)

        # softmax_o, logit_o = self._test_train_output_fn(*args)
        # print("output_logit:")
        # for t in logit_o:
        #     print("is_nan:{}, is_inf:{}".format(np.any(np.isnan(t)), np.any(np.isinf(t))))
        #
        # print("output_softmax:")
        # for t in softmax_o:
        #     print("is_nan:{}, is_inf:{}".format(np.any(np.isnan(t)), np.any(np.isinf(t))))

        train_summary = self._train_summary_fn(*args)
        metrics_model = self.metrics_model(*args)
        tf_util.add_value_scalar("metrics", metrics_model)
        return self._summary_fn(*tf_util.merge_value(), summary_input=train_summary)

    def _create_next_input(self, output, position_embedding, code_embedding):
        _, position, is_copy, key_word, copy_word = output
        return self._create_next_input_fn([position, is_copy, key_word, copy_word, position_embedding, code_embedding])

    def _create_next_code(self, actions, token_input, token_input_length, character_input, character_input_length, identifier_mask):
        """
        This function is used to create the new code based now action
        :param actions:
        :param token_input:
        :param token_input_length:
        :param character_input:
        :param character_input_length:
        :return:
        TODO: fill this function
        """
        is_continues, positions, is_copys, keyword_ids, copy_ids = actions
        for i in range(len(token_input)):
            position = positions[i]
            is_copy = is_copys[i]
            keyword_id = keyword_ids[i]
            copy_id = copy_ids[i]
            code_length = token_input_length[i][0]

            if position % 2 == 1 and is_copy == 0 and keyword_id == self.placeholder_token:
                # delete
                position = int(position/2)
                if position >= code_length:
                    # action position error
                    print('delete action position error', position, code_length)
                    continue
                token_input[i][0].pop(position)
                token_input_length[i][0] -= 1
                character_input[i][0].pop(position)
                character_input_length[i][0].pop(position)
                identifier_mask[i][0] = identifier_mask[i][0][0:position] + identifier_mask[i][0][position+1:]
            else:
                if is_copy:
                    if copy_id >= code_length:
                        # copy position error
                        print('copy position error', copy_id, code_length)
                        print('details:', position, is_copy, keyword_id, copy_id, code_length)
                        continue
                    word_token_id = token_input[i][0][copy_id]
                    word_character_id = character_input[i][0][copy_id]
                    word_character_length = character_input_length[i][0][copy_id]
                    iden_mask = identifier_mask[i][0][copy_id]
                else:
                    word_token_id = keyword_id
                    word = self.id_to_word(word_token_id)
                    if word == None:
                        # keyword id error
                        print('keyword id error', keyword_id)
                        continue
                    word_character_id = self.parse_token(word, character_position_label=True)
                    word_character_length = len(word_character_id)
                    iden_mask = [0] * len(identifier_mask[i][0][0])

                if position % 2 == 0:
                    # insert
                    position = int(position / 2)
                    if position > code_length:
                        # action position error
                        print('insert action position error', position, code_length)
                        continue
                    token_input[i][0].insert(position, word_token_id)
                    token_input_length[i][0] += 1
                    character_input[i][0].insert(position, word_character_id)
                    character_input_length[i][0].insert(position, word_character_length)
                    identifier_mask[i][0] = identifier_mask[i][0][0:position] + iden_mask + identifier_mask[i][0][position:]
                elif position % 2 == 1:
                    # change
                    position = int(position/2)
                    if position >= code_length:
                        # action position error
                        print('change action position error', position, code_length)
                        continue
                    token_input[i][0][position] = word_token_id
                    character_input[i][0][position] = word_character_id
                    character_input_length[i][0][position] = word_character_length
                    identifier_mask[i][0][position] = iden_mask

        return token_input, token_input_length, character_input, character_input_length, identifier_mask

    # def one_predict(self, inputs, states, labels):
    #     output, next_state, position_embedding, code_embedding = self._one_predict(inputs, labels, states)
    #     return self._create_output_and_next_input(code_embedding, next_state, output, position_embedding)

    def _create_output_and_next_input(self, code_embedding, next_state, output, position_embedding):
        is_continue, position, is_copy, keyword_id, copy_id = output
        is_continue = np.array(is_continue > 0.5)
        position = np.argmax(np.array(position), axis=1)
        is_copy = np.array(is_copy > 0.5)
        keyword_id = np.argmax(np.array(keyword_id), axis=1)
        copy_id = np.argmax(np.array(copy_id), axis=1)
        next_labels = self._create_next_input_fn(np.expand_dims(position, axis=1), np.expand_dims(is_copy, axis=1),
                                                 np.expand_dims(keyword_id, axis=1), np.expand_dims(copy_id, axis=1),
                                                 position_embedding, code_embedding)[:, 0, :]
        outputs = (is_continue, position, is_copy, keyword_id, copy_id)
        return outputs, next_state, next_labels

    def _one_predict(self, token_input, token_input_length, charactere_input, character_input_length, identifier_mask, labels, states):
        # token_input, token_input_length, charactere_input, character_input_length = inputs
        output, next_state, position_embedding, code_embedding \
            = self._one_predict_fn(
            token_input,
            token_input_length,
            charactere_input,
            character_input_length,
            identifier_mask,
            states,
            labels
        )
        return output, next_state, position_embedding, code_embedding

    def flat_beam_stack(self, one_input):
        res = [flat_list(inp) for inp in one_input]
        return res

    def batch_top_k(self, beam_stack, k:int):
        if k == 0:
            return [[] for bat in beam_stack]

        batch_indices = [beam_cal_top_k(one_batch, k) for one_batch in beam_stack]
        return batch_indices

    def stack_gather(self, one_output, indices:list):
        all = [beam_gather(one_batch_output, one_batch_indices) for one_batch_output, one_batch_indices in zip(one_output, indices)]
        return all

    def stack_flat_and_gather(self, one_output, indices:list):
        one_output_flat = self.flat_beam_stack(one_output)
        one_output_gather = self.stack_gather(one_output_flat, indices)
        return one_output_gather

    def get_key_from_action_stack(self, action_stack, key_name:str):
        return [beam_get_key_from_action(batch_actions, key_name) for batch_actions in action_stack]

    def top_to_beamid(self, action_stack):
        indices = self.get_key_from_action_stack(action_stack, 'beam_id')
        return indices

    def get_output_from_action_stack(self, action_stack):
        is_continues = self.get_key_from_action_stack(action_stack, 'is_continue')
        positions = self.get_key_from_action_stack(action_stack, 'position')
        is_copys = self.get_key_from_action_stack(action_stack, 'is_copy')
        keyword_ids = self.get_key_from_action_stack(action_stack, 'keyword')
        copy_ids = self.get_key_from_action_stack(action_stack, 'copy_id')
        return is_continues, positions, is_copys, keyword_ids, copy_ids

    def ready_input_stack(self, input_stack, next_labels_stack, next_states_stack):
        input_iterator_list = []
        batch_size = len(input_stack[0])

        for input_sta in input_stack:
            input_iterator_list.append(revert_batch_beam_iterator(input_sta, batch_size))
        label_iterator = revert_batch_beam_iterator(next_labels_stack, batch_size)
        state_iterator = revert_batch_beam_iterator(next_states_stack, batch_size)
        return input_iterator_list, label_iterator, state_iterator

    def calculate_output_score(self, output_stack_list, beam_size):
        batch_size = len(output_stack_list[0])

        output_stack_list = list(zip(*output_stack_list))
        beam_action_stack = [[[] * beam_size] * batch_size]
        beam_p_stack = [[[] * beam_size] * batch_size]
        for batch_id in range(batch_size):
            p_beam, action_beam = beam_calculate_output_score(output_stack_list[batch_id], beam_size)
            beam_action_stack[batch_id].append(p_beam)
            beam_p_stack[batch_id].append(action_beam)

        return beam_p_stack, beam_action_stack

    def calculate_score(self, beam_stack, output_p_stack):
        all_score = []
        for p_stack, output_p in zip(beam_stack, output_p_stack):
            batch_score = beam_calculate_score(p_stack, output_p)
            all_score.append(batch_score)
        return all_score

    def calculate_length_penalty(self, beam_length_stack, output_p_stack, penalty_factor=0.6):
        all_score = []
        for p_length_stack, output_p in zip(beam_length_stack, output_p_stack):
            batch_score = beam_calculate_length_penalty(p_length_stack, output_p, penalty_factor)
            all_score.append(batch_score)
        return all_score

    def concatenate_output(self, select_output_stack, current_output_stack):
        append_cur_output = lambda beam_sel_output, beam_cur_output: beam_sel_output + [beam_cur_output]
        batch_append_output = lambda batch_sel_output, batch_cur_output: [append_cur_output(beam_sel_output, beam_cur_output) for beam_sel_output, beam_cur_output in zip(batch_sel_output, batch_cur_output)]
        select_output_stack = [batch_append_output(batch_sel_output, batch_cur_output) for batch_sel_output, batch_cur_output in zip(select_output_stack, current_output_stack)]
        return select_output_stack

    def predict_model(self, *args,):
        # print('predict iterator start')
        import copy
        import more_itertools
        args = [copy.deepcopy([[ti[0]] for ti in one_input]) for one_input in args]
        token_input, token_input_length, charactere_input, character_input_length, identifier_mask = args
        start_label, initial_state = self._initial_state_and_initial_label_fn(*args)
        batch_size = len(token_input)
        cur_beam_size = 1
        beam_size = 5

        # shape = 5 * batch_size * beam_size * 1 * token_length
        input_stack = init_input_stack(args)
        # shape = batch_size * beam_size
        beam_stack = [[0]]*batch_size
        # shape = 5 * batch_size * beam_size * output_length
        output_stack = []
        # shape = batch_size * beam_size
        mask_stack = [[1]]*batch_size
        # shape = batch_size * beam_size
        beam_length_stack = [[0]]*batch_size
        # shape = 5 * batch_size * beam_size * max_decode_iterator_num
        select_output_stack_list = [[[[]]*cur_beam_size]*batch_size]*5

        # shape = batch_size * beam_size* start_label_length
        next_labels_stack = []
        # shape = batch_size * beam_size * initial_state_length
        next_states_stack = []

        next_states_stack = np.expand_dims(np.array(initial_state), axis=1).tolist()
        next_labels_stack = [[start_label]] * len(initial_state)

        for i in range(self.max_decode_iterator_num):

            input_flat = [flat_list(inp) for inp in input_stack]
            next_labels_flat = flat_list(next_labels_stack)
            next_states_flat = flat_list(next_states_stack)

            one_predict_fn = lambda chunked: self._one_predict(*list(zip(*chunked)))

            chunked_input = more_itertools.chunked(list(zip(*input_flat, next_labels_flat, next_states_flat)), batch_size)
            predict_returns = list(map(one_predict_fn, chunked_input))
            predict_returns = list(zip(*predict_returns))
            outputs, next_state, position_embedding, code_embedding = predict_returns

            output_list = [flat_list(out) for out in zip(*outputs)]
            state_list = flat_list(next_state)
            position_embedding_list = flat_list(position_embedding)
            code_embedding_list = flat_list(code_embedding)

            output_stack = [revert_batch_beam_stack(out_list, batch_size, cur_beam_size) for out_list in output_list]
            next_states_stack = revert_batch_beam_stack(state_list, batch_size, cur_beam_size)
            position_embedding_stack = revert_batch_beam_stack(position_embedding_list, batch_size, cur_beam_size)
            code_embedding_stack = revert_batch_beam_stack(code_embedding_list, batch_size, cur_beam_size)

            # beam_args = (list(zip(*input_stack)), list(zip(*output_stack)), beam_stack, next_states_stack,
            #              position_embedding_stack, code_embedding_stack, mask_stack, beam_length_stack,
            #              list(zip(*select_output_stack_list)), [beam_size] * batch_size)
            # batch_returns = list(util.parallel_map(core_num=3, f=beam_calculate_fn, args=list(zip(*beam_args))))
            batch_returns = list(map(beam_calculate, list(zip(*input_stack)), list(zip(*output_stack)), beam_stack, next_states_stack, position_embedding_stack, code_embedding_stack, mask_stack, beam_length_stack, list(zip(*select_output_stack_list)), [beam_size]*batch_size))
            def create_next(ret):
                ret = list(ret)
                ret[0] = self._create_next_code(ret[1], *ret[0])
                return ret
            batch_returns = [create_next(ret) for ret in batch_returns]
            input_stack, output_stack, select_output_stack_list, mask_stack, beam_stack, next_states_stack, position_embedding_stack, code_embedding_stack, beam_length_stack = list(zip(*batch_returns))
            input_stack = list(zip(*input_stack))
            output_stack = list(zip(*output_stack))
            select_output_stack_list = list(zip(*select_output_stack_list))

            if np.sum(output_stack[0]) == 0:
                break

            output_flat = [flat_list(out) for out in output_stack]
            position_embedding_flat = flat_list(position_embedding_stack)
            code_embedding_flat = flat_list(code_embedding_stack)

            create_label_lambda_fn = lambda is_continue, position, is_copy, keyword_id, copy_id, position_emb, code_emb: self._create_next_input_fn(np.expand_dims(position, axis=1), np.expand_dims(is_copy, axis=1),
                                                         np.expand_dims(keyword_id, axis=1), np.expand_dims(copy_id, axis=1), position_emb, code_emb)[:, 0, :]
            create_label_lambda_chunked_fn = lambda chunked: create_label_lambda_fn(*list(zip(*chunked)))
            next_labels_stack = list(map(create_label_lambda_chunked_fn, more_itertools.chunked(list(zip(*list(output_flat + [position_embedding_flat] + [code_embedding_flat]))), batch_size)))
            next_labels_stack = flat_list(next_labels_stack)
            next_labels_stack = np.reshape(next_labels_stack, (batch_size, beam_size, -1)).tolist()
            # [create_label_lambda_fn() for is_continue, position, is_copy, keyword_id, copy_id in more_itertools.chunked(list(zip(*output_stack)) + [position_embedding_stack] + [code_embedding_stack], batch_size)]

            input_stack = [list(util.padded(list(inp))) for inp in input_stack]
            mask_input_with_end_fn = lambda token_input: list([util.mask_input_with_end(batch_mask, batch_inp, n_dim=1).tolist() for batch_mask, batch_inp in zip(mask_stack, token_input)])
            input_stack = list(map(mask_input_with_end_fn, input_stack))
            cur_beam_size = beam_size


        summary = copy.deepcopy(select_output_stack_list)
        tf_util.add_value_histogram("predict_is_continue", util.padded(summary[0]))
        tf_util.add_value_histogram("predict_position_softmax", util.padded(summary[1]))
        tf_util.add_value_histogram("predict_is_copy", util.padded(summary[2]))
        tf_util.add_value_histogram("predict_key_word", util.padded(summary[3]))
        tf_util.add_value_histogram("predict_copy_word", util.padded(summary[4]))

        final_output = select_max_output(beam_stack, select_output_stack_list)
        return final_output

    def metrics_model(self, *args):
        # print("metrics input")
        # for t in args:
        #     print(np.array(t).shape)
        input_data = args[0:5]
        output_data = args[5:10]
        predict_data = self.predict_model(*input_data,)
        metrics_value = self.cal_metrics(output_data, predict_data)
        # print('metrics_value: ', metrics_value)
        return metrics_value

    def cal_metrics(self, output_data, predict_data):
        res_mask = []
        predict_is_continue = predict_data[0]
        for bat in predict_is_continue:
            zero_item = np.argwhere(np.array(bat) == 0)
            # print("bat:{}, zero_item:{}".format(bat, zero_item))
            if len(zero_item) == 0:
                iter_len = self.max_decode_iterator_num
            else:
                iter_len = np.min(zero_item) + 1
            res_mask.append([1] * iter_len + [0] * (self.max_decode_iterator_num-iter_len))
        res_mask = np.array(res_mask)
        # print("res_mask:{}".format(res_mask))

        true_mask = np.ones([len(output_data[0])])
        for i in range(len(predict_data)):
            # true_mask = 0
            output_idata = self.fill_output_data(output_data[i], self.max_decode_iterator_num)
            predict_idata = self.fill_output_data(predict_data[i], self.max_decode_iterator_num)

            predict_idata = np.where(res_mask, predict_idata, np.zeros_like(predict_idata))
            # print("index {}: output_data {}, predict_data {}".format(i, output_idata, predict_idata))

            res = np.equal(output_idata, predict_idata)
            res = res.reshape([res.shape[0], -1])
            res = np.all(res, axis=1)
            true_mask = np.logical_and(true_mask, res)
        return np.mean(true_mask)

    def fill_output_data(self, output:list, iter_len):
        res = [t + [0]*(iter_len-len(t)) for t in output]
        return np.array(res)


def beam_calculate_fn(args):
    return beam_calculate(*args)


def beam_calculate(inputs, outputs_logit, beam_score, next_states, position_embedding, code_embedding, end_beam, length_beam, select_beam, beam_size):
    # is_continues_beam, positions_beam, is_copys_beam, keyword_ids_beam, copy_ids_beam = outputs

    # if np.sum(end_beam) == 0:
    #     outputs = [[0]*len(out) for out in outputs_logit]
    #     return inputs, outputs, select_beam, end_beam, beam_score, next_states, position_embedding, code_embedding, length_beam

    length_beam = (np.array(length_beam) + np.array(end_beam)).tolist()
    cur_beam_size = len(outputs_logit[1])

    p_beam, action_beam = beam_calculate_output_score(outputs_logit, beam_size)
    p_beam = [(p_b if end_b else [0]) for p_b, end_b in zip(p_beam, end_beam)]
    p_score = beam_calculate_score(beam_score, p_beam)
    p_score_with_penalty = beam_calculate_length_penalty(length_beam, p_score)

    p_score = beam_flat(p_score)
    p_score_with_penalty = beam_flat(p_score_with_penalty)

    action_beam = beam_flat(action_beam)
    top_indices = beam_cal_top_k(p_score_with_penalty, beam_size)
    beam_score = beam_gather(p_score, top_indices)
    action_beam = beam_gather(action_beam, top_indices)
    beam_indices = beam_top_to_beamid(action_beam)

    # outputs_logit = [self.beam_gather(out, beam_indices) for out in outputs_logit]
    end_beam = beam_gather(end_beam, beam_indices)
    outputs = beam_get_output_from_action_beams(action_beam)
    outputs = [np.where(end_beam, out, np.zeros_like(out)).tolist() for out in outputs]
    # print('outputs:', outputs)
    select_beam = [beam_gather(sel_beam, beam_indices, deepcopy=True) for sel_beam in select_beam]
    is_continues_beam = outputs[0]
    end_beam = np.logical_and(end_beam, is_continues_beam).tolist()
    select_beam = [np.concatenate((sel_out, np.expand_dims(out, axis=1)), 1).tolist() for sel_out, out in zip(select_beam, outputs)]
    length_beam = beam_gather(length_beam, beam_indices)

    next_states = beam_gather(next_states, beam_indices, deepcopy=True)
    position_embedding = beam_gather(position_embedding, beam_indices, deepcopy=True)
    code_embedding = beam_gather(code_embedding, beam_indices, deepcopy=True)

    inputs = [beam_gather(inp, beam_indices, deepcopy=True) for inp in inputs]
    # inputs = self._create_next_code(outputs, *inputs)

    return inputs, outputs, select_beam, end_beam, beam_score, next_states, position_embedding, code_embedding, length_beam


def beam_calculate_output_score(output_beam_list, beam_size):
    import math

    output_is_continues, output_positions, output_is_copys, output_keyword_ids, output_copy_ids = output_beam_list
    cur_beam_size = len(output_positions)
    # print('cur_beam_size:',cur_beam_size)
    beam_action_stack = [[] for i in range(beam_size)]
    beam_p_stack = [[] for i in range(beam_size)]

    top_position_beam_id_list = [beam_cal_top_k(beam, beam_size) for beam in output_positions]
    top_keyword_beam_id_list = [beam_cal_top_k(beam, beam_size) for beam in output_keyword_ids]
    top_copy_beam_id_list = [beam_cal_top_k(beam, beam_size) for beam in output_copy_ids]
    sigmoid_to_p_distribute = lambda x: [1 - x, x]
    output_is_continues = [sigmoid_to_p_distribute(beam) for beam in output_is_continues]
    output_is_copys = [sigmoid_to_p_distribute(beam) for beam in output_is_copys]
    top_is_continue_beam_id_list = [beam_cal_top_k(beam, beam_size) for beam in output_is_continues]
    top_is_copy_beam_id_list = [beam_cal_top_k(beam, beam_size) for beam in output_is_copys]
    for beam_id in range(cur_beam_size):

        for position_id in top_position_beam_id_list[beam_id]:
            for is_continue in top_is_continue_beam_id_list[beam_id]:
                for is_copy in top_is_copy_beam_id_list[beam_id]:
                    if is_copy == 1:
                        for copy_id in top_copy_beam_id_list[beam_id]:
                            keyword = 0
                            is_continue_p = output_is_continues[beam_id][is_continue]
                            position_p = output_positions[beam_id][position_id]
                            is_copy_p = output_is_copys[beam_id][is_copy]
                            copy_id_p = output_copy_ids[beam_id][copy_id]
                            is_continue_p = is_continue_p if is_continue_p > 0 else 0.00001
                            position_p = position_p if position_p > 0 else 0.00001
                            is_copy_p = is_copy_p if is_copy_p > 0 else 0.00001
                            copy_id_p = copy_id_p if copy_id_p > 0 else 0.00001

                            action = {'is_continue': is_continue, 'position': position_id, 'is_copy': is_copy,
                                      'keyword': keyword, 'copy_id': copy_id, 'beam_id': beam_id}
                            # print(is_continue_p, position_p, is_copy_p, copy_id_p)
                            p = math.log(is_continue_p) + math.log(position_p) + math.log(is_copy_p) + math.log(copy_id_p)

                            beam_action_stack[beam_id].append(action)
                            beam_p_stack[beam_id].append(p)

                    else:
                        for keyword in top_keyword_beam_id_list[beam_id]:
                            copy_id = 0
                            is_continue_p = output_is_continues[beam_id][is_continue]
                            position_p = output_positions[beam_id][position_id]
                            is_copy_p = output_is_copys[beam_id][is_copy]
                            keyword_p = output_keyword_ids[beam_id][keyword]
                            is_continue_p = is_continue_p if is_continue_p > 0 else 0.00001
                            position_p = position_p if position_p > 0 else 0.00001
                            is_copy_p = is_copy_p if is_copy_p > 0 else 0.00001
                            keyword_p = keyword_p if keyword_p > 0 else 0.00001

                            action = {'is_continue': is_continue, 'position': position_id, 'is_copy': is_copy,
                                      'keyword': keyword, 'copy_id': copy_id, 'beam_id': beam_id}
                            p = math.log(is_continue_p) + math.log(position_p) + math.log(is_copy_p) + math.log(keyword_p)

                            beam_action_stack[beam_id].append(action)
                            beam_p_stack[beam_id].append(p)
    return beam_p_stack, beam_action_stack


def beam_get_output_from_action_beams(action_beams):
    is_continues = beam_get_key_from_action(action_beams, 'is_continue')
    positions = beam_get_key_from_action(action_beams, 'position')
    is_copys = beam_get_key_from_action(action_beams, 'is_copy')
    keyword_ids = beam_get_key_from_action(action_beams, 'keyword')
    copy_ids = beam_get_key_from_action(action_beams, 'copy_id')
    return is_continues, positions, is_copys, keyword_ids, copy_ids


def beam_top_to_beamid(beam_actions):
    indices = beam_get_key_from_action(beam_actions, 'beam_id')
    return indices


def beam_get_key_from_action(beam_actions, key_name):
    return [act[key_name] for act in beam_actions]


def revert_batch_beam_stack(one_output, batch_size, beam_size):
    one_output = np.array(one_output)
    one_output_shape = list(one_output.shape)
    one_output = np.reshape(one_output, [batch_size, beam_size] + one_output_shape[1:]).tolist()
    return one_output


def revert_batch_beam_iterator(one_input, batch_size):
    import more_itertools
    one_input = np.array(one_input)
    one_input_shape = list(one_input.shape)
    one_input = np.reshape(one_input, [one_input_shape[0] * one_input_shape[1]] + one_input_shape[2:]).tolist()
    one_input_iterator = more_itertools.chunked(one_input, batch_size)
    return one_input_iterator


def init_input_stack(args):
    # shape = 5 * batch_size * beam_size * iterator_size * token_length
    init_input_fn = lambda one_input: np.expand_dims(np.array(util.padded(one_input)), axis=1).tolist()
    input_stack = [init_input_fn(one_input) for one_input in args]
    return input_stack
