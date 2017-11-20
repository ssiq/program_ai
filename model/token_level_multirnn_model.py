import tensorflow as tf
from tensorflow.python.util import nest

from common import tf_util, rnn_cell, rnn_util, code_util
from common.rnn_util import create_decoder_initialize_fn, create_decode


class SequenceRNNCell(rnn_cell.RNNWrapper):
    def __init__(self, cell:tf.nn.rnn_cell.RNNCell,
                 position_embedding,
                 position_length,
                 keyword_size,
                 reuse=False):
        super().__init__(cell=cell, reuse=reuse)
        self.position_embedding = position_embedding
        self.position_length = position_length
        self.keyword_size = keyword_size

    @property
    def output_size(self):
        return super().output_size

    def call(self, inputs, state):

        input_shape = tf_util.get_shape(inputs)

        input_length = tf.tile(tf.Variable([input_shape[1]], dtype=tf.int32), tf.Variable([input_shape[0]]))
        outputs, _ = rnn_util.rnn(tf.nn.rnn_cell.GRUCell, inputs=inputs, length_of_inputs=input_length, initial_state=state)
        outputs_shape = tf_util.get_shape(outputs)

        state_mask = tf.get_variable('mask_state', shape=[tf_util.get_shape(state)[1], outputs_shape[-1]])
        masked_state = tf.matmul(state, state_mask)
        masked_state = tf.expand_dims(masked_state, axis=2)
        position_outputs_logits = tf.reshape(tf.matmul(outputs, masked_state), shape=(input_shape[0], input_shape[1]))
        position_output = tf.arg_max(position_outputs_logits, 1)

        position_mask = tf.tile(tf.expand_dims(tf.one_hot(position_output, depth=input_shape[1]), axis=2), (1, 1, input_shape[-1]))
        position_embedding = tf.reshape(tf.reduce_sum(outputs * position_mask, axis=1), (input_shape[0], input_shape[-1]))

        is_copy_weight = tf.get_variable('is_copy_weight', shape=(input_shape[-1], 1))
        is_copy_logits = tf.reshape(tf.matmul(position_embedding, is_copy_weight), shape=(-1))
        is_copy_output = tf.nn.sigmoid(is_copy_logits)

        keyword_id_weight = tf.get_variable('keyword_id_weight', shape=(input_shape[-1], self.keyword_size))
        keyword_id_logits = tf.matmul(position_embedding, keyword_id_weight)
        keyword_id_output = tf.arg_max(tf.nn.softmax(keyword_id_logits), 1)

        copy_id_weight = tf.get_variable('copy_id_weight', shape=(input_shape[-1], 1))
        copy_id_logits = tf.reshape(tf.matmul(position_embedding, copy_id_weight), shape=(-1))
        copy_id_output = tf.arg_max(tf.nn.softmax(copy_id_logits), 1)

        input_getemask = tf.get_variable('input_gatemask', shape=(input_shape[-1], tf_util.get_shape(state)[1]))
        state_gatemask = tf.get_variable('state_gatemask', shape=(tf_util.get_shape(state)[1], tf_util.get_shape(state)[1]))
        next_hidden_state = tf.matmul(position_embedding, input_getemask) + tf.matmul(state, state_gatemask)
        next_hidden_state = tf.tanh(next_hidden_state)

        return (position_output, is_copy_output, keyword_id_output, copy_id_output), next_hidden_state


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
        self.learning_rate = learning_rate
        self.token_input = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name="token_input")
        self.token_input_length = tf.placeholder(dtype=tf.int32, shape=(None, None,), name="token_input_length")
        self.character_input = tf.placeholder(dtype=tf.int32, shape=(None, None, None, None), name="character_input")
        self.character_input_length = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name="character_input_length")
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

    def _rnn_cell(self, hidden_size):
        return tf.nn.rnn_cell.GRUCell(hidden_size)

    def _multi_rnn_cell(self, hidden_size, layer_number):
        return tf.nn.rnn_cell.MultiRNNCell([self._rnn_cell(hidden_size) for _ in range(layer_number)])


    @tf_util.define_scope("batch_size")
    def batch_size_op(self):
        return tf_util.get_shape(self.token_input)[0]

    @tf_util.define_scope("word_embedding_op")
    def word_embedding_op(self):
        input_shape = tf_util.get_shape(self.token_input)
        input_op = tf.reshape(self.token_input, (-1, input_shape[2]))
        input_embedding_op = self.word_embedding_layer_fn(input_op)
        input_embedding_op = tf.reshape(input_embedding_op, (input_shape[0], input_shape[1], input_shape[2], -1))
        return input_embedding_op

    @tf_util.define_scope("character_embedding_op")
    def character_embedding_op(self):
        input_shape = tf_util.get_shape(self.character_input)
        input_length_shape = tf_util.get_shape(self.character_input_length)
        input_op = tf.reshape(self.character_input, (-1, input_shape[2], input_shape[3]))
        input_length_op = tf.reshape(self.character_input_length, (-1, input_length_shape[2]))
        input_embedding_op = self.character_embedding_layer_fn(input_op, input_length_op)
        input_embedding_op = tf.reshape(input_embedding_op, (input_shape[0], input_shape[1], input_shape[2], -1))
        return input_embedding_op

    @tf_util.define_scope("code_embedding_op")
    def code_embedding_op(self):
        token_embedding = self.word_embedding_op
        character_embedding = self.character_embedding_op
        mask = tf.equal(self.token_input, self.identifier_token)
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.tile(mask, multiples=[1, 1, 1, tf_util.get_shape(token_embedding[3])])
        return tf.where(mask, character_embedding, token_embedding)

    @tf_util.define_scope("bi_gru_encode_op")
    def bi_gru_encode_op(self):
        code_embedding = self.code_embedding_op
        code_embedding_shape = tf_util.get_shape(code_embedding)
        code_embedding = tf.reshape(code_embedding, (-1, code_embedding_shape[2], code_embedding[3]))
        code_input_length = self.token_input_length
        code_input_length = tf.reshape(code_input_length, [-1])
        (encode_output_fw, encode_output_bw), _ = rnn_util.bi_rnn(lambda: self._multi_rnn_cell(self.hidden_state_size, self.rnn_layer_number),
                                        code_embedding, code_input_length)
        encode_output_fw = tf.reshape(encode_output_fw, (code_embedding_shape[0], code_embedding_shape[1], code_embedding_shape[2], -1))
        encode_output_bw = tf.reshape(encode_output_bw, (code_embedding_shape[0], code_embedding_shape[1], code_embedding_shape[2], -1))
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
        token_length = tf.reshape(token_length, (-1))

        output = tf.concat((output_fw, output_bw), axis=2)
        output_fw_in = tf.concat((output_fw, tf.zeros((output_fw_shape_tmp[0], 1, output_fw_shape_tmp[2]), dtype=tf.float32)), axis=1)
        output_bw_in = tf.concat((tf.zeros((output_bw_shape_tmp[0], 1, output_bw_shape_tmp[2]), dtype=tf.float32), output_bw), axis=1)
        output_in = tf.concat((output_bw_in, output_fw_in), axis=2)
        output = tf.concat((output, output_in[:, 1:, :]), axis=2)
        output = tf.reshape(output, (output_fw_shape_tmp[0], -1, output_fw_shape_tmp[2] * 2))
        output = tf.concat((output_in[:, 0, :], output), axis=1)

        output = tf_util.sequence_mask_with_length(output, token_length, score_mask_value=0)
        output = tf.reshape(output, (output_fw_shape[0], output_fw_shape[1], -1, output_fw_shape[2]))
        return output

    @tf_util.define_scope("position_length")
    def position_length_op(self):
        return self.token_input_length * 2 + 1

    def sequence_rnn_length_op(self):
        token_input_length = self.token_input_length
        token_input_length = tf.where(tf.greater(token_input_length, tf.constant(0, dtype=tf.float32)), tf.constant(1, dtype=tf.float32), tf.constant(0, dtype=tf.float32))
        token_input_length = tf.reduce_sum(token_input_length, axis=1)
        return token_input_length

    def sequence_cell_op(self):
        return SequenceRNNCell(self._rnn_cell(self.hidden_state_size),
                               self.position_embedding_op,
                               self.position_length_op,
                               self.keyword_number)

    def sequence_rnn_op(self):
        return rnn_util.rnn(self.sequence_cell_op, self.position_embedding_op, self.sequence_rnn_length_op)

    @tf_util.define_scope("loss_op")
    def loss_op(self):
        output_logit, _ = self.sequence_rnn_op
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
                                         global_step=self.global_step_variable)


