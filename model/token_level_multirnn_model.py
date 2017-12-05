import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest

from common import tf_util, rnn_cell, rnn_util, code_util
from common.rnn_util import create_decoder_initialize_fn, create_decode


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
        loss_input_placeholder = tf.placeholder(tf.float32, shape=[], name="loss")
        metrics_input_placeholder = tf.placeholder(tf.float32, shape=[], name="metrics")
        tf_util.add_summary_scalar("loss", loss_input_placeholder, is_placeholder=True)
        tf_util.add_summary_scalar("metrics", metrics_input_placeholder, is_placeholder=True)
        self._summary_fn = tf_util.placeholder_summary_merge()

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
        output_logit, _, _ = self.decode_op
        output_logit, _ = output_logit
        print("output_logit:{}".format(output_logit))
        is_continue_logit, position_logit, is_copy_logit, key_word_logit, copy_word_logit = output_logit
        loss = tf.reduce_mean(
            tf.multiply(tf.cast(self.output_is_continue, tf.float32), tf.nn.sigmoid_cross_entropy_with_logits(logits=is_continue_logit,
                                                                                         labels=tf.cast(
                                                                                             self.output_is_continue,
                                                                                             tf.float32))))
        loss += tf.reduce_mean(
            tf.multiply(tf.cast(self.output_is_continue, tf.float32), tf.nn.sigmoid_cross_entropy_with_logits(logits=is_copy_logit,
                                                                                         labels=tf.cast(
                                                                                             self.output_is_copy,
                                                                                             tf.float32))))
        sparse_softmax_loss = lambda x, y: tf.reduce_mean(tf.multiply(tf.cast(self.output_is_continue, tf.float32),tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x,logits=y)))
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

    def train_model(self, *args):
        # print("train_model_input:")
        # for t in args:
        #     print(np.array(t).shape)
            # print("{}:{}".format(np.array(t).shape, np.array(t)))
        # l1, l2, _ = self.train(*args)
        # print(l1)
        # return l1, l2, None
        loss, _, train = self.train(*args)
        metrics = self.metrics_model(*args)
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
        loss = self._loss_fn(*args)
        return self._summary_fn(loss, loss)

    def _create_next_input(self, output, position_embedding, code_embedding):
        _, position, is_copy, key_word, copy_word = output
        return self._create_next_input_fn([position, is_copy, key_word, copy_word, position_embedding, code_embedding])

    def _create_next_code(self, actions, token_input, token_input_length, character_input, character_input_length):
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
            else:
                if is_copy:
                    if copy_id >= code_length:
                        # copy position error
                        print('copy position error', copy_id, code_length)
                        continue
                    word_token_id = token_input[i][0][copy_id]
                    word_character_id = character_input[i][0][copy_id]
                    word_character_length = character_input_length[i][0][copy_id]
                else:
                    word_token_id = keyword_id
                    word = self.id_to_word(word_token_id)
                    if word == None:
                        # keyword id error
                        print('keyword id error', keyword_id)
                        continue
                    word_character_id = self.parse_token(word, character_position_label=True)
                    word_character_length = len(word_character_id)

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

        return token_input, token_input_length, character_input, character_input_length

    def predict_model(self, *args):
        # print('predict iterator start')
        token_input, token_input_length, charactere_input, character_input_length = args
        start_label, initial_state = self._initial_state_and_initial_label_fn(*args)
        next_state = initial_state

        start_labels = []
        for i in range(len(initial_state)):
            start_labels.append(start_label)

        output_is_continues = np.array([[] for i in range(len(token_input))])
        output_position = np.array([[] for i in range(len(token_input))])
        output_is_copy = np.array([[] for i in range(len(token_input))])
        output_keyword_id = np.array([[] for i in range(len(token_input))])
        output_copy_id = np.array([[] for i in range(len(token_input))])

        from common.util import padded

        for i in range(self.max_decode_iterator_num):
            output, next_state, position_embedding, code_embedding \
                = self._one_predict_fn(
                token_input,
                token_input_length,
                charactere_input,
                character_input_length,
                next_state,
                start_labels
            )
            is_continue, position, is_copy, keyword_id, copy_id = output
            is_continue = np.array(is_continue)
            position = np.argmax(np.array(position), axis=1)
            is_copy = np.array(is_copy)
            keyword_id = np.argmax(np.array(keyword_id), axis=1)
            copy_id = np.argmax(np.array(copy_id), axis=1)

            output_is_continues = np.concatenate((output_is_continues, np.expand_dims(is_continue, axis=1)), axis=1)
            output_position = np.concatenate((output_position, np.expand_dims(position, axis=1)), axis=1)
            output_is_copy = np.concatenate((output_is_copy, np.expand_dims(is_copy, axis=1)), axis=1)
            output_keyword_id = np.concatenate((output_keyword_id, np.expand_dims(keyword_id, axis=1)), axis=1)
            output_copy_id = np.concatenate((output_copy_id, np.expand_dims(copy_id, axis=1)), axis=1)

            if np.sum(is_continue) == 0:
                break
            actions = (is_continue, position, is_copy, keyword_id, copy_id)
            token_input, token_input_length, charactere_input, character_input_length = self._create_next_code(actions, token_input, token_input_length, charactere_input, character_input_length)

            token_input = padded(token_input)
            token_input_length = padded(token_input_length)
            charactere_input = padded(charactere_input)
            character_input_length = padded(character_input_length)

        return output_is_continues, output_position, output_is_copy, output_keyword_id, output_copy_id

    def metrics_model(self, *args):
        import copy
        args = copy.deepcopy(args)
        input_data = args[0:4]
        output_data = args[4:9]
        predict_data = self.predict_model(*input_data)
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
