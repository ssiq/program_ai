from common import tf_util, code_util ,rnn_util, util

import tensorflow as tf

from common.rnn_cell import RNNWrapper

def cast_float(x):
    return tf.cast(x, tf.float32)

class QuestionAwareSelfMatchAttentionWrapper(RNNWrapper):
    def __init__(self,
                 cell: tf.nn.rnn_cell.RNNCell,
                 memory,
                 memory_length,
                 question,
                 attention_size,
                 reuse=False):
        super().__init__(cell, reuse)
        self._memory = memory
        self._memory_length = memory_length
        self._attention_size = attention_size
        self._question = question

    def call(self, inputs, state):
        with tf.variable_scope("self_match_attention"):
            inputs = util.convert_to_list(inputs)
            atten = rnn_util.soft_attention_reduce_sum(self._memory,
                                                              [inputs],
                                                              self._attention_size,
                                                              self._memory_length)
            atten = util.convert_to_list(atten)

            print("Self match, input:{}, atten:{}\n, added:{}".format(inputs, atten, inputs+atten))
            inputs = tf.concat(inputs + atten, axis=1)
            inputs = tf_util.weight_multiply("gate_weight", inputs, tf_util.get_shape(inputs)[1])
            return self._cell(inputs, state)

class TokenLevelMultiRnnModelGraph(tf_util.BaseModel):
    def __init__(self,
                 word_embedding_layer_fn,
                 character_embedding_layer_fn,
                 hidden_size,
                 rnn_layer_number,
                 output_layer_number,
                 keyword_number,
                 identifier_token,
                 learning_rate,
                 decay_steps,
                 decay_rate,
                 placeholders):
        super().__init__(learning_rate, decay_steps, decay_rate)
        self.word_embedding_layer_fn = word_embedding_layer_fn
        self.character_embedding_layer_fn = character_embedding_layer_fn
        self.hidden_state_size = hidden_size
        self.rnn_layer_number = rnn_layer_number
        self.identifier_token = identifier_token
        self.output_layer_number = output_layer_number
        self.keyword_number = keyword_number
        self.placeholders = placeholders
        self.token_input, self.token_input_length, self.character_input, self.character_input_length, \
        self.token_input_mask, self.output_is_continue, self.output_position_label, self.output_is_copy, \
        self.output_keyword_id, self.output_copy_word_id = placeholders

    def _rnn_cell(self, hidden_size):
        return tf.nn.rnn_cell.GRUCell(hidden_size)

    def _multi_rnn_cell(self, hidden_size, layer_number):
        return tf.nn.rnn_cell.MultiRNNCell([self._rnn_cell(hidden_size) for _ in range(layer_number)])

    @tf_util.define_scope("word_embedding_op")
    def word_embedding_op(self):
        input_embedding_op = self.word_embedding_layer_fn(self.token_input)
        return input_embedding_op

    @tf_util.define_scope("character_embedding_op")
    def character_embedding_op(self):
        input_embedding_op = self.character_embedding_layer_fn(self.character_input, self.character_input_length)
        return input_embedding_op

    @tf_util.define_scope("code_embedding_op")
    def code_embedding_op(self): #shape:[batch, length, dim]
        token_embedding = self.word_embedding_op
        character_embedding = self.character_embedding_op
        return code_util.code_embedding(token_embedding, character_embedding, self.token_input, self.identifier_token)

    @tf_util.define_scope("bi_rnn_op")
    def bi_rnn_op(self): #shape:[batch, length, dim]
        cell_fn = lambda: self._multi_rnn_cell(self.hidden_state_size, self.rnn_layer_number)
        return rnn_util.concat_bi_rnn_output(rnn_util.bi_rnn(
            cell_fn,
            self.code_embedding_op,
            self.token_input_length,
        ))

    @tf_util.define_scope("question_variable")
    def question_variable_op(self):
        return tf.Variable(
            initial_value=tf.zeros_like(self.bi_rnn_op[0, 0, :]),
            trainable=True,
            name="question_variable",
            dtype=tf.float32,
        )

    @tf_util.define_scope("self_matched_code")
    def self_matched_code_op(self):
        return rnn_util.bi_rnn(
            lambda: QuestionAwareSelfMatchAttentionWrapper(
                self._multi_rnn_cell(self.hidden_state_size, self.rnn_layer_number),
                self.bi_rnn_op,
                self.token_input_length,
                self.question_variable_op,
                self.hidden_state_size,
            ),
            self.bi_rnn_op,
            self.token_input_length
        )

    @tf_util.define_scope("self_matched_code")
    def self_matched_code_output_op(self):
        return self.self_matched_code_op[0]

    @tf_util.define_scope("self_matched_code")
    def self_matched_code_final_state_op(self):
        return rnn_util.concat_bi_rnn_final_state(self.self_matched_code_op)

    @tf_util.define_scope("is_continue")
    def is_continue_logit_op(self):
        o = self.self_matched_code_final_state_op
        name = "is_continue"
        o = self._forward_network(name, o, 1)
        return o

    def _forward_network(self, name, o, classes):
        for i in range(self.output_layer_number):
            o = tf.nn.relu(tf_util.dense(o, self.hidden_state_size, name="{}_{}".format(name, i)))
        o = tf_util.dense(o, classes, name=name)
        return o

    @tf_util.define_scope("position")
    def position_forward_op(self):
        with tf.variable_scope("forward"):
            forward = tf.contrib.layers.fully_connected(self.self_matched_code_op[0][0], 1)
        return forward

    @tf_util.define_scope("position")
    def position_backward_op(self):
        with tf.variable_scope("backward"):
            backward = tf.contrib.layers.fully_connected(self.self_matched_code_op[0][1], 1)
        return backward

    @tf_util.define_scope("position")
    def position_logit_op(self):
        o = code_util.position_embedding(self.position_forward_op, self.position_backward_op, self.token_input_length)
        return tf.contrib.layers.fully_connected(o, 1, None)

    @tf_util.define_scope("output")
    def output_state_op(self):
        o = tf.reduce_sum(self.position_forward_op * self.self_matched_code_output_op[0], axis=1)
        o += tf.reduce_sum(self.position_backward_op * self.self_matched_code_output_op[1], axis=1)
        return o

    @tf_util.define_scope("is_copy")
    def is_copy_logit_op(self):
        return self._forward_network("is_copy", self.output_state_op, 1)

    @tf_util.define_scope("keyword")
    def keyword_logit_op(self):
        return self._forward_network("keyword", self.output_state_op, self.keyword_number)

    @tf_util.define_scope("copy_word")
    def copy_word_logit_op(self):
        with tf.variable_scope("forward"):
            forward = tf.contrib.layers.fully_connected(self.self_matched_code_op[0][0], 1, None)
        with tf.variable_scope("backward"):
            backward = tf.contrib.layers.fully_connected(self.self_matched_code_op[0][0], 1, None)
        o = tf.nn.relu(forward + backward)
        return o

    @tf_util.define_scope("copy_word")
    def copy_word_mask_op(self):
        return self.token_input_mask

    @tf_util.define_scope("copy_word")
    def copy_word_masked_logit_op(self):
        original_logit = self.copy_word_logit_op
        original_logit = tf.expand_dims(original_logit, axis=1)
        masked_logit = tf.matmul(original_logit, self.copy_word_mask_op)
        masked_logit = tf.squeeze(masked_logit, axis=[1])
        return masked_logit

    @tf_util.define_scope("copy_word")
    def copy_word_token_number_op(self):
        o = tf.greater(tf.reduce_sum(self.copy_word_mask_op, axis=1), tf.constant(0, dtype=tf.int32))
        o = tf.reduce_sum(o, axis=1)
        return o

    @tf_util.define_scope("copy_word")
    def copy_word_softmax_op(self):
        return tf_util.variable_length_softmax(self.copy_word_masked_logit_op, self.copy_word_token_number_op)

    @tf_util.define_scope("loss")
    def loss_op(self):
        loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
            cast_float(self.output_is_continue), self.is_continue_logit_op
        ))
        loss += tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
            cast_float(self.output_is_copy), self.is_copy_logit_op
        ))
        loss += tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            self.output_position_label, self.position_logit_op
        ))
        loss += tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            self.output_copy_word_id, self.copy_word_logit_op
        ))
        loss += tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            self.output_keyword_id, self.keyword_logit_op
        ))
        return loss

    @tf_util.define_scope("predict")
    def predict_op(self):
        return tf.nn.sigmoid(self.is_continue_logit_op), \
               tf_util.variable_length_softmax(self.position_logit_op, 2*self.token_input_length+1), \
               tf.nn.sigmoid(self.is_copy_logit_op), \
               tf.nn.softmax(self.keyword_logit_op), \
               self.copy_word_softmax_op

class TokenLevelMultiRnnModel(object):
    def __init__(self,
                 word_embedding_layer_fn,
                 character_embedding_layer_fn,
                 hidden_size,
                 rnn_layer_number,
                 keyword_number,
                 end_token_id,
                 learning_rate,
                 decay_steps,
                 decay_rate,
                 output_layer_num,
                 identifier_token,
                 placeholder_token,
                 id_to_word_fn,
                 parse_token_fn,
                 ):
        self.word_embedding_layer_fn = word_embedding_layer_fn
        self.character_embedding_layer_fn = character_embedding_layer_fn
        self.hidden_state_size = hidden_size
        self.rnn_layer_number = rnn_layer_number
        self.keyword_number = keyword_number
        self.end_token_id = end_token_id
        self.identifier_token = identifier_token
        self.placeholder_token = placeholder_token
        self.id_to_word = id_to_word_fn
        self.parse_token = parse_token_fn
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.output_layer_num = output_layer_num

        #input
        self.token_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name="token_input")
        self.token_input_length = tf.placeholder(dtype=tf.int32, shape=(None, ), name="token_input_length")
        self.character_input = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name="character_input")
        self.character_input_length = tf.placeholder(dtype=tf.int32, shape=(None, None),
                                                     name="character_input_length")
        self.token_input_mask = tf.placeholder(dtype=tf.int32, shape=(None, None, None),
                                                     name="token_input_mask")

        self.input_placeholders = [self.token_input, self.token_input_length, self.character_input, self.character_input_length,
         self.token_input_mask]
        #output
        self.output_is_continue = tf.placeholder(dtype=tf.int32, shape=(None, ), name="output_length")
        self.output_position_label = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_position")
        self.output_is_copy = tf.placeholder(dtype=tf.int32, shape=(None, ),
                                             name="output_is_copy")  # 1 means using copy
        self.output_keyword_id = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_keyword_id")
        self.output_copy_word_id = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_copy_word_id")
        self.output_placeholders = [self.output_is_continue, self.output_position_label, self.output_is_copy,
        self.output_keyword_id, self.output_copy_word_id]

        self._model = TokenLevelMultiRnnModelGraph(
            self.word_embedding_layer_fn,
            self.character_embedding_layer_fn,
            self.hidden_state_size,
            self.rnn_layer_number,
            self.output_layer_num,
            self.keyword_number,
            self.identifier_token,
            self.learning_rate,
            self.decay_steps,
            self.decay_rate,
            self.input_placeholders + self.output_placeholders
        )

        tf_util.init_all_op(self._model)
        sess = tf_util.get_session()
        init = tf.global_variables_initializer()
        sess.run(init)

    @property
    def model(self):
        return self._model

    def _train(self, *args):
        """
        :param args:
        :return: (loss, loss,train)
        """
        fn = tf_util.function(self.input_placeholders+self.output_placeholders,
                              [self._model.loss_op, self._model.loss_op, self._model.train_op,])
        return fn(*args)

    def _loss_fn(self, *args):
        fn = tf_util.function(self.input_placeholders+self.output_placeholders,
                              self._model.loss_op, )
        return fn(*args)

    def _one_predict_fn(self, *args):
        fn = tf_util.function(self.input_placeholders,
                              self._model.predict_op, )
        return fn(*args)









