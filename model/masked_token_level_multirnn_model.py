import tensorflow as tf
from tensorflow.python.ops.losses.losses_impl import Reduction
import numpy as np

from common import tf_util, code_util, rnn_util, util
from common.rnn_cell import RNNWrapper
from common.tf_util import cast_float, cast_int
from common.beam_search_util import beam_cal_top_k, flat_list, \
    select_max_output, revert_batch_beam_stack, beam_calculate, _create_next_code_without_iter_dims, cal_metrics, find_copy_input_position


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
        self._question = tf.expand_dims(question, axis=-1)

    def call(self, inputs, state):
        with tf.variable_scope("self_match_attention"):
            inputs = util.convert_to_list(inputs)
            inputs = [tf.matmul(t, self._question) * t for t in inputs]
            atten = rnn_util.soft_attention_reduce_sum(self._memory,
                                                              [inputs],
                                                              self._attention_size,
                                                              self._memory_length)
            atten = util.convert_to_list(atten)

            print("Self match, input:{}, atten:{}\n, added:{}".format(inputs, atten, inputs+atten))
            inputs = tf.concat(inputs + atten, axis=1)
            inputs = tf_util.weight_multiply("gate_weight", inputs, tf_util.get_shape(inputs)[1])
            return self._cell(inputs, state)

class MaskedTokenLevelMultiRnnModelGraph(tf_util.BaseModel):
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
    @tf_util.debug_print("word_embedding_op value:")
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
    @tf_util.debug_print("self_match_code_output:")
    def self_matched_code_output_op(self):
        return self.self_matched_code_op[0]

    @tf_util.define_scope("self_matched_code")
    def self_matched_code_final_state_op(self):
        return rnn_util.concat_bi_rnn_final_state(self.self_matched_code_op)

    @tf_util.define_scope("is_continue")
    @tf_util.debug_print("is_continue_logit:")
    def is_continue_logit_op(self):
        o = self.self_matched_code_final_state_op
        name = "is_continue"
        o = self._forward_network(name, o, 1)
        return tf.squeeze(o, axis=[-1])

    def _forward_network(self, name, o, classes):
        for i in range(self.output_layer_number):
            o = tf.nn.relu(tf_util.dense(o, self.hidden_state_size, name="{}_{}".format(name, i)))
        o = tf_util.dense(o, classes, name=name)
        return o

    @tf_util.define_scope("position", initializer=tf.contrib.layers.xavier_initializer())
    @tf_util.debug_print("position_forward:")
    def position_forward_op(self):
        with tf.variable_scope("forward", ):
            forward = tf.contrib.layers.fully_connected(self.self_matched_code_op[0][0], 1)
        return forward

    @tf_util.define_scope("position", initializer=tf.contrib.layers.xavier_initializer())
    @tf_util.debug_print("position_backward:")
    def position_backward_op(self):
        with tf.variable_scope("backward"):
            backward = tf.contrib.layers.fully_connected(self.self_matched_code_op[0][1], 1)
        return backward

    @tf_util.define_scope("position", initializer=tf.contrib.layers.xavier_initializer())
    @tf_util.debug_print("position_logit:")
    def position_logit_op(self):
        o = code_util.position_embedding(self.position_forward_op, self.position_backward_op, self.token_input_length)
        return tf.squeeze(tf.contrib.layers.fully_connected(o, 1, None), axis=[-1])

    @tf_util.define_scope("output", initializer=tf.contrib.layers.xavier_initializer())
    @tf_util.debug_print("output_state:")
    def output_state_op(self):
        o = tf.reduce_sum(self.position_forward_op * self.self_matched_code_output_op[0], axis=1)
        o += tf.reduce_sum(self.position_backward_op * self.self_matched_code_output_op[1], axis=1)
        return o

    @tf_util.define_scope("is_copy")
    @tf_util.debug_print("is_copy_logit:")
    def is_copy_logit_op(self):
        return tf.squeeze(self._forward_network("is_copy", self.output_state_op, 1), axis=[-1])

    @tf_util.define_scope("keyword")
    @tf_util.debug_print("keyword_logit:")
    def keyword_logit_op(self):
        return self._forward_network("keyword", self.output_state_op, self.keyword_number)

    @tf_util.define_scope("copy_word", initializer=tf.contrib.layers.xavier_initializer())
    @tf_util.debug_print("copy_word_logit:")
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
    @tf_util.debug_print("copy_word_masked_logit:")
    def copy_word_masked_logit_op(self):
        original_logit = self.copy_word_logit_op
        original_logit = tf.squeeze(original_logit, axis=[-1])
        original_logit = tf.expand_dims(original_logit, axis=1)
        masked_logit = tf.matmul(original_logit, cast_float(self.copy_word_mask_op))
        masked_logit = tf.squeeze(masked_logit, axis=[1])
        return masked_logit

    @tf_util.define_scope("copy_word")
    def copy_word_token_number_op(self):
        o = tf.greater(tf.reduce_sum(self.copy_word_mask_op, axis=1), tf.constant(0, dtype=tf.int32))
        o = tf.reduce_sum(cast_int(o), axis=1)
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
        copyword_loss = tf.losses.sparse_softmax_cross_entropy(
            self.output_copy_word_id, self.copy_word_logit_op, reduction=Reduction.NONE
        )
        keyword_loss = tf.losses.sparse_softmax_cross_entropy(
            self.output_keyword_id, self.keyword_logit_op, reduction=Reduction.NONE
        )
        loss += tf.reduce_mean(tf.where(tf_util.cast_bool(self.output_is_copy), x=copyword_loss, y=keyword_loss))
        return loss

    @tf_util.define_scope("predict")
    def predict_op(self):
        return tf.nn.sigmoid(self.is_continue_logit_op), \
               tf_util.variable_length_softmax(self.position_logit_op, 2*self.token_input_length+1), \
               tf.nn.sigmoid(self.is_copy_logit_op), \
               tf.nn.softmax(self.keyword_logit_op), \
               self.copy_word_softmax_op

    @tf_util.define_scope("predict")
    def argmax_predict_op(self):
        argmax_copy_word = cast_int(tf.argmax(self.predict_op[3], axis=1))
        argmax_keyword = cast_int(tf.argmax(self.predict_op[4], axis=1))
        is_copy = tf.greater(self.predict_op[2], 0.5)
        return cast_int(tf.greater(self.predict_op[0], 0.5)), \
               cast_int(tf.argmax(self.predict_op[1], axis=1)), \
               cast_int(is_copy), \
               tf.where(tf_util.cast_bool(is_copy), x=argmax_copy_word, y=argmax_keyword), \
               tf.where(tf_util.cast_bool(is_copy), y=argmax_copy_word, x=argmax_keyword)

    @tf_util.define_scope("metrics")
    def metrics_op(self):
        # m = tf_util.cast_bool(tf.ones_like(self.output_is_copy))
        # for a, b in zip(self.argmax_predict_op,
        #                 [self.output_is_continue, self.output_position_label, self.output_is_copy,
        #                  self.output_copy_word_id, self.output_keyword_id]):
        #     m = tf.logical_and(m, tf.equal(a, b))
        pad_dims = lambda x: [tf.expand_dims(t, axis=-1) for t in x]
        predicts = tf.concat(pad_dims(self.argmax_predict_op), axis=1)
        labels = tf.concat(pad_dims([self.output_is_continue, self.output_position_label, self.output_is_copy,
                         self.output_copy_word_id, self.output_keyword_id]), axis=1)
        accuracy, accuracy_updates = tf.metrics.accuracy(labels=labels,
                            predictions=predicts)
        return accuracy, accuracy_updates

class MaskedTokenLevelMultiRnnModel(object):
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
        self.output_position_label = tf.placeholder(dtype=tf.int32, shape=(None, ), name="output_position")
        self.output_is_copy = tf.placeholder(dtype=tf.int32, shape=(None, ),
                                             name="output_is_copy")  # 1 means using copy
        self.output_keyword_id = tf.placeholder(dtype=tf.int32, shape=(None, ), name="output_keyword_id")
        self.output_copy_word_id = tf.placeholder(dtype=tf.int32, shape=(None, ), name="output_copy_word_id")
        self.output_placeholders = [self.output_is_continue, self.output_position_label, self.output_is_copy,
        self.output_keyword_id, self.output_copy_word_id]

        self._model = MaskedTokenLevelMultiRnnModelGraph(
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

        metrics_input_placeholder = tf.placeholder(tf.float32, shape=[], name="metrics")
        tf_util.add_summary_scalar("metrics", metrics_input_placeholder, is_placeholder=True)
        tf_util.add_summary_histogram("predict_is_continue",
                                      tf.placeholder(tf.float32, shape=(None, ), name="predict_is_continue"),
                                      is_placeholder=True)
        tf_util.add_summary_histogram("predict_position_softmax",
                                      tf.placeholder(tf.float32, shape=(None, None),
                                                     name="predict_position_softmax"),
                                      is_placeholder=True)
        tf_util.add_summary_histogram("predict_is_copy",
                                      tf.placeholder(tf.float32, shape=(None, ), name="predict_is_copy"),
                                      is_placeholder=True)
        tf_util.add_summary_histogram("predict_key_word",
                                      tf.placeholder(tf.float32, shape=(None, None), name="predict_keyword"),
                                      is_placeholder=True)
        tf_util.add_summary_histogram("predict_copy_word",
                                      tf.placeholder(tf.float32, shape=(None, None), name="predict_copy_word"),
                                      is_placeholder=True)
        tf_util.add_summary_scalar("loss", tf.placeholder(tf.float32, shape=[]), is_placeholder=True)
        tf_util.add_summary_histogram("is_continue", self._model.predict_op[0], is_placeholder=False)
        tf_util.add_summary_histogram("position_softmax", self._model.predict_op[1], is_placeholder=False)
        tf_util.add_summary_histogram("is_copy", self._model.predict_op[2], is_placeholder=False)
        tf_util.add_summary_histogram("key_word", self._model.predict_op[3], is_placeholder=False)
        tf_util.add_summary_histogram("key_word", self._model.predict_op[4], is_placeholder=False)
        self._summary_fn = tf_util.placeholder_summary_merge()
        self._summary_merge_op = tf_util.merge_op()

        sess = tf_util.get_session()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        self._train_summary_fn = tf_util.function(
            self.input_placeholders + self.output_placeholders,
            self._summary_merge_op
        )

        self._train = tf_util.function(self.input_placeholders+self.output_placeholders,
                              [self._model.loss_op, self._model.metrics_op[0], self._model.train_op,],
                                       updates=[self._model.metrics_op[1]])

        self._loss_fn = tf_util.function(self.input_placeholders+self.output_placeholders,
                              self._model.loss_op, )

        self._one_predict_fn = tf_util.function(self.input_placeholders,
                              self._model.predict_op, )

    @property
    def model(self):
        return self._model

    def summary(self, *args):
        train_summary = self._train_summary_fn(*args)
        metrics_model = self.metrics_model(*args)
        tf_util.add_value_scalar("metrics", metrics_model)
        return self._summary_fn(*tf_util.merge_value(), summary_input=train_summary)

    @property
    def global_step(self):
        return self._model.global_step

    def metrics_model(self, *args):
        # print("metrics input")
        # for t in args:
        #     print(np.array(t).shape)
        max_decode_iterator_num = 5
        input_data = args[0:5]
        output_data = args[5:10]
        predict_data = self.predict_model(*input_data, )
        metrics_value = cal_metrics(max_decode_iterator_num, output_data, predict_data)
        # print('metrics_value: ', metrics_value)
        return metrics_value

    def train_model(self, *args):
        # for t in args:
        #     print(np.array(t).shape)
            # print(t)
        # print(self.input_placeholders+self.output_placeholders)


        flat_args = [flat_list(one_input) for one_input in args]
        batch_size = len(args[0])
        total = len(flat_args[0])
        weight_array = int(total / batch_size) * [batch_size]
        if total%batch_size != 0:
            weight_array = weight_array + [total % batch_size]

        import more_itertools
        chunked_args = more_itertools.chunked(list(zip(*flat_args)), batch_size)
        train_chunk_fn = lambda chunked: self._train(*list(zip(*chunked)))
        train_res = map(train_chunk_fn, chunked_args)
        train_res = list(zip(*train_res))

        weight_fn = lambda one_res: [one_res[i] * weight_array[i] for i in range(len(one_res))]
        loss_array = weight_fn(train_res[0])
        accracy_array = weight_fn(train_res[1])
        # tt_array = weight_fn(train_res[2])
        loss = np.sum(loss_array) / total
        accracy = np.sum(accracy_array) / total
        # tt = np.sum(tt_array) / total

        # loss, accracy, tt = self._train(*args)
        return loss, accracy, None

    def predict_model(self, *args,):
        # print('predict iterator start')
        import copy
        import more_itertools
        # print('before args shape: ')
        # for i in range(5):
        #     print(np.array(args[i]).shape)
        args = [copy.deepcopy([ti[0] for ti in one_input]) for one_input in args]
        # print('args shape: ')
        # for i in range(5):
        #     print(np.array(args[i]).shape)
        # token_input, token_input_length, charactere_input, character_input_length, identifier_mask = args
        # start_label, initial_state = self._initial_state_and_initial_label_fn(*args)
        batch_size = len(args[0])
        cur_beam_size = 1
        beam_size = 5
        max_decode_iterator_num = 5

        # shape = 5 * batch_size * beam_size * token_length
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
        # next_labels_stack = []
        # shape = batch_size * beam_size * initial_state_length
        # next_states_stack = []

        # next_states_stack = np.expand_dims(np.array(initial_state), axis=1).tolist()
        # next_labels_stack = [[start_label]] * len(initial_state)

        for i in range(max_decode_iterator_num):

            input_flat = [flat_list(inp) for inp in input_stack]
            # next_labels_flat = flat_list(next_labels_stack)
            # next_states_flat = flat_list(next_states_stack)

            one_predict_fn = lambda chunked: self._one_predict_fn(*list(zip(*chunked)))

            chunked_input = more_itertools.chunked(list(zip(*input_flat)), batch_size)
            predict_returns = list(map(one_predict_fn, chunked_input))
            predict_returns = list(zip(*predict_returns))
            outputs = predict_returns

            output_list = [flat_list(out) for out in outputs]
            # state_list = flat_list(next_state)
            # position_embedding_list = flat_list(position_embedding)
            # code_embedding_list = flat_list(code_embedding)

            output_stack = [revert_batch_beam_stack(out_list, batch_size, cur_beam_size) for out_list in output_list]
            # next_states_stack = revert_batch_beam_stack(state_list, batch_size, cur_beam_size)
            # position_embedding_stack = revert_batch_beam_stack(position_embedding_list, batch_size, cur_beam_size)
            # code_embedding_stack = revert_batch_beam_stack(code_embedding_list, batch_size, cur_beam_size)

            # beam_args = (list(zip(*input_stack)), list(zip(*output_stack)), beam_stack, next_states_stack,
            #              position_embedding_stack, code_embedding_stack, mask_stack, beam_length_stack,
            #              list(zip(*select_output_stack_list)), [beam_size] * batch_size)
            # batch_returns = list(util.parallel_map(core_num=3, f=beam_calculate_fn, args=list(zip(*beam_args))))
            batch_returns = list(map(beam_calculate, list(zip(*input_stack)), list(zip(*output_stack)), beam_stack, mask_stack, beam_length_stack, list(zip(*select_output_stack_list)), [beam_size] * batch_size, [beam_calculate_output_score] * batch_size, [[]]*batch_size))
            def create_next(ret):
                ret = list(ret)
                ret[0] = _create_next_code_without_iter_dims(ret[1], ret[0], create_one_fn=self._create_one_next_code)
                return ret
            batch_returns = [create_next(ret) for ret in batch_returns]
            input_stack, output_stack, select_output_stack_list, mask_stack, beam_stack, beam_length_stack, _ = list(zip(*batch_returns))
            input_stack = list(zip(*input_stack))
            output_stack = list(zip(*output_stack))
            select_output_stack_list = list(zip(*select_output_stack_list))

            if np.sum(output_stack[0]) == 0:
                break

            # output_flat = [flat_list(out) for out in output_stack]
            # position_embedding_flat = flat_list(position_embedding_stack)
            # code_embedding_flat = flat_list(code_embedding_stack)

            # create_label_lambda_fn = lambda is_continue, position, is_copy, keyword_id, copy_id, position_emb, code_emb: self._create_next_input_fn(np.expand_dims(position, axis=1), np.expand_dims(is_copy, axis=1),
            #                                              np.expand_dims(keyword_id, axis=1), np.expand_dims(copy_id, axis=1), position_emb, code_emb)[:, 0, :]
            # create_label_lambda_chunked_fn = lambda chunked: create_label_lambda_fn(*list(zip(*chunked)))
            # next_labels_stack = list(map(create_label_lambda_chunked_fn, more_itertools.chunked(list(zip(*list(output_flat + [position_embedding_flat] + [code_embedding_flat]))), batch_size)))
            # next_labels_stack = flat_list(next_labels_stack)
            # next_labels_stack = np.reshape(next_labels_stack, (batch_size, beam_size, -1)).tolist()
            # [create_label_lambda_fn() for is_continue, position, is_copy, keyword_id, copy_id in more_itertools.chunked(list(zip(*output_stack)) + [position_embedding_stack] + [code_embedding_stack], batch_size)]

            input_stack = [[list(inp) for inp in one_input]for one_input in input_stack]

            # print('before inputs shape: ')
            # for i in range(5):
            #     print(np.array(input_stack[i]).shape)

            input_stack = [list(util.padded(list(inp))) for inp in input_stack]
            # print('after padded inputs shape: ')
            # for i in range(5):
            #     print(np.array(input_stack[i]).shape)
                # if len(np.array(input_stack[i]).shape) <= 2:
                #     print(input_stack[i])
            mask_input_with_end_fn = lambda token_input: list([util.mask_input_with_end(batch_mask, batch_inp, n_dim=1).tolist() for batch_mask, batch_inp in zip(mask_stack, token_input)])
            input_stack = list(map(mask_input_with_end_fn, input_stack))
            cur_beam_size = beam_size

            # print('after inputs shape: ')
            # for i in range(5):
            #     print(np.array(input_stack[i]).shape)


        summary = copy.deepcopy(select_output_stack_list)
        tf_util.add_value_histogram("predict_is_continue", util.padded(summary[0]))
        tf_util.add_value_histogram("predict_position_softmax", util.padded(summary[1]))
        tf_util.add_value_histogram("predict_is_copy", util.padded(summary[2]))
        tf_util.add_value_histogram("predict_key_word", util.padded(summary[3]))
        tf_util.add_value_histogram("predict_copy_word", util.padded(summary[4]))

        final_output = select_max_output(beam_stack, select_output_stack_list)
        return final_output

    def _create_one_next_code(self, action, token_input, token_input_length, character_input, character_input_length, identifier_mask):
        is_continue, position, is_copy, keyword_id, copy_id = action
        next_inputs = token_input, token_input_length, character_input, character_input_length, identifier_mask
        code_length = token_input_length

        if position % 2 == 1 and is_copy == 0 and keyword_id == self.placeholder_token:
            # delete
            position = int(position / 2)
            if position >= code_length:
                # action position error
                print('delete action position error', position, code_length)
                return next_inputs
            token_input = token_input[0:position] + token_input[position + 1:]
            token_input_length -= 1
            character_input = character_input[0:position] + character_input[position + 1:]
            character_input_length = character_input_length[0:position] + character_input_length[position + 1:]
            identifier_mask = identifier_mask[0:position] + identifier_mask[position + 1:]
        else:
            if is_copy:
                copy_position_id = find_copy_input_position(identifier_mask, copy_id)
                if copy_position_id >= code_length:
                    # copy position error
                    print('copy position error', copy_position_id, code_length)
                    print('details:', position, is_copy, keyword_id, copy_position_id, code_length)
                    return next_inputs
                word_token_id = token_input[copy_position_id]
                word_character_id = character_input[copy_position_id]
                word_character_length = character_input_length[copy_position_id]
                iden_mask = identifier_mask[copy_position_id]
            else:
                word_token_id = keyword_id
                word = self.id_to_word(word_token_id)
                if word == None:
                    # keyword id error
                    print('keyword id error', keyword_id)
                    return next_inputs
                word_character_id = self.parse_token(word, character_position_label=True)
                word_character_length = len(word_character_id)
                iden_mask = [0] * len(identifier_mask[0])

            if position % 2 == 0:
                # insert
                position = int(position / 2)
                if position > code_length:
                    # action position error
                    print('insert action position error', position, code_length)
                    return next_inputs
                token_input = token_input[0:position] + [word_token_id] + token_input[position:]
                token_input_length += 1
                character_input = character_input[0:position] + [word_character_id] + character_input[position:]
                character_input_length = character_input_length[0:position] + [word_character_length] + character_input_length[position:]
                identifier_mask = identifier_mask[0:position] + [iden_mask] + identifier_mask[position:]
            elif position % 2 == 1:
                # change
                position = int(position / 2)
                if position >= code_length:
                    # action position error
                    print('change action position error', position, code_length)
                    return next_inputs
                token_input[position] = word_token_id
                character_input[position] = word_character_id
                character_input_length[position] = word_character_length
                identifier_mask[position] = iden_mask
        next_inputs = token_input, token_input_length, character_input, character_input_length, identifier_mask
        return next_inputs


def beam_calculate_output_score(output_beam_list, beam_size):
    import math

    output_is_continues, output_positions, output_is_copys, output_keyword_ids, output_copy_ids = output_beam_list
    cur_beam_size = len(output_positions)
    # print('cur_beam_size:',cur_beam_size)
    beam_action_stack = [[] for i in range(beam_size)]
    beam_p_stack = [[] for i in range(beam_size)]
    beam_id_stack = [[] for i in range(beam_size)]

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

                            # action = {'is_continue': is_continue, 'position': position_id, 'is_copy': is_copy,
                            #           'keyword': keyword, 'copy_id': copy_id, 'beam_id': beam_id}
                            # print(is_continue_p, position_p, is_copy_p, copy_id_p)
                            p = math.log(is_continue_p) + math.log(position_p) + math.log(is_copy_p) + math.log(copy_id_p)

                            beam_action_stack[beam_id].append((is_continue, position_id, is_copy, keyword, copy_id))
                            beam_p_stack[beam_id].append(p)
                            beam_id_stack[beam_id].append(beam_id)

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

                            # action = {'is_continue': is_continue, 'position': position_id, 'is_copy': is_copy,
                            #           'keyword': keyword, 'copy_id': copy_id, 'beam_id': beam_id}
                            p = math.log(is_continue_p) + math.log(position_p) + math.log(is_copy_p) + math.log(keyword_p)

                            beam_action_stack[beam_id].append((is_continue, position_id, is_copy, keyword, copy_id))
                            beam_p_stack[beam_id].append(p)
                            beam_id_stack[beam_id].append(beam_id)
    return beam_p_stack, beam_id_stack, beam_action_stack


def init_input_stack(args):
    # shape = 4 * batch_size * beam_size * token_length
    init_input_fn = lambda one_input: np.expand_dims(np.array(util.padded(one_input)), axis=1).tolist()
    input_stack = [init_input_fn(one_input) for one_input in args]
    return input_stack









