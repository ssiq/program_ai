import tensorflow as tf
from tensorflow.python.ops.losses.losses_impl import Reduction
import numpy as np
import more_itertools
from tensorflow.python.util import nest
from common.tf_util import sparse_categorical_crossentropy

from common import tf_util, code_util, rnn_util, util
from common.rnn_cell import RNNWrapper
from common.tf_util import cast_float, cast_int, all_is_nan, all_is_zero
from common.beam_search_util import beam_cal_top_k, flat_list, beam_gather, \
    select_max_output, revert_batch_beam_stack, beam_calculate, _create_next_code_without_iter_dims, cal_metrics, find_copy_input_position, init_beam_search_stack


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
    def raw_bi_rnn_op(self):
        cell_fn = lambda: self._multi_rnn_cell(self.hidden_state_size, self.rnn_layer_number)
        return rnn_util.bi_rnn(
            cell_fn,
            self.code_embedding_op,
            self.token_input_length,
        )

    @tf_util.define_scope("bi_rnn_op")
    def bi_rnn_op(self): #shape:[batch, length, dim]
        return rnn_util.concat_bi_rnn_output(self.raw_bi_rnn_op)

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
        return self.raw_bi_rnn_op[0]

    @tf_util.define_scope("self_matched_code")
    def self_matched_code_final_state_op(self):
        return rnn_util.concat_bi_rnn_final_state(self.raw_bi_rnn_op)

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
            # forward = tf.contrib.layers.fully_connected(self.self_matched_code_op[0][0], 1, None)
            forward = rnn_util.soft_attention_logit(self.hidden_state_size,
                                                    self.self_matched_code_final_state_op,
                                                    self.self_matched_code_output_op[0],
                                                    self.token_input_length)

        return tf.expand_dims(forward, axis=-1)

    @tf_util.define_scope("position", initializer=tf.contrib.layers.xavier_initializer())
    @tf_util.debug_print("position_backward:")
    def position_backward_op(self):
        with tf.variable_scope("backward"):
            # backward = tf.contrib.layers.fully_connected(self.self_matched_code_op[0][1], 1, None)
            backward = rnn_util.soft_attention_logit(self.hidden_state_size,
                                                    self.self_matched_code_final_state_op,
                                                    self.self_matched_code_output_op[1],
                                                    self.token_input_length)
        return tf.expand_dims(backward, axis=-1)

    @tf_util.define_scope("position", initializer=tf.contrib.layers.xavier_initializer())
    @tf_util.debug_print("position_logit:")
    def position_logit_op(self):
        o = code_util.position_embedding(self.position_forward_op, self.position_backward_op, self.token_input_length)
        return tf.reduce_sum(o, axis=-1)

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
            forward = tf.contrib.layers.fully_connected(self.self_matched_code_output_op[0], 1, None)
        with tf.variable_scope("backward"):
            backward = tf.contrib.layers.fully_connected(self.self_matched_code_output_op[0], 1, None)
        o = tf.nn.relu(forward + backward)
        return o

    @tf_util.define_scope("copy_word")
    def copy_word_mask_op(self):
        return self.token_input_mask

    @tf_util.define_scope("copy_word")
    @tf_util.debug_print("copy_word_softmax_op:")
    def copy_word_softmax_op(self):
        original_softmax = tf_util.variable_length_mask_softmax(self.copy_word_logit_op[:, :, 0], self.token_input_length,
                                                                tf.equal(self.token_input, self.identifier_token))
        original_softmax = tf.expand_dims(original_softmax, axis=1)
        masked_softmax = tf.matmul(original_softmax, cast_float(self.copy_word_mask_op))
        masked_softmax = tf.squeeze(masked_softmax, axis=[1])
        return masked_softmax

    @tf_util.define_scope("copy_word")
    def copy_word_token_number_op(self):
        o = tf.greater(tf.reduce_sum(self.copy_word_mask_op, axis=1), tf.constant(0, dtype=tf.int32))
        o = tf.reduce_sum(cast_int(o), axis=1)
        return o

    @tf_util.define_scope("loss")
    def loss_op(self):
        loss = tf_util.debug(tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
            cast_float(self.output_is_continue), self.is_continue_logit_op
        )), "is_continue_loss:")
        loss += tf_util.debug(tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
            cast_float(self.output_is_copy), self.is_copy_logit_op
        )), "is_copy_loss:")
        loss += tf_util.debug(tf.reduce_mean(sparse_categorical_crossentropy(
           self.output_position_label,  self.predict_op[1]
        )), "position_logit_loss:")
        copyword_loss = tf_util.debug(sparse_categorical_crossentropy(
            tf_util.debug(self.output_copy_word_id, "output_copg_word_id:",
                          lambda o:[all_is_nan(o), all_is_zero(o), tf.shape(o), o]),
            self.copy_word_softmax_op
        ), "copyword_loss:")
        keyword_loss = tf_util.debug(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.output_keyword_id, logits=self.keyword_logit_op
        ), "keyword_loss:")
        loss += tf_util.debug(tf.reduce_mean(tf.where(tf_util.cast_bool(self.output_is_copy), x=copyword_loss, y=keyword_loss)),
                              "word_loss:")
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
        argmax_keyword = cast_int(tf.argmax(self.predict_op[3], axis=1))
        argmax_copy_word = cast_int(tf.argmax(self.predict_op[4], axis=1))
        is_copy = tf.greater(self.predict_op[2], 0.5)
        return cast_int(tf.greater(self.predict_op[0], 0.5)), \
               cast_int(tf.argmax(self.predict_op[1], axis=1)), \
               cast_int(is_copy), \
               tf.where(tf_util.cast_bool(is_copy), x=argmax_copy_word, y=tf.zeros_like(argmax_keyword)), \
               tf.where(tf_util.cast_bool(is_copy), x=tf.zeros_like(argmax_copy_word), y=argmax_keyword)

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
                                      tf.placeholder(tf.float32, shape=(None, None, None), name="predict_is_continue"),
                                      is_placeholder=True)
        tf_util.add_summary_histogram("predict_position_softmax",
                                      tf.placeholder(tf.float32, shape=(None, None, None),
                                                     name="predict_position_softmax"),
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
        tf_util.add_summary_scalar("loss", tf.placeholder(tf.float32, shape=[]), is_placeholder=True)
        tf_util.add_summary_histogram("is_continue", self._model.predict_op[0], is_placeholder=False)
        tf_util.add_summary_histogram("position_softmax", self._model.predict_op[1], is_placeholder=False)
        tf_util.add_summary_histogram("is_copy", self._model.predict_op[2], is_placeholder=False)
        tf_util.add_summary_histogram("key_word", self._model.predict_op[3], is_placeholder=False)
        tf_util.add_summary_histogram("key_word", self._model.predict_op[4], is_placeholder=False)
        self._summary_fn = tf_util.placeholder_summary_merge()
        self._summary_merge_op = tf_util.merge_op()

        max_batches = 5
        to_place_placeholders = [tf.placeholder(dtype=tf.string, name="to_merege_placeholder_{}".format(i))
                                           for i in range(max_batches)]
        merged_summary = tf.summary.merge(to_place_placeholders)

        _merge_fn = tf_util.function(to_place_placeholders, merged_summary)
        self._merge_fn = lambda *args: _merge_fn(*(list(args)+[""]*(max_batches-len(args))))

        sess = tf_util.get_session()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        self._train_summary_fn = tf_util.function(
            self.input_placeholders + self.output_placeholders,
            self._summary_merge_op
        )

        self._train_fn = tf_util.function(self.input_placeholders+self.output_placeholders,
                              [self._model.loss_op, self._model.metrics_op, self._model.train_op,
                               self._model.argmax_predict_op, self._model.position_logit_op],
                                       )


        self._loss_fn = tf_util.function(self.input_placeholders+self.output_placeholders,
                              self._model.loss_op, )

        self._loss_and_train_summary_fn = tf_util.function(self.input_placeholders+self.output_placeholders,
                              [self._model.loss_op, self._summary_merge_op], )

        self._one_predict_fn = tf_util.function(self.input_placeholders,
                              self._model.predict_op, )

    def _train(self, *args):
        print("output_id:{}".format(args[-1]))
        loss, metrics, _, argmax_predict, position_logit = self._train_fn(*args)
        print("argmax_predict:")
        for n, t, p in zip(["is_continue", "position", "is_copy", "copy_word", "keyword"], args[-5:], argmax_predict):
            print("{}_t:{}".format(n, t))
            print("{}_p:{}".format(n, p))
            if n == "position":
                print("{}_logit:{}".format(n, position_logit))
        return loss, metrics, _

    @property
    def model(self):
        return self._model

    def summary(self, *args):
        # train_summary = self._train_summary_fn(*args)
        metrics_model = self.metrics_model(*args)

        flat_args, eff_ids = flat_and_get_effective_args(args)

        batch_size = len(args[0])
        total = len(flat_args[0])
        weight_array = make_weight_array(batch_size, total)

        chunked_args = more_itertools.chunked(list(zip(*flat_args)), batch_size)
        train_chunk_fn = lambda chunked: self._loss_fn(*list(zip(*chunked)))
        train_res = list(map(train_chunk_fn, chunked_args))
        # train_res = list(zip(*train_res))

        weight_fn = lambda one_res: [one_res[i] * weight_array[i] for i in range(len(one_res))]
        loss_array = weight_fn(train_res)
        loss = np.sum(loss_array) / total

        # loss, summary = self._loss_and_train_summary_fn(*args) # For each batch you use this fn to get the loss and summary
        #the loss you should calculate the weighted mean, the summary you should merge this like this
        # train_summary = self._merge_fn([summary])
        # train_summary = self._merge_fn(*train_res[1])
        tf_util.add_value_scalar("metrics", metrics_model)
        tf_util.add_value_scalar("loss", loss) # add the weighted loss here
        # return self._summary_fn(*tf_util.merge_value(), summary_input=train_summary)
        return self._summary_fn(*tf_util.merge_value())

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
        flat_args, eff_ids = flat_and_get_effective_args(args)

        batch_size = len(args[0])
        total = len(flat_args[0])
        weight_array = make_weight_array(batch_size, total)


        import more_itertools
        chunked_args = list(more_itertools.chunked(list(zip(*flat_args)), batch_size))
        train_chunk_fn = lambda chunked: self._train(*list(zip(*chunked)))
        # print("len(chunked_args):{}".format(len(chunked_args)))
        train_res = list(map(train_chunk_fn, chunked_args))
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
        import copy
        args = [copy.deepcopy([ti[0] for ti in one_input]) for one_input in args]
        batch_size = len(args[0])
        cur_beam_size = 1
        beam_size = 5
        max_decode_iterator_num = 5

        # shape = 5 * batch_size * beam_size * token_length
        input_stack = init_input_stack(args)
        beam_length_stack, beam_stack, mask_stack, select_output_stack_list = init_beam_search_stack(batch_size,
                                                                                                     cur_beam_size)

        for i in range(max_decode_iterator_num):

            input_flat = [flat_list(inp) for inp in input_stack]

            one_predict_fn = lambda chunked: self._one_predict_fn(*list(zip(*chunked)))

            chunked_input = more_itertools.chunked(list(zip(*input_flat)), batch_size)
            predict_returns = list(map(one_predict_fn, chunked_input))
            predict_returns = list(zip(*predict_returns))
            outputs = predict_returns

            output_list = [flat_list(out) for out in outputs]

            output_stack = [revert_batch_beam_stack(out_list, batch_size, cur_beam_size) for out_list in output_list]

            batch_returns = list(map(beam_calculate, list(zip(*input_stack)), list(zip(*output_stack)), beam_stack, mask_stack, beam_length_stack, list(zip(*select_output_stack_list)), [beam_size for o in range(batch_size)], [beam_calculate_output_score for o in range(batch_size)], [[] for o in range(batch_size)]))
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

            input_stack = [[list(inp) for inp in one_input]for one_input in input_stack]

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
                # copy_position_id = copy_id
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
                iden_mask = [0 for i in range(len(identifier_mask[0]))]

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


def flat_and_get_effective_args(args):
    flat_args = [flat_list(one_input) for one_input in args]
    eff_ids = get_effective_id(flat_args[0])
    flat_args = [beam_gather(one, eff_ids) for one in flat_args]
    return flat_args, eff_ids


def make_weight_array(batch_size, total):
    weight_array = int(total / batch_size) * [batch_size]
    if total % batch_size != 0:
        weight_array = weight_array + [total % batch_size]
    return weight_array


def get_effective_id(one_flat_data):
    ids = [i if np.sum(list(more_itertools.collapse(one_flat_data[i]))) != 0 else -1 for i in range(len(one_flat_data))]
    fil_fn = lambda one: one != -1
    ids = list(filter(fil_fn, ids))
    return ids


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









