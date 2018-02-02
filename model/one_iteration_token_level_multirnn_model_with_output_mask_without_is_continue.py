import more_itertools
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from common import tf_util, code_util, rnn_util, util
from common.beam_search_util import beam_cal_top_k, flat_list, beam_gather, \
    select_max_output, revert_batch_beam_stack, _create_next_code_without_iter_dims, cal_metrics, \
    find_copy_input_position, init_beam_search_stack, metrics_output_directly, beam_calculate_without_iscontinue, \
    calculate_length_by_one_input, make_mask_stack_by_length_list, cal_metrics_without_iscontinue


def _transpose_mask(identifier_mask):
    identifier_mask_dims = len(tf_util.get_shape(identifier_mask))
    return tf.transpose(identifier_mask, perm=list(range(identifier_mask_dims - 2)) +
                                   [identifier_mask_dims - 1, identifier_mask_dims - 2])

# def _sample_mask(identifier_mask):
#     identifier_mask = tf_util.cast_float(identifier_mask)
#     identifier_mask = _transpose_mask(identifier_mask)
#     print("transposed_mask:{}".format(identifier_mask))
#     identifier_mask_shape = tf_util.get_shape(identifier_mask)
#     identifier_mask = tf.reshape(identifier_mask, (-1, identifier_mask_shape[-1]))
#     identifier_mask_sum = tf.reduce_sum(identifier_mask, axis=-1, keep_dims=True)
#     identifier_mask_mask = tf.greater(identifier_mask_sum, tf.constant(0.0, dtype=tf.float32))
#     identifier_mask_sum = tf.where(tf.equal(identifier_mask_sum, tf.constant(0.0, dtype=tf.float32)),
#                                    x=tf.ones_like(identifier_mask_sum),
#                                    y=identifier_mask_sum)
#     identifier_mask = identifier_mask / identifier_mask_sum
#     identifier_mask = tf.clip_by_value(identifier_mask, clip_value_min=1e-7, clip_value_max=1.0-1e-7)
#     identifier_mask = tf.log(identifier_mask)
#     sampled_mask = tf.multinomial(identifier_mask, 1)
#     print("sample_mask:{}".format(sampled_mask))
#     # sampled_mask = tf.Print(sampled_mask, [sampled_mask], "samples_mask")
#     diagnal = tf.diag(tf.ones((identifier_mask_shape[-1], ), dtype=tf.float32))
#     print("diagnal:{}".format(diagnal))
#     sampled_mask = tf.nn.embedding_lookup(diagnal, sampled_mask)
#     sampled_mask = tf.squeeze(sampled_mask, axis=-2)
#     print("looked_sample_mask:{}".format(sampled_mask))
#     sampled_mask = sampled_mask * tf_util.cast_float(identifier_mask_mask)
#     sampled_mask = tf.reshape(sampled_mask, identifier_mask_shape)
#     sampled_mask = _transpose_mask(sampled_mask)
#     return sampled_mask


def _sample_mask(identifier_mask):
    whehter_sample = tf.constant(False, dtype=tf.bool)
    identifier_mask = tf_util.cast_float(identifier_mask)
    ori_identifier_mask = identifier_mask
    identifier_mask = _transpose_mask(identifier_mask)
    print("transposed_mask:{}".format(identifier_mask))
    identifier_mask_shape = tf_util.get_shape(identifier_mask)
    identifier_mask = tf.reshape(identifier_mask, (-1, identifier_mask_shape[-1]))
    identifier_mask_sum = tf.reduce_sum(identifier_mask, axis=-1, keep_dims=True)
    identifier_mask_mask = tf.greater(identifier_mask_sum, tf.constant(0.0, dtype=tf.float32))
    identifier_mask_sum = tf.where(tf.equal(identifier_mask_sum, tf.constant(0.0, dtype=tf.float32)),
                                   x=tf.ones_like(identifier_mask_sum),
                                   y=identifier_mask_sum)
    identifier_mask = identifier_mask / identifier_mask_sum
    identifier_mask = tf.clip_by_value(identifier_mask, clip_value_min=1e-7, clip_value_max=1.0-1e-7)
    identifier_mask = tf.log(identifier_mask)
    sampled_mask = tf.multinomial(identifier_mask, 1)
    print("sample_mask:{}".format(sampled_mask))
    # sampled_mask = tf.Print(sampled_mask, [sampled_mask], "samples_mask")
    diagnal = tf.diag(tf.ones((identifier_mask_shape[-1], ), dtype=tf.float32))
    print("diagnal:{}".format(diagnal))
    sampled_mask = tf.nn.embedding_lookup(diagnal, sampled_mask)
    sampled_mask = tf.squeeze(sampled_mask, axis=-2)
    print("looked_sample_mask:{}".format(sampled_mask))
    sampled_mask = sampled_mask * tf_util.cast_float(identifier_mask_mask)
    sampled_mask = tf.reshape(sampled_mask, identifier_mask_shape)
    sampled_mask = _transpose_mask(sampled_mask)
    sampled_mask = tf.cond(whehter_sample,
                           true_fn=lambda: sampled_mask,
                           false_fn=lambda: ori_identifier_mask)
    return sampled_mask


class TokenLevelMultiRnnModelGraph(tf_util.BaseModel):
    def __init__(self,
                 character_embedding_layer_fn,
                 placeholder_token,
                 id_to_word_fn,
                 word_embedding_layer_fn,
                 parse_token_fn,
                 hidden_size,
                 learning_rate,
                 keyword_number,
                 identifier_token,
                 rnn_layer_number,
                 end_token_id,
                 placeholders,
                 decay_step=500,
                 decay_rate=1.0):
        super().__init__(learning_rate, decay_step, decay_rate)
        self.character_embedding_layer_fn = character_embedding_layer_fn
        self.placeholder_token = placeholder_token
        self.id_to_word = id_to_word_fn
        self.word_embedding_layer_fn = word_embedding_layer_fn
        self.parse_token = parse_token_fn
        self.hidden_state_size = hidden_size
        self.keyword_number = keyword_number
        self.identifier_token = identifier_token
        self.rnn_layer_number = rnn_layer_number
        self.end_token_id = end_token_id
        self.token_input, self.token_input_length, self.character_input, self.character_input_length, self.token_identifier_mask, \
        self.output_position_label, self.output_is_copy, self.output_keyword_id, self.output_copy_word_id = placeholders

    def _rnn_cell(self, hidden_size):
        return tf.nn.rnn_cell.GRUCell(hidden_size)

    def _multi_rnn_cell(self, hidden_size, layer_number):
        return tf.nn.rnn_cell.MultiRNNCell([self._rnn_cell(hidden_size) for _ in range(layer_number)])

    @tf_util.define_scope("batch_size")
    def batch_size_op(self):
        return tf_util.get_shape(self.token_input)[0]

    @tf_util.define_scope("word_embedding_op")
    def word_embedding_op(self):
        input_embedding_op = self.word_embedding_layer_fn(self.token_input)
        return input_embedding_op

    @tf_util.define_scope("character_embedding_op")
    def character_embedding_op(self):
        input_embedding_op = self.character_embedding_layer_fn(self.character_input, self.character_input_length)
        return input_embedding_op

    @tf_util.define_scope("code_embedding_op")
    def code_embedding_op(self):
        token_embedding = self.word_embedding_op
        character_embedding = self.character_embedding_op
        return code_util.code_embedding(token_embedding, character_embedding, self.token_input, self.identifier_token)

    @tf_util.define_scope("bi_gru_encode_op")
    def bi_gru_encode_op(self):
        code_embedding = self.code_embedding_op
        code_input_length = self.token_input_length
        (encode_output_fw, encode_output_bw), _ = rnn_util.bi_rnn(lambda: self._multi_rnn_cell(self.hidden_state_size, self.rnn_layer_number),
                                        code_embedding, code_input_length)

        return encode_output_fw, encode_output_bw

    @tf_util.define_scope("bi_gru_encode_op")
    def concat_gru_output_op(self):
        return tf.concat(self.bi_gru_encode_op, axis=-1)

    @tf_util.define_scope("position_length")
    def position_length_op(self):
        return self.token_input_length * 2 + 1

    @tf_util.define_scope("position_embedding")
    def position_embedding_op(self):
        """
        :return: (position_embedding tensor)
        """
        return code_util.position_embedding(self.bi_gru_encode_op[0],
                                            self.bi_gru_encode_op[1],
                                            self.token_input_length)

    @tf_util.define_scope("look_variable")
    def look_variable_op(self):
        return tf.tile(
            tf.Variable(tf.zeros((1, self.hidden_state_size), dtype=tf.float32),
                        dtype=tf.float32,
                        name="look_variable"),
            [self.batch_size_op, 1]
        )

    def _cal_next(self, _memory, _memory_length, _position_embedding, _position_length, inputs):
        with tf.variable_scope("input_attention"):
            atten = rnn_util.soft_attention_reduce_sum(_memory,
                                                       inputs,
                                                       self.hidden_state_size,
                                                       _memory_length)
        atten = nest.flatten(atten)
        inputs = nest.flatten(inputs)
        print("atten:{}, input:{}".format(atten, inputs))
        inputs = tf.concat(inputs + atten, axis=1)
        gate_weight = tf.get_variable("gate_weight",
                                      shape=(tf_util.get_shape(inputs)[1], tf_util.get_shape(inputs)[1]),
                                      dtype=tf.float32)
        cell_inputs = inputs * tf.sigmoid(tf.matmul(inputs, gate_weight))
        outputs = tf.nn.relu(tf_util.dense(cell_inputs, self.hidden_state_size, "input_to_output_weight"))
        outputs = tf_util.highway(outputs, self.hidden_state_size, tf.nn.relu)
        # a scalar indicating whether the code has been corrupted
        # is_continue_logit = tf_util.weight_multiply("continue_weight", outputs, 1)[:, 0]
        # position_logit
        with tf.variable_scope("poistion_logit"):
            position_logit = rnn_util.soft_attention_logit(self.hidden_state_size,
                                                           outputs,
                                                           _position_embedding,
                                                           _position_length)
            print("position_logit:{}".format(position_logit))
        position_softmax = tf_util.variable_length_softmax(position_logit, _position_length)
        replace_input = rnn_util.reduce_sum_with_attention_softmax(_position_embedding,
                                                                   position_softmax)[0]
        def _multi_layer(x, output_size, name, layer_num):
            for i in range(layer_num-1):
                x = tf.nn.relu(tf_util.dense(x, self.hidden_state_size, "{}_{}".format(name, i)))
            return tf_util.dense(x, output_size, name)
        replace_ouput = _multi_layer(replace_input, self.hidden_state_size, "replace_output_weight", 3)
        # replace_ouput = tf_util.weight_multiply("replace_output_weight", replace_input, self._attention_size)
        # a scalar indicating whether copies from the code
        (m_forward, m_backward), _ = rnn_util.bi_rnn(lambda: self._multi_rnn_cell(self.hidden_state_size, self.rnn_layer_number)
                                                     , _memory, _memory_length,)
        replace_ouput = tf.concat(rnn_util.soft_attention_reduce_sum([m_forward, m_backward], replace_ouput, self.hidden_state_size, _memory_length),
                                  axis=-1)
        print("replace_out:{}".format(replace_ouput))
        is_copy_logit = _multi_layer(replace_ouput, 1, "copy_weight", 3)[:, 0]
        # is_copy_logit = tf_util.weight_multiply("copy_weight", replace_ouput, 1)[:, 0]
        # key_word_logit
        key_word_logit = _multi_layer(replace_ouput, self.keyword_number, "key_word_logit_weight", 3)
        # key_word_logit = tf_util.weight_multiply("key_word_logit_weight", replace_ouput, self._keyword_size)
        # copy_word_logit
        with tf.variable_scope("copy_word_logit"):
            copy_word_state = _multi_layer(replace_ouput, self.hidden_state_size, "copy_word_weight", 3)
            copy_word_logit = rnn_util.soft_attention_logit(self.hidden_state_size, copy_word_state, [m_forward, m_backward],
                                                            _memory_length)
            print("copy_word_logit:{}".format(copy_word_logit))
        return copy_word_logit, is_copy_logit, key_word_logit, position_logit


    @tf_util.define_scope("logit_op")
    def logit_op(self):
        return self._cal_next(
            self.concat_gru_output_op,
            self.token_input_length,
            self.position_embedding_op,
            self.position_length_op,
            self.look_variable_op
        )

    @tf_util.define_scope("softmax_op")
    def softmax_op(self):
        copy_word_logit, is_copy_logit, key_word_logit, position_logit = self.logit_op
        position_softmax = tf_util.variable_length_softmax(position_logit, self.position_length_op)
        is_copy = tf.nn.sigmoid(is_copy_logit)
        key_word_softmax = tf.nn.softmax(key_word_logit)
        copy_word_softmax = tf_util.variable_length_softmax(copy_word_logit, self.token_input_length)
        copy_word_softmax = self._multiply_with_mask(copy_word_softmax, _sample_mask(self.token_identifier_mask))
        return position_softmax, is_copy, key_word_softmax, copy_word_softmax

    def _multiply_with_mask(self, softmax, mask):
        mask = tf_util.cast_float(mask)
        copy_word_softmax = tf.expand_dims(softmax, axis=-2)
        copy_word_softmax = tf.matmul(copy_word_softmax, mask)
        copy_word_softmax = tf.squeeze(copy_word_softmax, axis=-2)
        return copy_word_softmax

    @tf_util.define_scope("loss_op")
    def loss_op(self):
        copy_word_logit, is_copy_logit, key_word_logit, position_logit = self.logit_op
        loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=is_copy_logit,
                                                                labels=tf.cast(
                                                                    self.output_is_copy,
                                                                    tf.float32)))
        sparse_softmax_loss = lambda x, y: tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x, logits=y)
        sparse_categorical_loss = lambda x, y: tf_util.sparse_categorical_crossentropy(target=x, output=y)
        position_loss = tf.reduce_sum(
            sparse_categorical_loss(self.output_position_label, self.softmax_op[0]))
        keyword_loss = sparse_softmax_loss(self.output_keyword_id, key_word_logit)
        copy_word_loss = sparse_categorical_loss(self.output_copy_word_id, self.softmax_op[3])
        word_loss = tf.reduce_sum(tf.where(tf_util.cast_bool(self.output_is_copy), x=copy_word_loss, y=keyword_loss))
        loss += position_loss
        loss += word_loss
        return loss

    @tf_util.define_scope("predict")
    def predict_op(self):
        copy_word_logit, is_copy_logit, key_word_logit, position_logit = self.logit_op
        output = (tf_util.variable_length_softmax(position_logit, self.position_length_op),
                  tf.nn.sigmoid(is_copy_logit),
                  tf.nn.softmax(key_word_logit),
                  self._multiply_with_mask(tf_util.variable_length_softmax(copy_word_logit, self.token_input_length),
                                           self.token_identifier_mask))
        return output


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
        # self.output_is_continue = tf.placeholder(dtype=tf.int32, shape=(None, ), name="output_length")
        self.output_position_label = tf.placeholder(dtype=tf.int32, shape=(None, ), name="output_position")
        self.output_is_copy = tf.placeholder(dtype=tf.int32, shape=(None, ),
                                             name="output_is_copy")  # 1 means using copy
        self.output_keyword_id = tf.placeholder(dtype=tf.int32, shape=(None, ), name="output_keyword_id")
        self.output_copy_word_id = tf.placeholder(dtype=tf.int32, shape=(None, ), name="output_copy_word_id")
        self.output_placeholders = [self.output_position_label, self.output_is_copy,
        self.output_keyword_id, self.output_copy_word_id]

        self._model = TokenLevelMultiRnnModelGraph(
            self.character_embedding_layer_fn,
            self.placeholder_token,
            self.id_to_word,
            self.word_embedding_layer_fn,
            self.parse_token,
            self.hidden_state_size,
            self.learning_rate,
            self.keyword_number,
            self.identifier_token,
            self.rnn_layer_number,
            self.end_token_id,
            self.input_placeholders + self.output_placeholders,
            self.decay_steps,
            self.decay_rate
        )

        tf_util.init_all_op(self._model)

        metrics_input_placeholder = tf.placeholder(tf.float32, shape=[], name="metrics")
        tf_util.add_summary_scalar("metrics", metrics_input_placeholder, is_placeholder=True)
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
        tf_util.add_summary_histogram("position_softmax", self._model.predict_op[0], is_placeholder=False)
        tf_util.add_summary_histogram("is_copy", self._model.predict_op[1], is_placeholder=False)
        tf_util.add_summary_histogram("key_word", self._model.predict_op[2], is_placeholder=False)
        tf_util.add_summary_histogram("key_word", self._model.predict_op[3], is_placeholder=False)
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
                              [self._model.loss_op, self._model.train_op, ], )

        self._loss_fn = tf_util.function(self.input_placeholders+self.output_placeholders, self._model.loss_op, )

        self._loss_and_train_summary_fn = tf_util.function(self.input_placeholders+self.output_placeholders,
                              [self._model.loss_op, self._summary_merge_op], )

        self._one_predict_fn = tf_util.function(self.input_placeholders, self._model.predict_op, )

    def _train(self, *args):
        # print("output_id:{}".format(args[-1]))
        loss, _ = self._train_fn(*args)
        # print("argmax_predict:")
        # for n, t, p in zip(["is_continue", "position", "is_copy", "copy_word", "keyword"], args[-5:], argmax_predict):
        #     print("{}_t:{}".format(n, t))
        #     print("{}_p:{}".format(n, p))
        #     if n == "position":
        #         print("{}_logit:{}".format(n, position_logit))
        return loss, 0, _

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

    def quick_metrics_model(self, *args):
        input_data = args[0:5]
        output_data = args[5:9]

        predict_data = self._one_predict_fn(*input_data)
        predict_data = list(predict_data)
        predict_data[0] = np.argmax(predict_data[0], axis=1)
        predict_data[1] = np.greater(predict_data[1], 0.5)
        predict_data[2] = np.argmax(predict_data[2], axis=1)
        predict_data[2] = np.where(predict_data[1], np.zeros_like(predict_data[1]), predict_data[2])
        predict_data[3] = np.argmax(predict_data[3], axis=1)
        predict_data[3] = np.where(predict_data[1], predict_data[3], np.zeros_like(predict_data[1]))
        res_mask = metrics_output_directly(output_data, predict_data)
        metrics_value = res_mask.mean()

        name_list = ["position", "is_copy", "keyword", "copy_word"]
        for n, p, o in zip(name_list, predict_data, output_data):
            print("{}:predict:{}, output:{}".format(n, p, o))
        # print('metrics_value: ', metrics_value)
        return metrics_value

    def metrics_model(self, *args):
        # print("metrics input")
        # for t in args:
        #     print(np.array(t).shape)
        max_decode_iterator_num = 5
        input_data = args[0:5]
        output_data = args[5:9]
        predict_data = self.predict_model(*input_data, )
        name_list = ["position", "is_copy", "keyword", "copy_word"]
        for n, p, o in zip(name_list, predict_data, output_data):
            print("{}:predict:{}, output:{}".format(n, p, o))
        metrics_value = cal_metrics_without_iscontinue(max_decode_iterator_num, output_data, predict_data)
        # print('metrics_value: ', metrics_value)
        return metrics_value

    def train_model(self, *args):
        # for t in args:
        #     print(np.array(t).shape)
            # print(t)
        # print(self.input_placeholders+self.output_placeholders)
        # flat_args, eff_ids = flat_and_get_effective_args(args)
        #
        # batch_size = len(args[0])
        # total = len(flat_args[0])
        # weight_array = make_weight_array(batch_size, total)
        #
        #
        # import more_itertools
        # chunked_args = list(more_itertools.chunked(list(zip(*flat_args)), batch_size))
        # train_chunk_fn = lambda chunked: self._train(*list(zip(*chunked)))
        # # print("len(chunked_args):{}".format(len(chunked_args)))
        # train_res = list(map(train_chunk_fn, chunked_args))
        # train_res = list(zip(*train_res))
        #
        # weight_fn = lambda one_res: [one_res[i] * weight_array[i] for i in range(len(one_res))]
        # loss_array = weight_fn(train_res[0])
        # accracy_array = weight_fn(train_res[1])
        # # tt_array = weight_fn(train_res[2])
        # loss = np.sum(loss_array) / total
        # accracy = np.sum(accracy_array) / total
        # # tt = np.sum(tt_array) / total

        loss, accracy, tt = self._train(*args)
        return loss, accracy, None

    def predict_model(self, *args,):
        length_list = calculate_length_by_one_input(args[0])
        import copy
        args = [copy.deepcopy([ti[0] for ti in one_input]) for one_input in args]
        batch_size = len(args[0])
        cur_beam_size = 1
        beam_size = 5
        max_decode_iterator_num = 5

        # shape = 5 * batch_size * beam_size * token_length
        input_stack = init_input_stack(args)
        beam_length_stack, beam_stack, mask_stack, select_output_stack_list = init_beam_search_stack(batch_size,
                                                                                                     cur_beam_size, output_num=4)

        mask_stack = make_mask_stack_by_length_list(cur_beam_size, 0, length_list)

        for i in range(max_decode_iterator_num):

            input_flat = [flat_list(inp) for inp in input_stack]

            one_predict_fn = lambda chunked: self._one_predict_fn(*list(zip(*chunked)))

            chunked_input = more_itertools.chunked(list(zip(*input_flat)), batch_size)
            predict_returns = list(map(one_predict_fn, chunked_input))
            predict_returns = list(zip(*predict_returns))
            outputs = predict_returns

            output_list = [flat_list(out) for out in outputs]

            output_stack = [revert_batch_beam_stack(out_list, batch_size, cur_beam_size) for out_list in output_list]

            batch_returns = list(map(beam_calculate_without_iscontinue, list(zip(*input_stack)), list(zip(*output_stack)), beam_stack, mask_stack, beam_length_stack, list(zip(*select_output_stack_list)), [beam_size for o in range(batch_size)], [beam_calculate_output_score_without_iscontinue for o in range(batch_size)], [[] for o in range(batch_size)]))
            def create_next(ret):
                ret = list(ret)
                ret[0] = _create_next_code_without_iter_dims(ret[1], ret[0], create_one_fn=self._create_one_next_code_without_continue)
                return ret
            batch_returns = [create_next(ret) for ret in batch_returns]
            input_stack, output_stack, select_output_stack_list, mask_stack, beam_stack, beam_length_stack, _ = list(zip(*batch_returns))
            input_stack = list(zip(*input_stack))
            output_stack = list(zip(*output_stack))
            select_output_stack_list = list(zip(*select_output_stack_list))

            cur_beam_size = beam_size
            mask_stack = make_mask_stack_by_length_list(cur_beam_size, i + 1, length_list)

            if np.sum(mask_stack) == 0:
                break

            input_stack = [[list(inp) for inp in one_input]for one_input in input_stack]

            input_stack = [list(util.padded(list(inp))) for inp in input_stack]
            mask_input_with_end_fn = lambda token_input: list([util.mask_input_with_end(batch_mask, batch_inp, n_dim=1).tolist() for batch_mask, batch_inp in zip(mask_stack, token_input)])
            input_stack = list(map(mask_input_with_end_fn, input_stack))


        summary = copy.deepcopy(select_output_stack_list)
        tf_util.add_value_histogram("predict_position_softmax", util.padded(summary[0]))
        tf_util.add_value_histogram("predict_is_copy", util.padded(summary[1]))
        tf_util.add_value_histogram("predict_key_word", util.padded(summary[2]))
        tf_util.add_value_histogram("predict_copy_word", util.padded(summary[3]))

        final_output = select_max_output(beam_stack, select_output_stack_list)
        return final_output


    def record_predict_model(self, *args,):
        import copy
        args = [copy.deepcopy([ti[0] for ti in one_input]) for one_input in args]
        batch_size = len(args[0])
        cur_beam_size = 1
        beam_size = 5
        max_decode_iterator_num = 5
        output_num = 4

        length_list = [max_decode_iterator_num for i in range(batch_size)]

        # shape = 5 * batch_size * beam_size * token_length
        input_stack = init_input_stack(args)
        beam_length_stack, beam_stack, mask_stack, select_output_stack_list = init_beam_search_stack(batch_size,
                                                                                                     cur_beam_size, output_num=4)

        record_output_list = [[[] for j in range(batch_size)] for i in range(output_num)]
        mask_stack = make_mask_stack_by_length_list(cur_beam_size, 0, length_list)

        for i in range(max_decode_iterator_num):

            input_flat = [flat_list(inp) for inp in input_stack]

            one_predict_fn = lambda chunked: self._one_predict_fn(*list(zip(*chunked)))

            chunked_input = more_itertools.chunked(list(zip(*input_flat)), batch_size)
            predict_returns = list(map(one_predict_fn, chunked_input))
            predict_returns = list(zip(*predict_returns))
            outputs = predict_returns

            output_list = [flat_list(out) for out in outputs]

            output_stack = [revert_batch_beam_stack(out_list, batch_size, cur_beam_size) for out_list in output_list]

            batch_returns = list(map(beam_calculate_without_iscontinue, list(zip(*input_stack)), list(zip(*output_stack)), beam_stack, mask_stack, beam_length_stack, list(zip(*select_output_stack_list)), [beam_size for o in range(batch_size)], [beam_calculate_output_score_without_iscontinue for o in range(batch_size)], [[] for o in range(batch_size)]))
            def create_next(ret):
                ret = list(ret)
                ret[0] = _create_next_code_without_iter_dims(ret[1], ret[0], create_one_fn=self._create_one_next_code_without_continue)
                return ret
            batch_returns = [create_next(ret) for ret in batch_returns]
            input_stack, output_stack, select_output_stack_list, mask_stack, beam_stack, beam_length_stack, _ = list(zip(*batch_returns))
            input_stack = list(zip(*input_stack))
            output_stack = list(zip(*output_stack))
            select_output_stack_list = list(zip(*select_output_stack_list))
            record_output_list = [[record_one_batch+copy.deepcopy(select_one_batch)  for record_one_batch, select_one_batch in zip(record_one_output, select_one_output)] for record_one_output, select_one_output in zip(record_output_list, select_output_stack_list)]

            cur_beam_size = beam_size
            mask_stack = make_mask_stack_by_length_list(cur_beam_size, i + 1, length_list)

            if np.sum(mask_stack) == 0:
                break

            input_stack = [[list(inp) for inp in one_input]for one_input in input_stack]

            input_stack = [list(util.padded(list(inp))) for inp in input_stack]
            mask_input_with_end_fn = lambda token_input: list([util.mask_input_with_end(batch_mask, batch_inp, n_dim=1).tolist() for batch_mask, batch_inp in zip(mask_stack, token_input)])
            input_stack = list(map(mask_input_with_end_fn, input_stack))

        summary = copy.deepcopy(select_output_stack_list)
        tf_util.add_value_histogram("predict_position_softmax", util.padded(summary[0]))
        tf_util.add_value_histogram("predict_is_copy", util.padded(summary[1]))
        tf_util.add_value_histogram("predict_key_word", util.padded(summary[2]))
        tf_util.add_value_histogram("predict_copy_word", util.padded(summary[3]))

        final_output = select_max_output(beam_stack, select_output_stack_list)
        return record_output_list


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

    def _create_one_next_code_without_continue(self, action, token_input, token_input_length, character_input, character_input_length, identifier_mask):
        position, is_copy, keyword_id, copy_id = action
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
                            is_continue_p = is_continue_p if is_continue_p > 0 else 0.00000001
                            position_p = position_p if position_p > 0 else 0.00000001
                            is_copy_p = is_copy_p if is_copy_p > 0 else 0.00000001
                            copy_id_p = copy_id_p if copy_id_p > 0 else 0.00000001

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
                            is_continue_p = is_continue_p if is_continue_p > 0 else 0.00000001
                            position_p = position_p if position_p > 0 else 0.00000001
                            is_copy_p = is_copy_p if is_copy_p > 0 else 0.00000001
                            keyword_p = keyword_p if keyword_p > 0 else 0.00000001

                            # action = {'is_continue': is_continue, 'position': position_id, 'is_copy': is_copy,
                            #           'keyword': keyword, 'copy_id': copy_id, 'beam_id': beam_id}
                            p = math.log(is_continue_p) + math.log(position_p) + math.log(is_copy_p) + math.log(keyword_p)

                            beam_action_stack[beam_id].append((is_continue, position_id, is_copy, keyword, copy_id))
                            beam_p_stack[beam_id].append(p)
                            beam_id_stack[beam_id].append(beam_id)
    return beam_p_stack, beam_id_stack, beam_action_stack


def beam_calculate_output_score_without_iscontinue(output_beam_list, beam_size):
    import math

    output_positions, output_is_copys, output_keyword_ids, output_copy_ids = output_beam_list
    cur_beam_size = len(output_positions)
    # print('cur_beam_size:',cur_beam_size)
    beam_action_stack = [[] for i in range(beam_size)]
    beam_p_stack = [[] for i in range(beam_size)]
    beam_id_stack = [[] for i in range(beam_size)]

    top_position_beam_id_list = [beam_cal_top_k(beam, beam_size) for beam in output_positions]
    top_keyword_beam_id_list = [beam_cal_top_k(beam, beam_size) for beam in output_keyword_ids]
    top_copy_beam_id_list = [beam_cal_top_k(beam, beam_size) for beam in output_copy_ids]
    sigmoid_to_p_distribute = lambda x: [1 - x, x]
    output_is_copys = [sigmoid_to_p_distribute(beam) for beam in output_is_copys]
    top_is_copy_beam_id_list = [beam_cal_top_k(beam, beam_size) for beam in output_is_copys]
    for beam_id in range(cur_beam_size):

        for position_id in top_position_beam_id_list[beam_id]:
            for is_copy in top_is_copy_beam_id_list[beam_id]:
                if is_copy == 1:
                    for copy_id in top_copy_beam_id_list[beam_id]:
                        keyword = 0
                        position_p = output_positions[beam_id][position_id]
                        is_copy_p = output_is_copys[beam_id][is_copy]
                        copy_id_p = output_copy_ids[beam_id][copy_id]
                        position_p = position_p if position_p > 0 else 0.00000001
                        is_copy_p = is_copy_p if is_copy_p > 0 else 0.00000001
                        copy_id_p = copy_id_p if copy_id_p > 0 else 0.00000001

                        # action = {'is_continue': is_continue, 'position': position_id, 'is_copy': is_copy,
                        #           'keyword': keyword, 'copy_id': copy_id, 'beam_id': beam_id}
                        # print(is_continue_p, position_p, is_copy_p, copy_id_p)
                        p = math.log(position_p) + math.log(is_copy_p) + math.log(copy_id_p)

                        beam_action_stack[beam_id].append((position_id, is_copy, keyword, copy_id))
                        beam_p_stack[beam_id].append(p)
                        beam_id_stack[beam_id].append(beam_id)

                else:
                    for keyword in top_keyword_beam_id_list[beam_id]:
                        copy_id = 0
                        position_p = output_positions[beam_id][position_id]
                        is_copy_p = output_is_copys[beam_id][is_copy]
                        keyword_p = output_keyword_ids[beam_id][keyword]
                        position_p = position_p if position_p > 0 else 0.00000001
                        is_copy_p = is_copy_p if is_copy_p > 0 else 0.00000001
                        keyword_p = keyword_p if keyword_p > 0 else 0.00000001

                        # action = {'is_continue': is_continue, 'position': position_id, 'is_copy': is_copy,
                        #           'keyword': keyword, 'copy_id': copy_id, 'beam_id': beam_id}
                        p = math.log(position_p) + math.log(is_copy_p) + math.log(keyword_p)

                        beam_action_stack[beam_id].append((position_id, is_copy, keyword, copy_id))
                        beam_p_stack[beam_id].append(p)
                        beam_id_stack[beam_id].append(beam_id)
    return beam_p_stack, beam_id_stack, beam_action_stack


def init_input_stack(args):
    # shape = 4 * batch_size * beam_size * token_length
    init_input_fn = lambda one_input: np.expand_dims(np.array(util.padded(one_input)), axis=1).tolist()
    input_stack = [init_input_fn(one_input) for one_input in args]
    return input_stack
