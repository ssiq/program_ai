import more_itertools
import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.python.util import nest


from common.tf_util import weight_multiply, get_shape, sequence_mask_with_length
from common.util import is_sequence, sequence_sum


def bi_rnn(cell_fn, inputs, length_of_input, initial_state_fw=None, initial_state_bw=None):
    """
    :param cell_fn: a function to create the rnn cell object
    :param inputs: the input of [batch, time, dim]
    :param length_of_input: the length of the inputs [batch]
    :param initial_state_fw:
    :param initial_state_bw:
    :return: outputs, output_states
    """
    cell_fw = cell_fn()
    cell_bw = cell_fn()
    print("bi_rnn_inputs:{}, bi_rnn_length:{}".format(inputs, length_of_input))
    # initial_state_fw = cell_fw.zero_state(get_shape(inputs)[0], tf.float32)
    # initial_state_bw = cell_bw.zero_state(get_shape(inputs)[0], tf.float32)
    return tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                           cell_bw=cell_bw,
                                           inputs=inputs,
                                           sequence_length=length_of_input,
                                           initial_state_fw=initial_state_fw,
                                           initial_state_bw=initial_state_bw,
                                           dtype=tf.float32,
                                           swap_memory=True)


def rnn(cell, inputs, length_of_inputs, initial_state=None):
    """
    a dynamic rnn
    :param cell: a function to create the rnn cell object
    :param inputs: the input of [batch, time, dim]
    :param length_of_inputs: the length of the inputs [batch]
    :param initial_state: initial state of rnn
    :return: outputs, output_states
    """
    rnn_cell = cell()
    return tf.nn.dynamic_rnn(cell=rnn_cell,
                             inputs=inputs,
                             sequence_length=length_of_inputs,
                             initial_state=initial_state,
                             dtype=tf.float32,
                             swap_memory=True)


def soft_attention_reduce_sum(memory, inputs, attention_size, memory_length):
    """
    :param memory:  a memory which is paid attention to.[batch, time, dim] or a tuple of this shape
    :param inputs: a tensor of shape [batch, dim] or a list of tensor with the same shape
    :param attention_size: the hidden state of attention
    :param memory_length: the sequence length of the memory
    :return: a weighted sum of the memory by time [batch, dim]
    """
    with tf.variable_scope("soft_attention_reduce_sum"):
        output = soft_attention_logit(attention_size, inputs, memory, memory_length)
        output = tf.nn.softmax(output)
        return reduce_sum_with_attention_softmax(memory, output)


def reduce_sum_with_attention_softmax(memory, attention_softmax):
    if not is_sequence(memory):
        memory = [memory]
    memory = more_itertools.collapse(memory)
    attention_softmax = tf.expand_dims(attention_softmax, axis=2)
    return [tf.reduce_sum(m * attention_softmax, axis=1) for m in memory]


def soft_attention_logit(attention_size, inputs, memory, memory_length):
    if not is_sequence(memory):
        memory = [memory]
    memory = list(more_itertools.collapse(memory))
    weighted_memory_sum = sequence_sum(weight_multiply("memory_weighted_{}".format(i), m, attention_size)
                              for i, m in enumerate(memory))
    if not is_sequence(inputs):
        inputs = [inputs]
    inputs = list(more_itertools.collapse(inputs))
    print("soft_attention_inputs:{}".format(inputs))
    weighted_inputs_sum = sequence_sum(
        weight_multiply("input_weight_{}".format(i), t, attention_size) for i, t in enumerate(inputs))
    v = tf.get_variable("v",
                        shape=(attention_size, 1),
                        dtype=tf.float32)
    output = tf.tanh(weighted_memory_sum + tf.expand_dims(weighted_inputs_sum, axis=1))
    output = tf.reshape(output, (-1, attention_size))
    output = tf.matmul(output, v)
    memory_shape = get_shape(memory[0])
    output = tf.reshape(output, (memory_shape[0], memory_shape[1]))
    output = sequence_mask_with_length(output, memory_length, score_mask_value=0.0)
    return output


def concat_bi_rnn_output(o):
    o = list(more_itertools.collapse(o[0]))
    return tf.concat(o, axis=2)

def concat_bi_rnn_final_state(o):
    o = list(more_itertools.collapse(o[1]))
    return tf.concat(o, axis=1)


def create_decoder_initialize_fn(start_label, batch_size):
    """
    :param start_label: [dim]
    :param batch_size: a scalar
    :return:
    """
    start_label = tf.expand_dims(start_label, axis=0)
    def initialize_fn():
        """Returns `(initial_finished, initial_inputs)`."""
        return tf.tile([False], [batch_size]), tf.tile(start_label, [batch_size, 1])
    return initialize_fn

class BasicDecoder(seq2seq.BasicDecoder):
    def __init__(self, cell, helper, initial_state, sample_shape, sample_dtype, output_layer=None):
        super().__init__(cell, helper, initial_state, output_layer)
        self._sample_shape = sample_shape
        self._sample_dtype = sample_dtype
        nest.assert_same_structure(self._sample_dtype, self._sample_shape, check_types=False)

    @property
    def output_size(self):
        return seq2seq.BasicDecoderOutput(
            rnn_output=self._rnn_output_size(),
            sample_id=self._sample_shape)

    @property
    def output_dtype(self):
        dtype = nest.flatten(self._initial_state)[0].dtype
        return seq2seq.BasicDecoderOutput(
            nest.map_structure(lambda _: dtype, self._rnn_output_size()), self._sample_dtype)




def create_decode(sample_helper_fn,
                  sample_sample_output_shape,
                  sample_dtype,
                  training_helper_fn,
                  training_sample_output_shape,
                  training_dtype,
                  decode_cell,
                  initial_state,
                  max_decode_iterator_num=None):

    train_decode = create_train_decode(training_helper_fn, training_sample_output_shape, training_dtype, decode_cell,
                                       initial_state, max_decode_iterator_num)
    # decode_cell.build()
    sample_helper = seq2seq.CustomHelper(*sample_helper_fn)
    sample_decoder = BasicDecoder(decode_cell, sample_helper, initial_state, sample_sample_output_shape, sample_dtype)
    print("decoder dtype:", sample_decoder.output_dtype)
    sample_decode = seq2seq.dynamic_decode(sample_decoder, maximum_iterations=max_decode_iterator_num,
                                           swap_memory=True, impute_finished=True)
    return train_decode, sample_decode


def create_train_decode(training_helper_fn, training_sample_output_shape, training_dtype, decode_cell, initial_state,
                        max_decode_iterator_num):
    training_helper = seq2seq.CustomHelper(*training_helper_fn)
    train_decoder = BasicDecoder(decode_cell, training_helper, initial_state, training_sample_output_shape,
                                 training_dtype)
    train_decode = seq2seq.dynamic_decode(train_decoder, maximum_iterations=max_decode_iterator_num,
                                          swap_memory=True, impute_finished=True)
    return train_decode


def gather_sequence(param, indices):
    """
    :param param: a batch of sequence [..., time, dim]
    :param indices: a index tensor [...] to difference batch along the time dimension
    :return:
    """

    ori_param_shape = get_shape(param)
    ori_indices_shape = get_shape(indices)
    print("ori_param_shape:{}".format(ori_param_shape))
    print("ori_indices_shape:{}".format(ori_indices_shape))
    assert len(ori_param_shape[:-2]) == len(ori_indices_shape), "The param first n-2 dim should have the same dimensions with indices"

    param = tf.reshape(param, (-1, ori_param_shape[-2], ori_param_shape[-1]))
    indices = tf.reshape(indices, (-1, ))

    batch_size = get_shape(param)[0]
    batch_index = tf.range(0, batch_size, dtype=tf.int32)
    for _ in range(len(get_shape(indices))):
        batch_index = tf.expand_dims(batch_index, axis=-1)
    indices = tf.expand_dims(indices, axis=-1)

    indices_shape = get_shape(indices)
    indices_shape[0] = 1
    batch_index = tf.tile(batch_index, indices_shape)
    # batch_index = tf.tile(batch_index, get_shape(indices))


    result = tf.gather_nd(param,
                          tf.concat(
                              (batch_index, indices),
                              axis=-1)
                          )

    result = tf.reshape(result, ori_indices_shape + [ori_param_shape[-1], ])
    return result