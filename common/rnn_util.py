import more_itertools
import tensorflow as tf

from common.tf_util import weight_multiply, get_shape, sequence_mask_with_length
from common.util import is_sequence, sequence_sum


def bi_rnn(cell, inputs, length_of_input):
    """
    :param cell: a function to create the rnn cell object
    :param inputs: the input of [batch, time, dim]
    :param length_of_input: the length of the inputs [batch]
    :return: outputs, output_states
    """
    cell_fw = cell()
    cell_bw = cell()
    print("bi_rnn_inputs:{}, bi_rnn_length:{}".format(inputs, length_of_input))
    # initial_state_fw = cell_fw.zero_state(get_shape(inputs)[0], tf.float32)
    # initial_state_bw = cell_bw.zero_state(get_shape(inputs)[0], tf.float32)
    return tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                           cell_bw=cell_bw,
                                           inputs=inputs,
                                           sequence_length=length_of_input,
                                           dtype=tf.float32)


def soft_attention_reduce_sum(memory, inputs, attention_size, memory_length):
    """
    :param memory:  a memory which is paied attention to.[batch, time, dim] or a tuple of this shape
    :param inputs: a tensor of shape [batch, dim] or a list of tensor with the same shape
    :param attention_size: the hidden state of attention
    :param memory_length: the sequence length of the memory
    :return: a weighted sum of the memory by time [batch, dim]
    """
    with tf.variable_scope("soft_attention_reduce_sum"):
        output = soft_attention_logit(attention_size, inputs, memory, memory_length)
        output = tf.nn.softmax(output)
        if not is_sequence(memory):
            memory = [memory]
        memory = more_itertools.collapse(memory)
        output = tf.expand_dims(output, axis=2)
        return [tf.reduce_sum(m * output, axis=1) for m in memory]


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