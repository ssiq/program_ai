import tensorflow as tf
from . import tf_util


def position_embedding(output_fw, output_bw, length):
    output = tf.concat((output_fw, output_bw), axis=2)
    output_in = tf.concat((output_bw[:, :-1, :], output_fw[:, 1:, :]), axis=2)
    output_in_shape = tf_util.get_shape(output_in)
    output_in = tf.concat((output_in, tf.zeros((output_in_shape[0], 1, output_in_shape[2]), dtype=tf.float32)), axis=1)
    output = tf.concat((output, output_in), axis=2)
    output = tf.reshape(output, (output_in_shape[0], -1, output_in_shape[2]))
    output = tf_util.sequence_mask_with_length(output, length, score_mask_value=0.0)
    return output