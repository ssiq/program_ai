import tensorflow as tf
from . import tf_util


def position_embedding(output_fw, output_bw, length):
    output_fw_shape_tmp = tf_util.get_shape(output_fw)
    output_bw_shape_tmp = tf_util.get_shape(output_bw)
    token_length = tf.reshape(length, (-1,))

    output = tf.concat((output_fw, output_bw), axis=2)
    output_fw_in = tf.concat(
        (output_fw, tf.zeros((output_fw_shape_tmp[0], 1, output_fw_shape_tmp[2]), dtype=tf.float32)), axis=1)
    output_bw_in = tf.concat(
        (tf.zeros((output_bw_shape_tmp[0], 1, output_bw_shape_tmp[2]), dtype=tf.float32), output_bw), axis=1)
    output_in = tf.concat((output_bw_in, output_fw_in), axis=2)
    output = tf.concat((output, output_in[:, 1:, :]), axis=2)
    output = tf.reshape(output, (output_fw_shape_tmp[0], -1, output_fw_shape_tmp[2] * 2))
    output = tf.concat((output_in[:, :1, :], output), axis=1)

    output = tf_util.sequence_mask_with_length(output, token_length*2+1, score_mask_value=0)
    return output

def code_embedding(token_embedding: tf.Tensor, character_embedding, input_seq, identifier_token):
    mask = tf.equal(input_seq, identifier_token)
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.tile(mask, multiples=[1] * input_seq.shape.ndims + [tf_util.get_shape(token_embedding)[-1], ])
    return tf.where(mask, character_embedding, token_embedding)
