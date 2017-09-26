import tensorflow as tf

from . import util


def build_bi_rnn(state_size, embedding_matrix, action_num):
    x = tf.placeholder(dtype=tf.int32, shape=(None, None), name='x')
    embedding_matrix = tf.Variable(initial_value=embedding_matrix, name='embedding', dtype=tf.float32)
    embedding_m = tf.nn.embedding_lookup(embedding_matrix, x)
    length_of_x = util.length(tf.one_hot(x, embedding_matrix.shape[0], dtype=tf.int32))
    cell_bw = tf.nn.rnn_cell.GRUCell(state_size)
    cell_fw = tf.nn.rnn_cell.GRUCell(state_size)
    fw_init_state = cell_fw.zero_state(tf.shape(x)[0], tf.float32)
    bw_init_state = cell_bw.zero_state(tf.shape(x)[0], tf.float32)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                 cell_bw=cell_bw,
                                                 inputs=embedding_m,
                                                 sequence_length=length_of_x,
                                                 initial_state_fw=fw_init_state,
                                                 initial_state_bw=bw_init_state,
                                                 dtype=tf.float32,
                                                 swap_memory=True)
    print(outputs)
    output_fw, output_bw = outputs
    reshape_init_state = lambda t: tf.reshape(t, [util.get_shape(t)[0], 1, util.get_shape(t)[1]])
    output_fw = tf.concat([reshape_init_state(fw_init_state), output_fw], axis=1)
    output_bw = tf.reverse_sequence(output_bw, seq_lengths=length_of_x, seq_axis=1, batch_axis=0)
    output_bw = tf.concat([reshape_init_state(bw_init_state), output_bw], axis=1)
    output_bw = tf.reverse_sequence(output_bw, seq_lengths=tf.add(length_of_x, tf.constant(1, dtype=tf.int32)),
                                    seq_axis=1,
                                    batch_axis=0)

    embedding_code = tf.concat([reshape_init_state(output_fw[:, 0, :]), reshape_init_state(output_bw[:, 0, :])], axis=2)
    i0 = tf.constant(1, dtype=tf.int32, name="i")

    def cond(code, i):
        return tf.less(i, tf.shape(output_bw)[1])

    def body(code, i):
        h0_f = reshape_init_state(output_fw[:, i, :])
        h0_b = reshape_init_state(output_bw[:, i, :])
        hm1_b = reshape_init_state(output_bw[:, i - 1, :])
        code = tf.concat([code, tf.concat([h0_f, hm1_b], axis=2)], axis=1)
        code = tf.concat([code, tf.concat([h0_f, h0_b], axis=2)], axis=1)
        return code, tf.add(i, 1)

    embedding_code, _ = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=[embedding_code, i0],
        swap_memory=True,
        shape_invariants=[tf.TensorShape([None, None, util.get_shape(embedding_code)[2]]), i0.get_shape()])

    print(util.get_shape(embedding_code))

    output = tf.contrib.layers.fully_connected(embedding_code, num_outputs=action_num, activation_fn=None)

    return x, output_fw, output_bw, length_of_x, embedding_code, output
