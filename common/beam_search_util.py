import numpy
import numpy as np
import more_itertools


def length_penalty(sequence_lengths, penalty_factor=0.6):
    """Calculates the length penalty according to
    https://arxiv.org/abs/1609.08144

     Args:
      sequence_lengths: The sequence length of all hypotheses, a tensor
        of shape [beam_size, vocab_size].
      penalty_factor: A scalar that weights the length penalty.

    Returns:
      The length penalty factor, a tensor fo shape [beam_size].
     """
    return (float(5. + sequence_lengths) ** penalty_factor) / float((5. + 1.) ** penalty_factor)


def beam_calculate_length_penalty(beam_lengths, output_p_beams, penalty_factor=0.6):
    batch_score = [[float(p)/length_penalty(leng, penalty_factor) for p in p_list] for leng, p_list in zip(beam_lengths, output_p_beams)]
    return batch_score


def beam_calculate_score(p_stack, output_p):
    batch_score = [[(p + p_old) for p in p_list] for p_old, p_list in zip(p_stack, output_p)]
    return batch_score


def beam_cal_top_k(one_batch, k):
    one_batch = np.array(one_batch)
    return one_batch.argsort()[-k:][::-1]


def flat_list_using_np(one_input, n_dim=0, flat_len=2):
    if flat_len < 2:
        flat_len = 2
    one_input = np.array(one_input)
    one_input_shape = list(one_input.shape)
    one_input = np.reshape(one_input, one_input_shape[0:n_dim] + [-1] + one_input_shape[n_dim+flat_len:]).tolist()
    return one_input


def flat_list(one_input, levels = 1):
    flat_input = list(more_itertools.collapse(one_input, levels=levels))
    return flat_input


def beam_flat(one_input):
    return flat_list(one_input)


def select_max_output(beam_score_stack, select_output_stack):
    beam_max_indices = np.argmax(np.array(beam_score_stack), axis=1).tolist()
    cal_output_fn = lambda one_output_stack: [select_output[ind] for ind, select_output in zip(beam_max_indices, one_output_stack)]
    res_output = [cal_output_fn(one_output) for one_output in select_output_stack]
    return res_output


def beam_gather(one_output, indices:list, deepcopy=False):
    import copy
    if deepcopy:
        one_output = [copy.deepcopy(one_output[ind]) for ind in indices]
    else:
        one_output = [one_output[ind] for ind in indices]
    return one_output


def revert_batch_beam_stack(one_output, batch_size, beam_size):
    '''
    revert [batch*beam, :] to [batch, beam, :]
    :param one_output:
    :param batch_size:
    :param beam_size:
    :return:
    '''
    one_output = np.array(one_output)
    one_output_shape = list(one_output.shape)
    one_output = np.reshape(one_output, [batch_size, beam_size] + one_output_shape[1:]).tolist()
    return one_output


# def beam_get_key_from_action(beam_actions, key_name):
#     return [act[key_name] for act in beam_actions]


def beam_calculate(inputs, outputs_logit, beam_score, end_beam, length_beam, select_beam, beam_size, beam_calculate_output_score_fn, beam_gather_args):
    '''

    :param inputs:
    :param outputs_logit:
    :param beam_score:
    :param next_states:
    :param position_embedding:
    :param code_embedding:
    :param end_beam:
    :param length_beam:
    :param select_beam:
    :param beam_size:
    :param beam_calculate_output_score_fn: return p_beam, action_beam.
    :return:
    '''
    # is_continues_beam, positions_beam, is_copys_beam, keyword_ids_beam, copy_ids_beam = outputs

    # if np.sum(end_beam) == 0:
    #     outputs = [[0]*len(out) for out in outputs_logit]
    #     return inputs, outputs, select_beam, end_beam, beam_score, next_states, position_embedding, code_embedding, length_beam

    length_beam = (np.array(length_beam) + np.array(end_beam)).tolist()
    cur_beam_size = len(outputs_logit[1])

    p_beam, id_beam, action_beam = beam_calculate_output_score_fn(outputs_logit, beam_size)
    p_beam = [(p_b if end_b else [0]) for p_b, end_b in zip(p_beam, end_beam)]
    p_score = beam_calculate_score(beam_score, p_beam)
    p_score_with_penalty = beam_calculate_length_penalty(length_beam, p_score)
    p_score = beam_flat(p_score)
    p_score_with_penalty = beam_flat(p_score_with_penalty)

    action_beam = beam_flat(action_beam)
    id_beam = beam_flat(id_beam)
    top_indices = beam_cal_top_k(p_score_with_penalty, beam_size)
    beam_score = beam_gather(p_score, top_indices)
    action_beam = beam_gather(action_beam, top_indices)
    beam_indices = beam_gather(id_beam, top_indices)

    # outputs_logit = [self.beam_gather(out, beam_indices) for out in outputs_logit]
    end_beam = beam_gather(end_beam, beam_indices)
    # outputs = beam_get_output_from_action_beams(action_beam)
    outputs = list(zip(*action_beam))
    outputs = [np.where(end_beam, out, np.zeros_like(out)).tolist() for out in outputs]
    # print('outputs:', outputs)
    select_beam = [beam_gather(sel_beam, beam_indices, deepcopy=True) for sel_beam in select_beam]
    is_continues_beam = outputs[0]
    end_beam = np.logical_and(end_beam, is_continues_beam).tolist()
    select_beam = [np.concatenate((sel_out, np.expand_dims(out, axis=1)), 1).tolist() for sel_out, out in zip(select_beam, outputs)]
    length_beam = beam_gather(length_beam, beam_indices)

    # next_states = beam_gather(next_states, beam_indices, deepcopy=True)
    # position_embedding = beam_gather(position_embedding, beam_indices, deepcopy=True)
    # code_embedding = beam_gather(code_embedding, beam_indices, deepcopy=True)

    beam_gather_args = [beam_gather(arg, beam_indices, deepcopy=True)for arg in beam_gather_args]

    inputs = [beam_gather(inp, beam_indices, deepcopy=True) for inp in inputs]
    # inputs = self._create_next_code(outputs, *inputs)

    return inputs, outputs, select_beam, end_beam, beam_score, length_beam, beam_gather_args


def beam_calculate_without_iscontinue(inputs, outputs_logit, beam_score, end_beam, length_beam, select_beam, beam_size, beam_calculate_output_score_fn, beam_gather_args):
    '''

    :param inputs:
    :param outputs_logit:
    :param beam_score:
    :param next_states:
    :param position_embedding:
    :param code_embedding:
    :param end_beam:
    :param length_beam:
    :param select_beam:
    :param beam_size:
    :param beam_calculate_output_score_fn: return p_beam, action_beam.
    :return:
    '''
    # is_continues_beam, positions_beam, is_copys_beam, keyword_ids_beam, copy_ids_beam = outputs

    # if np.sum(end_beam) == 0:
    #     outputs = [[0]*len(out) for out in outputs_logit]
    #     return inputs, outputs, select_beam, end_beam, beam_score, next_states, position_embedding, code_embedding, length_beam
    length_beam = (np.array(length_beam) + np.array(end_beam)).tolist()
    cur_beam_size = len(outputs_logit[1])

    p_beam, id_beam, action_beam = beam_calculate_output_score_fn(outputs_logit, beam_size)
    p_beam = [(p_b if end_b else [0]) for p_b, end_b in zip(p_beam, end_beam)]
    # print('before beam score: ', beam_score)
    # print('end beam: ', end_beam)
    # print('outputs_logit: ', outputs_logit)
    # print('beam_size: ', beam_size)
    p_score = beam_calculate_score(beam_score, p_beam)
    p_score_with_penalty = beam_calculate_length_penalty(length_beam, p_score)
    p_score = beam_flat(p_score)
    # print('p_score: ', p_score)
    # print('length_beam: ', length_beam)
    p_score_with_penalty = beam_flat(p_score_with_penalty)

    action_beam = beam_flat(action_beam)
    id_beam = beam_flat(id_beam)
    # print('p_score_with_penalty: ', p_score_with_penalty)
    top_indices = beam_cal_top_k(p_score_with_penalty, beam_size)
    # print('top_indices: ', top_indices)
    beam_score = beam_gather(p_score, top_indices)
    action_beam = beam_gather(action_beam, top_indices)
    beam_indices = beam_gather(id_beam, top_indices)

    # outputs_logit = [self.beam_gather(out, beam_indices) for out in outputs_logit]
    end_beam = beam_gather(end_beam, beam_indices)
    # outputs = beam_get_output_from_action_beams(action_beam)
    outputs = list(zip(*action_beam))
    outputs = [np.where(end_beam, out, np.zeros_like(out)).tolist() for out in outputs]
    # print('outputs:', outputs)
    select_beam = [beam_gather(sel_beam, beam_indices, deepcopy=True) for sel_beam in select_beam]
    # is_continues_beam = outputs[0]
    # end_beam = np.logical_and(end_beam, is_continues_beam).tolist()
    # print('outputs: ', outputs)
    select_beam = [np.concatenate((sel_out, np.expand_dims(out, axis=1)), 1).tolist() for sel_out, out in zip(select_beam, outputs)]
    length_beam = beam_gather(length_beam, beam_indices)

    # next_states = beam_gather(next_states, beam_indices, deepcopy=True)
    # position_embedding = beam_gather(position_embedding, beam_indices, deepcopy=True)
    # code_embedding = beam_gather(code_embedding, beam_indices, deepcopy=True)

    beam_gather_args = [beam_gather(arg, beam_indices, deepcopy=True)for arg in beam_gather_args]

    inputs = [beam_gather(inp, beam_indices, deepcopy=True) for inp in inputs]
    # inputs = self._create_next_code(outputs, *inputs)
    # print('after beam score: ', beam_score)

    return inputs, outputs, select_beam, end_beam, beam_score, length_beam, beam_gather_args


def _create_next_code_without_iter_dims(actions, inputs_without_iter, create_one_fn):
    create_one_next_code_fn = lambda zipped: create_one_fn(*zipped)
    next_inputs = list(map(create_one_next_code_fn, list(zip(list(zip(*actions)), *inputs_without_iter))))
    next_inputs = list(zip(*next_inputs))
    return next_inputs


def _create_next_code(actions, inputs, create_one_fn):
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
    # inputs = token_input, token_input_length, character_input, character_input_length
    remove_iter_dims_fn = lambda one_input: [one[0] for one in one_input]
    add_iter_dims_fn = lambda one_input: [[one] for one in one_input]

    inputs_without_iter = [remove_iter_dims_fn(one) for one in inputs]
    next_inputs = _create_next_code_without_iter_dims(actions, inputs_without_iter, create_one_fn)
    next_inputs = [add_iter_dims_fn(one) for one in next_inputs]

    return next_inputs


def fill_output_data(output:list, iter_len):
    res = [t + [0 for i in range((iter_len-len(t)))] for t in output]
    return np.array(res)


def cal_metrics(max_decode_iterator_num, output_data, predict_data):
    res_mask = []
    predict_is_continue = predict_data[0]
    for bat in predict_is_continue:
        zero_item = np.argwhere(np.array(bat) == 0)
        # print("bat:{}, zero_item:{}".format(bat, zero_item))
        if len(zero_item) == 0:
            iter_len = max_decode_iterator_num
        else:
            iter_len = np.min(zero_item) + 1
        res_mask.append([1 for i in range(iter_len)] + [0 for i in range(max_decode_iterator_num - iter_len)])
    res_mask = np.array(res_mask)
    # print("res_mask:{}".format(res_mask))

    true_mask = np.ones([len(output_data[0])])
    for i in range(len(predict_data)):
        # true_mask = 0
        output_idata = fill_output_data(output_data[i], max_decode_iterator_num)
        predict_idata = fill_output_data(predict_data[i], max_decode_iterator_num)

        predict_idata = np.where(res_mask, predict_idata, np.zeros_like(predict_idata))
        # print("index {}: output_data {}, predict_data {}".format(i, output_idata, predict_idata))

        res = np.equal(output_idata, predict_idata)
        res = res.reshape([res.shape[0], -1])
        res = np.all(res, axis=1)
        true_mask = np.logical_and(true_mask, res)
    return np.mean(true_mask)

def cal_metrics_without_iscontinue(max_decode_iterator_num, output_data, predict_data):
    true_mask = np.ones([len(output_data[0])])
    for i in range(len(predict_data)):
        # true_mask = 0
        output_idata = fill_output_data(output_data[i], max_decode_iterator_num)
        predict_idata = fill_output_data(predict_data[i], max_decode_iterator_num)

        # predict_idata = np.where(res_mask, predict_idata, np.zeros_like(predict_idata))
        # print("index {}: output_data {}, predict_data {}".format(i, output_idata, predict_idata))

        res = np.equal(output_idata, predict_idata)
        res = res.reshape([res.shape[0], -1])
        res = np.all(res, axis=1)
        true_mask = np.logical_and(true_mask, res)
    return np.mean(true_mask)


def find_copy_input_position(iden_mask, copy_id):
    for i in range(len(iden_mask)):
        iden = iden_mask[i]
        if iden[int(copy_id)] == 1:
            return i
    return -1


def beam_calculate_fn(args):
    return beam_calculate(*args)


def init_beam_search_stack(batch_size, cur_beam_size, output_num=5):
    # shape = batch_size * beam_size
    beam_stack = [[0] for i in range(batch_size)]
    # shape = 5 * batch_size * beam_size * output_length
    output_stack = []
    # shape = batch_size * beam_size
    mask_stack = [[1] for i in range(batch_size)]
    # shape = batch_size * beam_size
    beam_length_stack = [[0] for i in range(batch_size)]
    # shape = 5 * batch_size * beam_size * max_decode_iterator_num
    select_output_stack_list = [[[[] for i in range(cur_beam_size)] for j in range(batch_size)] for k in range(output_num)]
    return beam_length_stack, beam_stack, mask_stack, select_output_stack_list


def metrics_output_directly(output_data, predict_data):
    res_mask = [1 for i in range(len(output_data[0]))]
    for i in range(len(output_data)):
        res = np.equal(predict_data[i], output_data[i])
        res_mask = np.logical_and(res_mask, res)
    return res_mask


def make_mask_by_iter_num(i, length_list):
    mask = [1 if i < le else 0 for le in length_list]
    return mask


def calculate_length_by_one_input(one_input):
    batch_size = len(one_input)
    iter_size = len(one_input[0])
    one_input = np.array(one_input).reshape([batch_size, iter_size, -1])
    length_input = np.sum(one_input, axis=2)
    length_input = np.where(length_input, np.ones_like(length_input), np.zeros_like(length_input))
    length_input = np.sum(length_input, axis=1)
    return length_input.tolist()


def reshape_batch_length_list_to_mask_stack(batch_mask, cur_beam_size):
    batch_mask = np.array(batch_mask)
    batch_mask = np.expand_dims(batch_mask, axis=1)
    batch_mask = np.repeat(batch_mask, cur_beam_size, axis=1)
    return batch_mask.tolist()


def make_mask_stack_by_length_list(cur_beam_size, i, length_list):
    batch_mask = make_mask_by_iter_num(i, length_list)
    batch_mask = reshape_batch_length_list_to_mask_stack(batch_mask, cur_beam_size)
    return batch_mask