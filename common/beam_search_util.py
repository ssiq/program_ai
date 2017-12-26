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