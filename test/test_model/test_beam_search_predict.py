import math
import unittest

from common.beam_search_util import *
from common.beam_search_util import beam_calculate
from experiment.experiment_util import sample, create_embedding, parse_xy_with_identifier_mask
from model.masked_token_level_multirnn_model import *


class TestBeamSearch(unittest.TestCase):
    # @classmethod
    # def setUpClass(self):
    #     tf.reset_default_graph()
    #     with tf.Session():
    #         word_embedding, character_embedding = create_embedding()
    #         self.model = MaskedTokenLevelMultiRnnModel(
    #             word_embedding.create_embedding_layer(),
    #             character_embedding.create_embedding_layer(),
    #             200,
    #             2,
    #             len(word_embedding.word_id_map),
    #             word_embedding.word_to_id(word_embedding.end_label),
    #             0.0001,
    #             1000,
    #             0.96,
    #             2,
    #             word_embedding.word_to_id(word_embedding.identifier_label),
    #             word_embedding.word_to_id(word_embedding.placeholder_label),
    #             word_embedding.id_to_word,
    #             character_embedding.parse_token
    #         )
    #     self.train, self.test, self.validation = sample()
    #     parse_xy_param = [word_embedding, character_embedding]
    #     self.train, self.test, self.validation = \
    #         parse_xy_with_identifier_mask(self.train, 'train', *parse_xy_param, max_bug_number=2, min_bug_number=2), \
    #         parse_xy_with_identifier_mask(self.test, 'valid', *parse_xy_param, max_bug_number=2, min_bug_number=2), \
    #         parse_xy_with_identifier_mask(self.validation, 'test', *parse_xy_param, max_bug_number=2, min_bug_number=2)
    #     self.character_embedding = character_embedding
    #     print('End SetUpClass')

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def init_inputs(self):
        input_token = [[[1, 2, 0, 0, 0]],
                       [[7, 8, 9, 10, 11]],
                       [[15, 0, 0, 0, 0]]]
        input_token_length = [[2],
                              [5],
                              [1]]
        input_character = [
            [[[1, 1, 1, 1, 0, 0], [2, 2, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]],
            [[[7, 7, 7, 0, 0, 0], [8, 0, 0, 0, 0, 0], [9, 9, 0, 0, 0, 0], [10, 10, 10, 10, 10, 0], [11, 11, 11, 11, 0, 0]]],
            [[[15, 15, 15, 15, 15, 15], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]]]
        input_character_length = [[[4, 3, 0, 0, 0]],
                                  [[3, 1, 2, 5, 4]],
                                  [[6, 0, 0, 0, 0]]]
        return input_token, input_token_length, input_character, input_character_length

    def test_length_penalty(self):
        length1 = length_penalty(1)
        self.assertEqual(length1, 1)
        length2 = length_penalty(5)
        score2 = length2 * float((5. + 1.) ** 0.6)
        self.assertEqual(round(score2, 8), round((float(5. + 5) ** 0.6), 8))

    def test_beam_calculate_length_penalty(self):
        beam_length = [1, 2, 3, 4, 5]
        beam_output_p = [[0.1, 0.3, 0.4, 0.2], [0.1, 0.5, 0.4], [0.3, 0.3, 0.4], [0.1, 0.3, 0.4, 0.2], [0.3, 0.7]]
        res = beam_calculate_length_penalty(beam_length, beam_output_p)
        self.assertEqual(len(res), 5)
        self.assertEqual(len(res[0]), 4)
        self.assertEqual(len(res[1]), 3)
        self.assertEqual(len(res[4]), 2)
        first = res[0][0]
        self.assertEqual(round(res[0][1], 8), round(3 * first, 8))
        self.assertEqual(round(res[0][2], 8), round(4 * first, 8))
        self.assertEqual(round(res[0][3], 8), round(2 * first, 8))

    def test_beam_calculate_score(self):
        beam_old_p = [-0.1, -0.2, -0.3]
        beam_output_p = [[-0.005, -0.015, -0.025], [-0.005, -0.015], [-0.005]]
        res = beam_calculate_score(beam_old_p, beam_output_p)
        self.assertEqual(len(res), 3)
        self.assertEqual(len(res[0]), 3)
        self.assertEqual(len(res[1]), 2)
        self.assertEqual(len(res[2]), 1)
        self.assertEqual(round(res[0][0], 8), -0.105)
        self.assertEqual(round(res[1][1], 8), -0.215)

    def test_beam_cal_top_k(self):
        one_batch = [i for i in range(15)]
        res = beam_cal_top_k(one_batch, 5)
        self.assertEqual(len(res), 5)
        self.assertEqual(res[0], 14)
        self.assertEqual(res[4], 10)

    def test_flat_list(self):
        l = [[1, [5]], 2, 3]
        res = flat_list(l, levels=2)
        self.assertEqual(len(res), 4)
        self.assertEqual(res, [1, 5, 2, 3])
        res2 = flat_list(l, levels=1)
        self.assertEqual(len(res2), 4)
        self.assertEqual(res2, [1, [5], 2, 3])

    def test_beam_flat(self):
        one_batch = [[1, 3, 5], [2, 4, 6], [2, 3, 4]]
        res = beam_flat(one_batch)
        self.assertEqual(res,  [1, 3, 5, 2, 4, 6, 2, 3, 4])

    def test_select_max_output(self):
        beam_stack = [[1, 2, 3], [8, 5, 1], [1, 5, 4], [2, 2, 1]]
        select_stack = [[[[1, 1, 1], [1, 1, 2], [1, 1, 3]], [[1, 2, 1], [1, 2, 2], [1, 2, 3]], [[1, 3, 1], [1, 3, 2], [1, 3, 3]], [[1, 4, 1], [1, 4, 2], [1, 4, 3]]],
                        [[[2, 1, 1], [2, 1, 2], [2, 1, 3]], [[2, 2, 1], [2, 2, 2], [2, 2, 3]], [[2, 3, 1], [2, 3, 2], [2, 3, 3]], [[2, 4, 1], [2, 4, 2], [2, 4, 3]]],
                        [[[3, 1, 1], [3, 1, 2], [3, 1, 3]], [[3, 2, 1], [3, 2, 2], [3, 2, 3]], [[3, 3, 1], [3, 3, 2], [3, 3, 3]], [[3, 4, 1], [3, 4, 2], [3, 4, 3]]]]
        res = select_max_output(beam_stack, select_stack)
        self.assertEqual(res, [[[1, 1, 3], [1, 2, 1], [1, 3, 2], [1, 4, 1]], [[2, 1, 3], [2, 2, 1], [2, 3, 2], [2, 4, 1]], [[3, 1, 3], [3, 2, 1], [3, 3, 2], [3, 4, 1]]])

    def test_init_input_stack(self):
        input_token, input_token_length, input_character, input_character_length = self.init_inputs()
        self.assertEqual(np.array(input_token).shape, (3, 1, 5))
        self.assertEqual(np.array(input_token_length).shape, (3, 1))
        self.assertEqual(np.array(input_character).shape, (3, 1, 5, 6))
        self.assertEqual(np.array(input_character_length).shape, (3, 1, 5))
        args = (input_token, input_token_length, input_character, input_character_length)
        res = init_input_stack(args)
        self.assertEqual(len(res), 4)
        self.assertEqual(np.array(res[0]).shape, (3, 1, 1, 5))
        self.assertEqual(np.array(res[1]).shape, (3, 1, 1))
        self.assertEqual(np.array(res[2]).shape, (3, 1, 1, 5, 6))
        self.assertEqual(np.array(res[3]).shape, (3, 1, 1, 5))

    def test_revert_batch_beam_stack(self):
        batch_size = 4
        beam_size = 3
        output_dig = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        output_list = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        output_mix = [[0], [0], [0], [0], [0], [0], [0, 0], [[0, 0, 0]], [0], [0], 0, [0]]
        res_dig = revert_batch_beam_stack(output_dig, batch_size, beam_size)
        res_list = revert_batch_beam_stack(output_list, batch_size, beam_size)
        res_mix = revert_batch_beam_stack(output_mix, batch_size, beam_size)
        self.assertEqual(np.array(res_dig).shape, (4, 3))
        self.assertEqual(res_dig, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        self.assertEqual(np.array(res_list).shape, (4, 3, 1))
        self.assertEqual(res_list, [[[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]]])
        self.assertEqual(np.array(res_mix).shape, (4, 3))
        self.assertEqual(res_mix, [[[0], [0], [0]], [[0], [0], [0]], [[0, 0], [[0, 0, 0]], [0]], [[0], 0, [0]]])

    def test_beam_gather(self):
        predict = [0, [1], 2, 3, 4, 5, 6, 7, 8, 9]
        indices = [5, 2, 4, 7, 1]
        self.assertEqual(beam_gather(predict, indices), [5, 2, 4, 7, [1]])

    # def test_beam_get_key_from_action(self):
    #     action_list = [{'k1': 'v1'+str(i), 'k2': 'v2'+str(i)} for i in range(5)]
    #     self.assertEqual(beam_get_key_from_action(action_list, 'k1'), ['v10', 'v11', 'v12', 'v13', 'v14'])
    #     self.assertEqual(beam_get_key_from_action(action_list, 'k2'), ['v20', 'v21', 'v22', 'v23', 'v24'])

    # def test_beam_top_to_beamid(self):
    #     action_list = [{'beam_id': i} for i in range(5)]
    #     self.assertEqual(beam_top_to_beamid(action_list), [0, 1, 2, 3, 4])

    def test_beam_calculate_output_score(self):
        import math
        beam_size = 3
        output_beam_list = self.init_output_logits()
        beam_p_stack, beam_id_stack, beam_action_stack = beam_calculate_output_score(output_beam_list, beam_size)
        self.assertEqual(len(beam_p_stack), 3)
        self.assertEqual(len(beam_p_stack[0]), 24)
        self.assertEqual(len(beam_p_stack[1]), 24)
        self.assertEqual(len(beam_p_stack[2]), 16)
        self.assertEqual(round(beam_p_stack[0][0], 8), round(math.log(0.168), 8))
        self.assertEqual(round(beam_p_stack[0][-1], 8), round(math.log(0.0048), 8))

    def init_output_logits(self):
        is_continue = [0.7, 0.3, 0.6]
        positions = [[0.2, 0.3, 0.1, 0.4], [0.3, 0.4, 0.3], [0.8, 0.2]]
        is_copy = [0.4, 0.5, 0.1]
        keyword_ids = [[1], [0.5, 0.5], [0.3, 0.3, 0.4]]
        copy_ids = [[0.2, 0.3, 0.5], [0.7, 0.3], [1]]
        return is_continue, positions, is_copy, keyword_ids, copy_ids

    def test_beam_calculate(self):
        inputs = self.init_inputs()
        outputs_logit = self.init_output_logits()
        beam_score = [-10, -10, -1]
        next_states = [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]
        position_embedding = [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]
        code_embedding = [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]
        end_beam = [1, 1, 1]
        length_beam = [1, 1, 1]
        select_beam = [([1, 1], [1, 1], [1, 1]), ([], [], []), ([1, 0], [0, 1], [0, 0]), ([], [], []), ([], [], [])]
        beam_size = 3
        res = beam_calculate(inputs, outputs_logit, beam_score, end_beam, length_beam, select_beam, beam_size, beam_calculate_output_score, [next_states, position_embedding, code_embedding])
        inputs, outputs, select_beam, end_beam, beam_score, length_beam, beam_args = res
        next_states, position_embedding, code_embedding = list(zip(*beam_args))
        print(inputs[0][0])
        self.assertEqual(outputs[0][0], 1)
        self.assertEqual(outputs[1][0], 0)
        self.assertEqual(outputs[2][0], 0)
        self.assertEqual(outputs[3][0], 2)
        self.assertEqual(outputs[4][0], 0)
        self.assertEqual(len(beam_score), 3)
        self.assertEqual(next_states[0], [3, 3, 3, 3, 3])
        self.assertEqual(position_embedding[0], [3, 3, 3, 3, 3])
        self.assertEqual(code_embedding[0], [3, 3, 3, 3, 3])
        self.assertEqual(length_beam, [2, 2, 2])
        self.assertEqual(round(beam_score[0], 8), -1 + round(math.log(0.1728), 8))
        self.assertEqual(select_beam[0][0], [1, 1, 1])
        self.assertEqual(select_beam[2][0], [0, 0, 0])
        self.assertEqual(select_beam[1], [[0], [0], [0]])
        self.assertEqual(end_beam[0], 1)


if __name__ == '__main__':
    unittest.main()
