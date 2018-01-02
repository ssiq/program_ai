import unittest
import tensorflow as tf
import numpy as np

from common.beam_search_util import _create_next_code, cal_metrics
from model.masked_token_level_multirnn_model import MaskedTokenLevelMultiRnnModel, flat_and_get_effective_args, get_effective_id, make_weight_array
from embedding.wordembedding import load_vocabulary
from embedding.character_embedding import load_character_vocabulary
from experiment.masked_token_level_multirnn_on_fake_cpp_data import parse_xy_with_identifier_mask
from experiment.experiment_util import sample, create_embedding
from common.util import padded
from test.test_package_util import almost_equal_array

"""
for one time only can run one test case
"""

class Test(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        tf.reset_default_graph()
        with tf.Session():
            word_embedding, character_embedding = create_embedding()
            self.model = MaskedTokenLevelMultiRnnModel(
                word_embedding.create_embedding_layer(),
                character_embedding.create_embedding_layer(),
                200,
                2,
                len(word_embedding.word_id_map),
                word_embedding.word_to_id(word_embedding.end_label),
                0.0001,
                1000,
                0.96,
                2,
                word_embedding.word_to_id(word_embedding.identifier_label),
                word_embedding.word_to_id(word_embedding.placeholder_label),
                word_embedding.id_to_word,
                character_embedding.parse_token
            )
        self.train, self.test, self.validation = sample()
        parse_xy_param = [word_embedding, character_embedding]
        self.train, self.test, self.validation = \
            parse_xy_with_identifier_mask(self.train, '',*parse_xy_param, max_bug_number=2, min_bug_number=2), \
            parse_xy_with_identifier_mask(self.test, '',*parse_xy_param, max_bug_number=2, min_bug_number=2), \
            parse_xy_with_identifier_mask(self.validation, '',*parse_xy_param, max_bug_number=2, min_bug_number=2)
        self.character_embedding = character_embedding


    def test_cal_metrics(self):
        is_continue = [[1, 1, 0], [1,1,1, 0], [0]]
        position_label = [[2, 3, 3], [2,45,3, 1], [2]]
        is_copy = [[1, 1, 0], [1, 0, 1, 1], [1]]
        keyword_id = [[0, 0, 1], [0, 2,0, 0], [0]]
        copy_word_id = [[12, 32, 0], [21, 0, 32,32], [12]]
        o = (is_continue, position_label, is_copy, keyword_id, copy_word_id)
        po = [np.array(padded(t)).tolist() for t in o]
        res = cal_metrics(5, o, po)
        self.assertAlmostEqual(res, 1.0)

    def test__create_next_code(self):
        args = [list(t[:10]) for t in self.train]
        # print("args_shape")
        # for t in args:
        #     print(np.array(t).shape)
        get_patch = lambda x: [padded([[a[x]] for a in t]) for t in args[:5]]
        get_action = lambda x: [padded([a[x] for a in t]) for t in args[5:]]
        inputs = get_patch(0)
        print("input_shape")
        for t in inputs:
            print(np.array(t).shape)
        result = get_patch(1)
        print("result_shape")
        for t in result:
            print(np.array(t).shape)
        action = get_action(0)
        print("action")
        print(action)
        res = _create_next_code(action, inputs[:5], self.model._create_one_next_code)
        res = [padded(t) for t in res]
        print("res_shape")
        for t in res:
            print(np.array(t).shape)
        print("character_begin_token:{}".format(self.character_embedding.character_to_id_dict[(self.character_embedding.BEGIN, )]))
        for i, (a, b) in enumerate(zip(res, result[:5])):
            print("test {}".format(i))
            print(a)
            print(b)
            a = np.array(a)
            b = np.array(b)
            print(np.argwhere((a==b)==False))
            error_position = [np.argwhere((a==b)==False)[:, i] for i in range(len(a.shape))]
            print("res:{}".format(a[error_position]))
            print("true:{}".format(b[error_position]))
            self.assertTrue(almost_equal_array(a, b))

    def test_flat_and_get_effective_args(self):
        args = [
            [
            [[1, 0, 0], [[0, 0], [0, 0], [0, 0, 0], [0], [0, 0]], [[[1], 0], [0, 0], [0, 0, 0], [0], [0, 0]]],
            [[[0, 0], [0, 0], [0, 0, 0], [0], [0, 0]], [[[1], 0], [0, 0], [0, 0, 0], [0], [0, 0]]]
        ]
        ]
        flat_args, eff_ids = flat_and_get_effective_args(args)
        self.assertEqual(flat_args, [[[1, 0, 0], [[[1], 0], [0, 0], [0, 0, 0], [0], [0, 0]], [[[1], 0], [0, 0], [0, 0, 0], [0], [0, 0]]]])
        self.assertEqual(eff_ids, [0, 2, 4])

    def test_get_effective_id(self):
        a = [[1, 0, 0], [[0, 0], [0, 0], [0, 0, 0], [0], [0, 0]], [[[1], 0], [0, 0], [0, 0, 0], [0], [0, 0]]]
        res = get_effective_id(a)
        self.assertEqual(res, [0, 2])

    def test_make_weight_array(self):
        weight_array = make_weight_array(2, 3)
        self.assertEqual(weight_array, [2, 1])
        weight_array = make_weight_array(2, 2)
        self.assertEqual(weight_array, [2])
        weight_array = make_weight_array(2, 1)
        self.assertEqual(weight_array, [1])
        weight_array = make_weight_array(2, 7)
        self.assertEqual(weight_array, [2, 2, 2, 1])





