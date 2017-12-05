import unittest
import tensorflow as tf

from model.token_level_multirnn_model import TokenLevelMultiRnnModel
from embedding.wordembedding import load_vocabulary
from embedding.character_embedding import load_character_vocabulary

class Test(unittest.TestCase):
    def setUp(self):
        with tf.Session():
            test_code = [["#include", "iostream"]]

            word_embedding = load_vocabulary("keyword", 200)
            character_embedding = load_character_vocabulary('bigru', 3, embedding_shape=100, token_list=test_code)
            self.model = TokenLevelMultiRnnModel(
                word_embedding.create_embedding_layer(),
                character_embedding.create_embedding_layer(),
                200,
                2,
                len(word_embedding.word_id_map),
                word_embedding.word_to_id(word_embedding.end_label),
                0.0001,
                6,
                word_embedding.word_to_id(word_embedding.identifier_label),
                word_embedding.word_to_id(word_embedding.placeholder_label),
                word_embedding.id_to_word,
                character_embedding.parse_token
            )

    def test_cal_metrics(self):
        is_continue = [[1, 1, 0], [1,1,1, 0], [0]]
        position_label = [[2, 3, 3], [2,45,3, 1], [2]]
        is_copy = [[1, 1, 0], [1, 0, 1, 1], [1]]
        keyword_id = [[0, 0, 1], [0, 2,0, 0], [0]]
        copy_word_id = [[12, 32, 0], [21, 0, 32,32], [12]]
        o = (is_continue, position_label, is_copy, keyword_id, copy_word_id)
        res = self.model.cal_metrics(o, o)
        self.assertAlmostEqual(res, 1.0)
