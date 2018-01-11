import unittest
import tensorflow as tf
from model.token_level_multirnn_model_with_output_mask import _sample_mask


class Test(unittest.TestCase):
    def test__sample_mask(self):
        tf.set_random_seed(10)
        mask = tf.constant(
            [[
                [1, 0],
                [0, 0],
                [0, 0],
                [1, 0]
            ],
             [
                 [1, 0],
                 [0, 1],
                 [0, 1],
                 [1, 0]
             ]]
        )
        o = _sample_mask(mask)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        print(sess.run(o))
