from model.bi_rnn import build_bi_rnn
from test.test_package_util import almost_equal_array
import numpy as np
import tensorflow as tf
import unittest
import math


class BuildBiRnnTest(unittest.TestCase):
    def test_build_bi_rnn(self):
        tf.reset_default_graph()
        np.random.seed(10)
        tf.set_random_seed(1)
        inf = math.inf
        x = tf.placeholder(dtype=tf.int32, shape=(None, None), name='x')
        outputs = build_bi_rnn(x, 32, np.random.randn(20, 32), 4)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        o = sess.run(outputs, feed_dict={x: np.array([[1, 2, 3], [2, 3, -1]])})
        r = almost_equal_array(o,
                               np.array([[-0.12153799, -0.23034486, -0.28465682, -0.22284305, 0.01000046, -0.15733679,-0.20365854, -0.38925439, 0.05213334, 0.11997216, -0.32148057, -0.37703431,
                                          0.60620207, -0.06309471, 0.14420858, -0.41868153, 0.68483686, -0.46944386,
                                          0.19035062, -0.10814955, 0.75968105, -0.5217731, 0.46471033, -0.39412326,
                                          0.76045138, -0.16238804, 0.82104713, -0.49403226],
                                         [-0.07940511, 0.04696409, -0.40247884, -0.21062298, 0.53047502, -0.11390948,
                                          0.09350666, -0.29160762, 0.60910982, -0.52025867, 0.13964871, 0.01892438,
                                          0.69716948, -0.52892387, 0.40801582, -0.2746318, 0.69793981, -0.16953878,
                                          0.76435262, -0.37454081, -inf, -inf, -inf, -inf,
                                          -inf, -inf, -inf, -inf]]))
        self.assertTrue(r)
