from model import util
import unittest
import tensorflow as tf
import numpy as np
import test.test_package_util as test_util

class Test(unittest.TestCase):
    def test_variable_length_softmax(self):
        tf.reset_default_graph()
        x = tf.placeholder(shape=(None, None), dtype=tf.float32)
        length = tf.placeholder(shape=(None,), dtype=tf.int32)
        o = util.variable_length_softmax(x, length)
        ix = np.array([[5.0, 5.0, 5.0], [1.0, 1.0, 0.0]])
        il = np.array([3, 2])
        sess = tf.Session()
        l = sess.run(o, feed_dict={x: ix, length: il})
        self.assertTrue(
            test_util.almost_equal_array(l, np.array([[0.33333331, 0.33333331, 0.33333331], [0.5, 0.5, 0.18393973]])))
