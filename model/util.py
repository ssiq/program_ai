import functools
import tensorflow as tf
import typing


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def init_all_op(self: typing.Any) -> None:
    """
    The function is used to add all op in the model to the default tensorflow graph.
    """
    for i in dir(self):
        if i.endswith('op'):
            getattr(self, i)


def length(seq: tf.Tensor) -> tf.Tensor:
    """
    :param seq: a tenser of sequence [batch, time, ....]
    :return: the length of the sequence
    """
    return tf.reduce_sum(tf.reduce_max(tf.sign(seq), 2), 1)


def perplexity(logits: tf.Tensor, one_hot_labels: tf.Tensor):
    """
    :param logits: the logits which has the size [batch, time, label_size]
    :param one_hot_labels: the one_hot_labels with the size [batch, time, label_size]
    :return: the perplexity
    """
    length_of_series = length(one_hot_labels)
    pp = tf.nn.softmax(logits, )
    pp_one_product = tf.log(pp) * one_hot_labels
    exponent_sum = -tf.reduce_sum(pp_one_product, axis=tf.range(1, get_shape(tf.shape(pp))[0]))
    return tf.reduce_mean(tf.pow(tf.constant(2.0, dtype=tf.float32),
                                 exponent_sum / length_of_series))


def get_shape(tensor: tf.Tensor) -> typing.List:
  """Returns static shape if available and dynamic shape otherwise."""
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims


if __name__ == '__main__':
    class Test(object):
        def __init__(self):
            print("begin init")
            init_all_op(self)
            print("end init")

        @define_scope
        def a_op(self):
            print("a_op inited")

        @define_scope
        def b_op(self):
            print("b_op inited")

        @define_scope
        def c(self):
            print("c is not a op")


    test = Test()
