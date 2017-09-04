import tensorflow as tf
import os


def make_dir(path):
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise ValueError("The path {} already exits but it is not a directory".format(path))
        return
    os.mkdir(path)


def load_check_point(checkpoint_path, sess, saver):
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_path))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

def format_dict_to_string(to_format_dict: dict):
    """
    :param to_format_dict: a dict to format
    :return:
    """

    return '__'.join(str(a)+str(b) for a, b in to_format_dict.items())
