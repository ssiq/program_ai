import tensorflow as tf
import os
import typing


def make_dir(*path: str) -> None:
    """
    This method will recursively create the directory
    :param path: a variable length parameter
    :return:
    """
    path = os.path.join(*path)

    if not path:
        return

    if os.path.exists(path):
        if not os.path.isdir(path):
            raise ValueError("The path {} already exits but it is not a directory".format(path))
        return

    base, _ = os.path.split(path)
    make_dir(base)
    os.mkdir(path)


def load_check_point(checkpoint_path: str, sess: tf.Session, saver: tf.train.Saver) -> None:
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_path))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

def format_dict_to_string(to_format_dict: dict) -> str:
    """
    :param to_format_dict: a dict to format
    :return:
    """

    return '__'.join(str(a)+str(b) for a, b in to_format_dict.items())


if __name__ == '__main__':
    make_dir('data', 'cache_data')
