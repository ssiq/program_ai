import sklearn
import tensorflow as tf
import typing
import functools
import itertools
import more_itertools
import cytoolz as toolz
import pickle
import errno
import os


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


def ensure_directory(directory):
    """
    Create the directories along the provided directory path that do not exist.
    """
    directory = os.path.expanduser(directory)
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

def disk_cache(basename, directory, method=False):
    """
    Function decorator for caching pickleable return values on disk. Uses a
    hash computed from the function arguments for invalidation. If 'method',
    skip the first argument, usually being self or cls. The cache filepath is
    'directory/basename-hash.pickle'.
    """
    directory = os.path.expanduser(directory)
    ensure_directory(directory)

    def wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            key = (tuple(args), tuple(kwargs.items()))
            # Don't use self or cls for the invalidation hash.
            if method and key:
                key = key[1:]
            filename = '{}-{}.pickle'.format(basename, hash(key))
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as handle:
                    return pickle.load(handle)
            result = func(*args, **kwargs)
            with open(filepath, 'wb') as handle:
                pickle.dump(result, handle)
            return result
        return wrapped

    return wrapper


def overwrite_graph(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with tf.Graph().as_default():
            return function(*args, **kwargs)
    return wrapper


def padded_code(batch_code):
    '''
    padded code string to the same length.
    :param batch_code: a list of code string. batch obs list contains several code list.
     each code list consist of sign charactor. sign' type is int. char_sign_dict see constants.char_to_sign and sign_to_char
    :return:
    '''
    if not isinstance(batch_code, list):
        return batch_code
    elif not isinstance(batch_code[0], list):
        return batch_code
    max_len = max(map(len, batch_code))
    # print("max_len:{}".format(max_len))
    return list(map(lambda x:list(more_itertools.padded(x, fillvalue=-1, n=max_len)), batch_code))


def get_sign_list(code_string):
    from code_data.constants import char_sign_dict
    char_list = list(code_string)
    try:
        sign_list = [char_sign_dict[x] for x in char_list]
    except KeyError:
        return None
    assert len(code_string) == len(sign_list)
    return sign_list


def batch_holder(*data: typing.List, batch_size=32, epoches=10):
    """
    :param data:
    :return:
    """
    def iterator():
        def padded(x):
            if not isinstance(x, list):
                return x
            elif isinstance(x[0], list):
                return padded_code(x)
            else:
                return x


        def one_epoch():
            i_data = sklearn.utils.shuffle(*data)
            i_data = list(map(lambda x:map(padded, more_itertools.chunked(x, batch_size)), i_data))
            return zip(*i_data)
        for m in more_itertools.repeatfunc(one_epoch, times=epoches):
            for t in m:
                yield t

    return iterator


def dataset_holder(*args):
    def f():
        return args
    return f


def set_cuda_devices(deviceid:int=0):
    '''
    set video card which cuda uses. if you want to use both card, do not call this function.
    :param deviceid: video card id. default is 0
    :return:
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceid)


if __name__ == '__main__':
    make_dir('data', 'cache_data')
