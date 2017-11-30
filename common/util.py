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
import logging


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

    return ('__'.join(str(a)+str(b) for a, b in to_format_dict.items()))[0:64]


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
    return list(map(lambda x:list(more_itertools.padded(x, fillvalue=-1, n=max_len)), batch_code))


def padded_code_new(batch_code):
    if not isinstance(batch_code, list):
        return batch_code
    elif not isinstance(batch_code[0], list):
        return batch_code

    batch_root = batch_code
    while True:
        if not isinstance(batch_root, list):
            return batch_code
        elif not isinstance(batch_root[0], list):
            return batch_code
        fill_value = 0
        if isinstance(batch_root[0][0], list):
            fill_value = []
        max_len = max(map(len, batch_root))
        for b in batch_root:
            while len(b) < max_len:
                b.append(fill_value)
        # list(map(lambda x: list(more_itertools.padded(x, fillvalue=fill_value, n=max_len)), batch_root))

        tmp = []
        for son in batch_root:
            for s in son:
                tmp.append(s)
        batch_root = tmp


def get_sign_list(code_string):
    from code_data.constants import char_sign_dict
    char_list = list(code_string)
    try:
        sign_list = [char_sign_dict[x] for x in char_list]
    except KeyError:
        return None
    assert len(code_string) == len(sign_list)
    return sign_list

def padded(x):
    import copy
    x = copy.deepcopy(x)
    if not isinstance(x, list):
        return x
    elif isinstance(x[0], list):
        return padded_code_new(x)
    else:
        return x

def batch_holder(*data: typing.List, batch_size=32, epoches=10):
    """
    :param data:
    :return:
    """
    def iterator():

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


def initCustomerLogger(name, filepath, level=logging.INFO):
    import time
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    logger = logging.getLogger(name)
    logger.setLevel(level=level)

    name = name + time.strftime('_%Y-%m-%d %H:%M:%S.log',time.localtime(time.time()))
    filepath = os.path.join(filepath, name)
    fh = logging.FileHandler(filename=filepath)
    fh.setLevel(level=level)
    fh.setFormatter(formatter)

    logger.addHandler(fh)


def initLogging():
    from code_data.constants import debug_logger_name_list, output_logger_name_list, DEBUG_LOG_PATH, OUTPUT_LOG_PATH
    for name in debug_logger_name_list:
        initCustomerLogger(name, DEBUG_LOG_PATH, level=logging.DEBUG)
    for name in output_logger_name_list:
        initCustomerLogger(name, OUTPUT_LOG_PATH, level=logging.INFO)


# ================================================================
# sequence function
# ================================================================

def is_sequence(s):
    try:
        iterator = iter(s)
    except TypeError:
        return False
    else:
        return True


def convert_to_list(s):
    if is_sequence(s):
        return list(s)
    else:
        return [s]


def sequence_sum(itr):
    return sum(itr)

def train_test_split(data, test_size):
    from sklearn.model_selection import train_test_split
    data = train_test_split(*data, test_size=test_size)

    d_len = len(data)
    train_data = [data[i] for i in range(0, d_len, 2)]
    test_data = [data[i] for i in range(1, d_len, 2)]
    return train_data, test_data

# ================================================================
# dict function
# ================================================================

def reverse_dict(d: dict) -> dict:
    """
    swap key and value of a dict
    dict(key->value) => dict(value->key)
    """
    return dict(map(reversed, d.items()))

if __name__ == '__main__':
    make_dir('data', 'cache_data')
