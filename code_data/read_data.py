import math
import sqlite3
import typing

import pandas as pd

from code_data.constants import cache_data_path
from common import util


def read_submit_data(conn: sqlite3.Connection) -> pd.DataFrame:
    problems_df = pd.read_sql('select problem_name, tags from {}'.format('problem'), conn)
    submit_df = pd.read_sql('select * from {}'.format('submit'), conn)
    submit_joined_df = submit_df.join(problems_df.set_index('problem_name'), on = 'problem_name')

    submit_joined_df['time'] = submit_joined_df['time'].str.replace('ms', '').astype('int')
    submit_joined_df['memory'] = submit_joined_df['memory'].str.replace('KB', '').astype('int')
    submit_joined_df['submit_time'] = pd.to_datetime(submit_joined_df['submit_time'])
    submit_joined_df['tags'] = submit_joined_df['tags'].str.split(':')
    submit_joined_df['code'] = submit_joined_df['code'].str.slice(1,-1)

    verdict = {'OK': 1, 'REJECTED': 2, 'WRONG_ANSWER': 3, 'RUNTIME_ERROR': 4, 'TIME_LIMIT_EXCEEDED': 5, 'MEMORY_LIMIT_EXCEEDED': 6,
               'COMPILATION_ERROR': 7, 'CHALLENGED': 8, 'FAILED': 9, 'PARTIAL': 10, 'PRESENTATION_ERROR': 11, 'IDLENESS_LIMIT_EXCEEDED': 12,
               'SECURITY_VIOLATED': 13, 'CRASHED': 14, 'INPUT_PREPARATION_CRASHED': 15, 'SKIPPED': 16, 'TESTING': 17, 'SUBMITTED': 18}
    langdict = {'GNU C': 1, 'GNU C11': 2, 'GNU C++': 3, 'GNU C++11': 4, 'GNU C++14': 5,
                'MS C++': 6, 'Mono C#': 7, 'MS C#': 8, 'D': 9, 'Go': 10,
                'Haskell': 11, 'Java 8': 12, 'Kotlin': 13, 'Ocaml': 14, 'Delphi': 15,
                'FPC': 16, 'Perl': 17, 'PHP': 18, 'Python 2': 19, 'Python 3': 20,
                'PyPy 2': 21, 'PyPy 3': 22, 'Ruby': 23, 'Rust': 24, 'Scala': 25,
                'JavaScript': 26}
    submit_joined_df['language'] = submit_joined_df['language'].replace(langdict)
    submit_joined_df['status'] = submit_joined_df['status'].replace(verdict)

    return submit_joined_df

def read_cpp_fake_data(conn: sqlite3.Connection) -> pd.DataFrame:
    fake_df = pd.read_sql('select * from {}'.format('fake_error_code'), conn)
    return fake_df

def read_cpp_token_fake_data(conn: sqlite3.Connection) -> pd.DataFrame:
    from code_data.constants import FAKE_ERROR_TOKEN_CODE
    fake_df = pd.read_sql('select * from {}'.format(FAKE_ERROR_TOKEN_CODE), conn)
    return fake_df

def read_cpp_fake_records(conn: sqlite3.Connection) -> pd.DataFrame:
    from code_data.constants import FAKE_CODE_RECORDS
    records_df = pd.read_sql('select * from {} LIMIT 400000'.format(FAKE_CODE_RECORDS), conn)
    return records_df

def read_local_submit_data() -> pd.DataFrame:
    from code_data.constants import local_db_path
    import sqlite3
    con = sqlite3.connect("file:{}?mode=ro".format(local_db_path), uri=True)
    return read_submit_data(con)

def read_local_fake_data() -> pd.DataFrame:
    from code_data.constants import local_db_path
    import sqlite3
    con = sqlite3.connect("file:{}?mode=ro".format(local_db_path), uri=True)
    return read_cpp_fake_data(con)

def read_local_token_fake_data() -> pd.DataFrame:
    from code_data.constants import local_db_path
    import sqlite3
    con = sqlite3.connect("file:{}?mode=ro".format(local_db_path), uri=True)
    return read_cpp_token_fake_data(con)

def read_code_list(filter_function: typing.Callable[[pd.DataFrame], pd.Series], head: int) -> pd.DataFrame:
    """
    This is a function to read code from database. It will cache the code list in the cache directory.
    :param filter_function: It get df parameter return a binary value list, not use the lambda expression
    :param head: it get the number of code
    :return:
    """
    import os
    from common.util import make_dir
    name = filter_function.__name__ + '_' + str(head)
    make_dir(cache_data_path)
    path = os.path.join(cache_data_path, name)
    if os.path.exists(path):
        return pd.read_pickle(path)
    else:
        df = read_local_submit_data()
        df = df[filter_function(df)].head(head)['code']
        df.to_pickle(path)
        return df

def read_fake_code_records() -> pd.DataFrame:
    from code_data.constants import local_db_path
    import sqlite3
    con = sqlite3.connect("file:{}?mode=ro".format(local_db_path), uri=True)
    return read_cpp_fake_records(con)

@util.disk_cache(basename='cpp', directory=cache_data_path)
def read_cpp_error_code_list() -> pd.DataFrame:
    print('start read cpp data from database')
    all_data = read_local_submit_data()
    df = all_data.loc[(all_data['status'] == 7) & (all_data['language'] == 3) & (all_data['code'].map(len) > 50), ['id', 'submit_url', 'code']]
    df['code_len'] = df['code'].map(len)
    df.sort_values(by=['code_len'])
    return df

@util.disk_cache(basename='less_cpp', directory=cache_data_path)
def read_less_cpp_code_list() -> pd.DataFrame:
    all_data = read_local_submit_data()
    df = all_data.loc[(all_data['status'] == 7) & (all_data['language'] == 3) & (all_data['code'].map(len) > 50), ['id', 'submit_url', 'code']]
    df['code_len'] = df['code'].map(len)
    df = df.sort_values(by=['code_len']).head(100)
    return df

@util.disk_cache(basename='length_cpp', directory=cache_data_path)
def read_length_cpp_code_list(code_length) -> pd.DataFrame:
    all_data = read_local_submit_data()
    df = all_data.loc[(all_data['status'] == 7) & (all_data['language'] == 3) & (all_data['code'].map(len) > 50) & (all_data['code'].map(len) < code_length), ['id', 'submit_url', 'code']]
    df['code_len'] = df['code'].map(len)
    df = df.sort_values(by=['code_len'])
    return df

@util.disk_cache(basename='code_cpp', directory=cache_data_path)
def read_cpp_code_list(status=1, language=3, code_length=math.inf) -> pd.DataFrame:
    all_data = read_local_submit_data()
    df = all_data.loc[(all_data['status'] == status) & (all_data['language'] == language) & (all_data['code'].map(len) > 50) & (all_data['code'].map(len) < code_length)]
    df['code_len'] = df['code'].map(len)
    df = df.sort_values(by=['code_len'])
    return df

@util.disk_cache(basename='fake_cpp_code', directory=cache_data_path)
def read_cpp_fake_code_list() -> pd.DataFrame:
    all_data = read_local_fake_data()
    all_data['code_len'] = all_data['code'].map(len)
    all_data = all_data.sort_values(by=['code_len'])
    return all_data

@util.disk_cache(basename='fake_cpp_token_code', directory=cache_data_path)
def read_cpp_token_fake_code_list() -> pd.DataFrame:
    all_data = read_local_token_fake_data()
    all_data['code_len'] = all_data['code'].map(len)
    all_data = all_data.sort_values(by=['code_len'])
    return all_data

@util.disk_cache(basename='fake_cpp_code_set', directory=cache_data_path)
def read_cpp_fake_code_set() -> tuple:
    df = read_cpp_fake_code_list()
    test = df.sample(10000, random_state=666)
    df = df.drop(test.index)
    vaild = df.sample(10000, random_state=888)
    train = df.drop(vaild.index)
    return (train, test, vaild)

@util.disk_cache(basename='fake_cpp_token_code_set', directory=cache_data_path)
def read_cpp_token_fake_code_set() -> pd.DataFrame:
    df = read_local_token_fake_data()
    test = df.sample(10000, random_state=666)
    df = df.drop(test.index)
    vaild = df.sample(10000, random_state=888)
    train = df.drop(vaild.index)
    return (train, test, vaild)


@util.disk_cache(basename='fake_code_records_set', directory=cache_data_path)
def read_cpp_fake_code_records_set() -> pd.DataFrame:
    df = read_fake_code_records()
    test = df.sample(10000, random_state=666)
    df = df.drop(test.index)
    vaild = df.sample(10000, random_state=888)
    train = df.drop(vaild.index)
    return (train, test, vaild)



