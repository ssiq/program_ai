import json
import logging
import re

import numpy as np
import scandir

from code_data.read_data import read_student_local_data, read_local_test_code_records
from experiment.experiment_util import do_new_tokenize


def initLogging() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def scan_dir(dir_path, pattern=None, dir_level=-1):
    def inner_scan_project(in_path, level=0):
        for entry in scandir.scandir(in_path):
            if dir_level == level:
                if pattern is None:
                    yield entry.path
                else:
                    if pattern(entry.name):
                        yield entry.path
                continue
            if entry.is_dir():
                yield from inner_scan_project(entry.path, level=level+1)
            elif entry.is_file():
                if pattern is None:
                    yield entry.path
                else:
                    if pattern(entry.name):
                        yield entry.path
    yield from inner_scan_project(dir_path)


def remove_comments(code):
    pattern = r"(\".*?(?<!\\)\"|\'.*?(?<!\\)\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return "" # so we will return empty to remove the comment
        else: # otherwise, we will return the 1st group
            return match.group(1) # captured quoted-string
    return regex.sub(_replacer, code)


def remove_blank_line(code):
    code = "\n".join([line for line in code.split('\n') if line.strip() != ''])
    return code


def remove_r_char(code):
    code = code.replace('\r', '')
    return code


def remove_blank(code):
    pattern = re.compile('''('.*?'|".*?"|[^ \t\r\f\v"']+)''')
    mat = re.findall(pattern, code)
    processed_code = ' '.join(mat)
    return processed_code


def get_student_test_set():
    # build_df = read_student_code_bf()
    build_df = read_student_code_test_records_bf()
    build_df['res'] = ''
    build_df = build_df.apply(filter_actions, raw=True, axis=1)
    build_df = build_df[build_df['res'].map(lambda x: x is not None)].copy()

    build_df['code'] = build_df['code'].map(init_code)
    build_df['tokenize'] = build_df['code'].map(do_tokenize)
    build_df = build_df[build_df['tokenize'].map(lambda x: x is not None)].copy()

    build_df = build_df.apply(filter_include, raw=True, axis=1)
    build_df = build_df[build_df['res'].map(lambda x: x is not None)].copy()

    return build_df


def read_student_code_bf():
    build_df = read_student_local_data()
    build_df = build_df[build_df['distance'].map(lambda x: x < 6 and x > 0)]
    build_df['modify_action_list'] = build_df['modify_action_list'].map(json.loads)
    return build_df


def read_student_code_test_records_bf():
    build_df = read_local_test_code_records()
    build_df = build_df[build_df['distance'].map(lambda x: x < 6 and x > 0)]
    build_df['action_character_list'] = build_df['action_character_list'].map(json.loads)
    return build_df


def filter_actions(one):
    actions = one['action_character_list']
    error_code = one['code']
    ac_code = one['ac_code']
    res_list = [check_action(act, error_code) for act in actions]
    res = np.sum(res_list)
    if res > 0:
        one['res'] = None
    return one


def filter_include(one):
    tokens = one['tokenize']
    for token in tokens:
        if isinstance(token.value, list):
            inc_str = ''.join(token.value)
            if '"' in inc_str:
                one['res'] = None
                return one
    return one


def check_action(action, error_code):
    res = has_include(action)
    if not res:
        return 1
    res = check_from_char(action, error_code)
    if not res:
        return 1
    return 0

include_error_count = 0
def has_include(action):
    global include_error_count
    from_char = action['from_char']
    to_char = action['to_char']
    if 'include' in from_char or 'include' in to_char:
        include_error_count += 1
        return False
    return True

from_char_error_count = 0
def check_from_char(action, error_code):
    global from_char_error_count
    from_char = action['from_char']
    if from_char not in error_code:
        from_char_error_count += 1
        return False
    return True


def init_code(code):
    code = code.replace('\ufeff', '').replace('\u3000', ' ')
    code = remove_blank(code)
    code = remove_r_char(code)
    code = remove_comments(code)
    code = remove_blank_line(code)
    return code


token_count = 0
special_count = 0
exception_count = 0
def do_tokenize(code):
    global token_count, special_count, exception_count
    if token_count % 1000 == 0:
        print('tokenize count: {}'.format(token_count))
    token_count += 1
    code = code.replace('\r', '')
    if code.find('define') != -1 or code.find('defined') != -1 or code.find('undef') != -1 or \
                    code.find('pragma') != -1 or code.find('ifndef') != -1 or \
                    code.find('ifdef') != -1 or code.find('endif') != -1:
        special_count += 1
        return None
    try:
        tokens = do_new_tokenize(code)
    except Exception as e:
        exception_count += 1
        return None
    return tokens