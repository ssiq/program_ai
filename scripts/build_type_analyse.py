import pandas as pd
import numpy as np
import more_itertools
import json

from common.new_tokenizer import tokenize, operators, keywords
from database.database_util import create_table, insert_items
from code_data.constants import local_student_db_path, STUDENT_TEST_BUILD_ERROR_STAT
from scripts.scripts_util import get_student_test_set


def add_dict(one_dict: dict, key, value):
    if one_dict == None:
        return
    if key in one_dict.keys():
        one_dict[key] += value
    else:
        one_dict[key] = value


def sort_dict_value(one_dict:dict, key_fn=lambda x: x[1], reverse=True):
    sort_list = sorted(one_dict.items(), key=key_fn, reverse=True)
    return sort_list


def cal_percentage(sorted_value, pattern_fn=lambda x: x[1]):
    total_list = [pattern_fn(i) for i in sorted_value]
    total = np.sum(total_list)
    sorted_value = [list(i) + [pattern_fn(i)/total] for i in sorted_value]
    sorted_value = [pattern_fn(i)/total for i in sorted_value]
    return sorted_value


def get_top_percent(percented_value, max_percent=0.9):
    i = 0
    while np.sum(percented_value[0:i]) <= max_percent and i < len(percented_value):
        i += 1
    return i


def part_code_normalize(code:str):
    if len(code) > 1 and code[1] == ':':
        return '<FILE_PATH>'
    tokens = tokenize(code)
    i = len(tokens) - 1
    while i >= 0:
        token = tokens[i]
        word = '<' + token.type + '>'
        if token.type in operators.keys() or token.type in keywords.values():
            word = token.value
        code = code[:token.lexpos] + word + code[token.lexpos + len(token.value):]
        i -= 1
    return code


def normalize_error_message(message:str):
    import re
    split_sign = "'"
    if message.find('“') >= 0:
        split_sign = '“”'
    # strs = message.split("'")
    strs = re.split(split_sign, message)
    for i in range(1, len(strs), 2):
        try:
            strs[i] = part_code_normalize(strs[i])
        except Exception as e:
            # print('{} | {}'.format(message, strs[i]))
            pass
    return "'".join(strs)


def summary_build_info_by_error_code(build_df):
    build_res = {}
    for data in build_df['build_error_info']:
        for da in data:
            if da['code'][0] == 'C':
                add_dict(build_res, da['code'], 1)
    return build_res


def summary_build_info_by_message(build_df):
    build_res = {}
    for data in build_df['build_error_info']:
        for da in data:
            if da['code'][0] == 'C':
                da['message'] = normalize_error_message(da['message'])
                add_dict(build_res, da['message'], 1)
    return build_res


def summary_build_info_by_message_with_code(build_df):
    build_res = {}
    # print(build_df)
    build_df.apply(cal_one_build, axis=1, raw=True, build_res=build_res)
    return build_res


def cal_one_build(one, build_res):
    # print('one', one)
    error_message = one['build_error_info']
    cpp_code = one['file_content']
    if not check_one_file(cpp_code):
        return
    for da in error_message:
        if da['code'][0] == 'C':
            da['message'] = normalize_error_message(da['message'])
            if da['message'] not in build_res.keys():
                item = [0, da['code'], cpp_code]
                build_res[da['message']] = item
            item = build_res[da['message']]
            item[0] += 1
            # add_dict(build_res, da['message'], 1)


def check_one_file(code:str):
    code = code.replace('\r', '')
    lines = code.split('\n')
    import re
    pattern = re.compile(r'^#include ".*"$')
    for line in lines:
        match = pattern.match(line)
        if match:
            return False
    return True


def save_build_error(build_res):
    create_table(local_student_db_path, STUDENT_TEST_BUILD_ERROR_STAT)
    insert_items(local_student_db_path, STUDENT_TEST_BUILD_ERROR_STAT, build_res)


if __name__ == '__main__':
    build_df = get_student_test_set()
    print(len(build_df.index))
    # build_res = summary_build_info_by_error_code(build_df)
    # build_res = summary_build_info_by_message(build_df)
    # build_res = sort_dict_value(build_res)
    # build_res = [list(bui) + [per] for bui, per in zip(build_res, percent_build_res)]

    build_dict = summary_build_info_by_message_with_code(build_df)
    build_res = sort_dict_value(build_dict, lambda x: x[1][0])
    percent_build_res = cal_percentage(build_res, lambda x: x[1][0])
    build_res = [list(bui) + [per] for bui, per in zip(build_res, percent_build_res)]
    build_res = [list(more_itertools.collapse(bui, levels=1)) for bui in build_res]

    n = get_top_percent(list(zip(*build_res))[-1], 0.8)
    # n = 20

    save_build_error(build_res[:n])

    res_sum = np.sum(list(zip(*build_res[:n]))[-1])
    print(len(build_res), n, res_sum)
