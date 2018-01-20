import pandas as pd
import Levenshtein
import json
import numpy as np
from code_data.read_data import read_student_local_data
from common.new_tokenizer import tokenize
from scripts.scripts_util import remove_comments, remove_blank_line, remove_r_char, remove_blank
from experiment.experiment_util import do_new_tokenize
from database.database_util import run_sql_statment
from database.sql_statment import sql_dict
from code_data.constants import local_student_db_path, STUDENT_BUILD_INFO
import random

CHANGE = 0
INSERT = 1
DELETE = 2


def levenshtenin_distance_iterator(leven_matrix, a_list, b_list, i, j, equal_fn=lambda a, b: a == b, max_distance=None):
    if max_distance != None and abs(i-j) > max_distance:
        leven_matrix[i][j] = max_distance + 1
        return max_distance + 1
    if leven_matrix[i][j] != None and max_distance != None and leven_matrix[i][j] > max_distance:
        return max_distance + 1

    if i == 0:
        leven_matrix[i][j] = j
        return j
    elif j == 0:
        leven_matrix[i][j] = i
        return i

    if leven_matrix[i][j] != None:
        return leven_matrix[i][j]

    bias = 0 if equal_fn(a_list[i-1], b_list[j-1]) else 1
    insert = levenshtenin_distance_iterator(leven_matrix, a_list, b_list, i, j - 1, equal_fn, max_distance) + 1
    change = levenshtenin_distance_iterator(leven_matrix, a_list, b_list, i - 1, j - 1, equal_fn, max_distance) + bias
    delete = levenshtenin_distance_iterator(leven_matrix, a_list, b_list, i - 1, j, equal_fn, max_distance) + 1

    leven_matrix[i][j] = min(insert, change, delete)

    return leven_matrix[i][j]


def levenshtenin_distance(a_list, b_list, equal_fn=lambda a, b: a == b, max_distance=None):
    a_len = len(a_list)
    b_len = len(b_list)
    matrix = make_metrix(a_len, b_len)
    res = levenshtenin_distance_iterator(matrix, a_list, b_list, a_len, b_len, equal_fn, max_distance)
    return res, matrix


def make_metrix(i, j):
    return [[None for k in range(j+1)] for o in range(i+1)]


def init_code(code):
    code = code.replace('\ufeff', '').replace('\u3000', ' ')
    code = remove_blank(code)
    code = remove_r_char(code)
    code = remove_comments(code)
    code = remove_blank_line(code)
    return code

def do_tokenize(code):
    code = code.replace('\r', '')
    if code.find('define') != -1 or code.find('defined') != -1 or code.find('undef') != -1 or \
                    code.find('pragma') != -1 or code.find('ifndef') != -1 or \
                    code.find('ifdef') != -1 or code.find('endif') != -1:
        return None
    try:
        tokens = do_new_tokenize(code)
    except Exception as e:
        return None
    return tokens


def equal_fn(x, y):
    x_val = x.value
    y_val = y.value
    if isinstance(x_val, list):
        x_val = ''.join(x_val)
    if isinstance(y_val, list):
        y_val = ''.join(y_val)
    return x_val == y_val


def token_value_fn(x):
    val = x.value
    if isinstance(val, list):
        val = ''.join(val)
    return val

def split_file_name(id:str):
    strs = id.split('_')
    return strs[0]+strs[1]

count = 0
def find_closest_token_text(one, build_ac_bf):
    global count
    print('find closest token. total {}'.format(count))
    count += 1
    a_tokenize = one['tokenize']
    file_name = one['file_name']
    build_ac_bf = build_ac_bf[build_ac_bf['file_name'].map(lambda x: x == file_name)]

    cal_distance_fn = lambda x: levenshtenin_distance(a_tokenize, x, equal_fn=equal_fn, max_distance=5)[0]
    distance_series = build_ac_bf['tokenize'].map(cal_distance_fn)
    max_id = distance_series.idxmax()
    max_value = distance_series.loc[max_id]
    if max_value > 5:
        one['similar_code'] = ''
        one['action_list'] = []
        return one

    matrix = levenshtenin_distance(a_tokenize, build_ac_bf['tokenize'].loc[max_id], equal_fn=equal_fn, max_distance=5)[1]
    b_tokenize = build_ac_bf['tokenize'].loc[max_id]
    action_list = cal_action_list(matrix, a_tokenize, b_tokenize, equal_fn, token_value_fn)

    one['similar_code'] = build_ac_bf['file_content'].loc[max_id]
    one['action_list'] = action_list

    return one


def left_move_action(i, j, a_token, b_token, value_fn=lambda x: x):
    action = {'act_type': DELETE, 'from_char': value_fn(b_token), 'to_char': '', 'token_pos': j-1}
    return action


def top_move_action(i, j, a_token, b_token, value_fn=lambda x: x):
    action = {'act_type': INSERT, 'from_char': '', 'to_char': value_fn(a_token), 'token_pos': j}
    return action


def left_top_move_action(matrix, i, j, a_token, b_token, value_fn=lambda x: x):
    if matrix[i][j] == matrix[i-1][j-1]:
        return None
    action = {'act_type': CHANGE, 'from_char': value_fn(b_token), 'to_char': value_fn(a_token), 'token_pos': j-1}
    return action


def check_action(token):
    if token == None:
        return True
    if isinstance(token.value, list):
        return False
    return True


def get_action(matrix, i, j, a_token, b_token, equal_fn, value_fn=lambda x: x):
    if i == 0 and j == 0:
        return None, -1, -1
    if i == 0:
        action = left_move_action(i, j, a_token, b_token, value_fn)
        j -= 1
        return action, i, j
    if j == 0:
        action = top_move_action(i, j, a_token, b_token, value_fn)
        i -= 1
        return action, i, j

    bias = 1
    if equal_fn(a_token, b_token):
        bias = 0

    # print('one: ', i, j, bias)
    # print('{} {} \n{} {}'.format(matrix[i-1][j-1], matrix[i-1][j], matrix[i][j-1], matrix[i][j]))

    if matrix[i][j] == (matrix[i-1][j-1] + bias):
        # do left_top
        action = left_top_move_action(matrix, i, j, a_token, b_token, value_fn)
        i -= 1
        j -= 1
    elif matrix[i][j] == (matrix[i-1][j] + 1):
        # do top
        action = top_move_action(i, j, a_token, b_token, value_fn)
        i -= 1
    elif matrix[i][j] == (matrix[i][j-1] + 1):
        #do left
        action = left_move_action(i, j, a_token, b_token, value_fn)
        j -= 1
    else:
        print('get action position error')
        return None, None, None

    return action, i, j


def cal_action_list(matrix, a_tokens, b_tokens, equal_fn=lambda x, y: x == y, value_fn=lambda x: x):
    len_a = len(a_tokens)
    len_b = len(b_tokens)

    action_list = []
    i = len_a
    j = len_b
    while i >= 0 and j >= 0:
        a_token = a_tokens[i-1] if i > 0 else None
        b_token = b_tokens[j-1] if j > 0 else None
        if check_action(a_token) or check_action(b_token):
            return None
        action, i, j = get_action(matrix, i, j, a_token, b_token, equal_fn, value_fn)
        if action is not None:
            action_list = action_list + [action]
        if i is None:
            return None

    return action_list


def recovery_code(tokens, action_list):
    # action_list.reverse()

    for act in action_list:
        act_type = act['act_type']
        pos = act['token_pos']
        from_char = act['from_char']
        to_char = act['to_char']
        if act_type == INSERT:
            tokens = tokens[0:pos] + [to_char] + tokens[pos:]
        elif act_type == DELETE:
            tokens = tokens[0:pos] + tokens[pos+1:]
        elif act_type == CHANGE:
            tokens = tokens[0:pos] + [to_char] + tokens[pos+1:]
    return tokens


def testLe():
    import random, string
    def randomword(length):
        letters = string.ascii_lowercase
        return ''.join([random.choice(letters) for i in range(length)])

    for i in range(1000):
        len_a = random.randint(250, 300)
        len_b = random.randint(250, 300)
        a = randomword(len_a)
        b = randomword(len_b)
        res, matrix = levenshtenin_distance(a, b)
        real_res = Levenshtein.distance(a, b)
        assert res == real_res

        print('in Test: ', len(a), len(b))
        act_list = cal_action_list(matrix, a, b)
        real_act_list = Levenshtein.editops(b, a)
        assert len(act_list), len(real_act_list)

        b = recovery_code(list(b), act_list)
        b = ''.join(b)
        assert len(b) == len(a)
        for i in range(len(b)):
            assert b[i] == a[i]
        assert b == a


if __name__ == '__main__':
    # testLe()

    print('Start')
    build_bf = read_student_local_data()
    # build_bf = build_bf.sample(100)
    print('Start . Total {} code'.format(len(build_bf.index)))

    build_bf['file_content'] = build_bf['file_content'].map(init_code)

    build_bf['tokenize'] = build_bf['file_content'].map(do_tokenize)
    build_bf = build_bf[build_bf['tokenize'].map(lambda x: x is not None)].copy()
    print('After tokenize. Total {} code'.format(len(build_bf.index)))

    build_bf['token_len'] = build_bf['tokenize'].map(len)
    build_bf = build_bf[build_bf['token_len'].map(lambda x: x < 300)].copy()
    print('After length. Total {} code'.format(len(build_bf.index)))

    build_bf['file_name'] = build_bf['id'].map(split_file_name)

    build_error_bf = build_bf[build_bf['build_result'].map(lambda x: x == 0)].copy()
    build_ac_bf = build_bf[build_bf['build_result'].map(lambda x: x == 1)].copy()
    print('Build error. Total {} code'.format(len(build_error_bf.index)))
    print('Build success. Total {} code'.format(len(build_ac_bf.index)))

    build_error_bf = build_error_bf.apply(find_closest_token_text, axis=1, raw=True, build_ac_bf=build_ac_bf)
    build_effect_bf = build_error_bf[build_error_bf['similar_code'].map(lambda x: x != '')].copy()
    print('final effective code Total {}'.format(len(build_effect_bf.index)))

    store_list = build_error_bf[['similar_code', 'action_list', 'id']].to_dict('list')
    store_list = list(zip(store_list['similar_code'], store_list['action_list'], store_list['id']))
    json_presist = lambda x: [x[0], json.dumps(x[1]), x[2]]
    store_list = [json_presist(i) for i in store_list]

    run_sql_statment(local_student_db_path, STUDENT_BUILD_INFO, 'update_similar_code', store_list)

