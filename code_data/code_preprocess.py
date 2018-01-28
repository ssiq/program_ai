from code_data.read_data import read_cpp_code_list
import json
import logging
import multiprocessing as mp
import os
import queue
# from scripts.scripts_util import initLogging
import random
import time

from code_data.action_mapitem import ACTION_MAPITEM, ERROR_CHARACTER_MAPITEM
from code_data.constants import cpp_tmp_dir, cpp_tmp_path, COMMON_ERROR_TOKEN_CODE_RECORDS, pre_defined_cpp_token, \
    local_token_code_db
from code_data.error_action_reducer import create_error_action_fn
from code_data.read_data import read_cpp_code_list
from common.new_tokenizer import tokenize
from database.database_util import insert_items, create_table, find_ids_by_user_problem_id
from scripts.scripts_util import remove_comments, remove_blank_line, remove_r_char, remove_blank

preprocess_logger = logging.getLogger('code_preprocess')
# 设置logger的level为DEBUG

preprocess_logger.setLevel(logging.DEBUG)
preprocess_logger.__setattr__('propagate', False)
# 创建一个输出日志到控制台的StreamHandler
hdr = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
hdr.setFormatter(formatter)
# 给logger添加上handler
preprocess_logger.addHandler(hdr)


error_max_count = 1
failed_max_count = 1

def preprocess():
    # initLogging()
    preprocess_logger.info("Start Read Code Data")
    code_df = read_cpp_code_list()
    preprocess_logger.info("Code Data Read Finish. Total: {}".format(code_df.shape[0]))
    que_read = mp.Queue()
    que_write = mp.Queue()

    pros = []
    for i in range(8):
        pro = mp.Process(target=make_fake_code, args=(que_read, que_write, i))
        pro.start()
        pros.append(pro)
    save_pro = mp.Process(target=save_fake_code, args=(que_write, code_df.shape[0]))
    save_pro.start()

    count = 0
    ids = []
    items = []
    for index, row in code_df.iterrows():
        count += 1

        item = {'try_count': 0}
        item['id'] = row['problem_id'] + '_' + row['user_id']
        item['submit_id'] = row['id']
        item['problem_id'] = row['problem_id']
        item['user_id'] = row['user_id']
        item['originalcode'] = row['code'].replace('\ufeff', '').replace('\u3000', ' ')
        items.append(item)

        ids.append(item['id'])

        if len(ids) == 10000:
            push_code_to_queue(que_read, ids, items)
            preprocess_logger.info('Total Preprocess {}'.format(count))
            ids = []
            items = []

    push_code_to_queue(que_read, ids, items)
    preprocess_logger.info('Total Preprocess {}'.format(count))

    for p in pros:
        p.join()
    save_pro.join()


def push_code_to_queue(que, ids, items):
    ids = ["'" + t + "'" for t in ids]
    result_list = find_ids_by_user_problem_id(db_full_path=local_token_code_db, table_name=COMMON_ERROR_TOKEN_CODE_RECORDS, ids=ids)
    ids_repeat = [row[0] for row in result_list]
    count = 0
    for it in items:
        if it['id'] not in ids_repeat:
            count += 1
            que.put(it)
        else:
            que.put(None)
    preprocess_logger.info('Preprocess {} code in {}'.format(count, len(ids)))

def make_fake_code(que_read:mp.Queue, que_write:mp.Queue, ind:int):
    preprocess_logger.info('Start Make Fake Code Process {}'.format(ind))
    tmp_code_file_path = os.path.join(cpp_tmp_dir, 'code'+str(ind)+'.cpp')
    timeout_count = 0
    count = 0
    success_count = 0
    err_count = 0
    fail_count = 0
    repeat_count = 0
    while True:
        if timeout_count >= 5:
            break

        if count % 1000 == 0:
            preprocess_logger.info("Process {} | count: {} | error_count: {} | fail_count: {} | repeat_count: {}".format(ind, count, err_count, fail_count, repeat_count))

        try:
            item = que_read.get(timeout=600)
        except queue.Empty:
            timeout_count += 1
            continue
        except TimeoutError:
            timeout_count += 1
            continue

        timeout_count = 0
        count += 1
        if not item:
            repeat_count += 1
            que_write.put(None)
            continue

        # item['originalcode'] = item['originalcode'].replace('\ufeff', '').replace('\u3000', ' ')

        try:
            before_code, after_code, action_maplist, error_character_maplist, error_count = preprocess_code(item['originalcode'], cpp_file_path=tmp_code_file_path)
        except:
            before_code = None
            after_code = None
            action_maplist = None
            error_character_maplist = None
            error_count = 1

        count += 1
        if before_code:
            success_count += 1
            item['ac_code'] = before_code
            item['code'] = after_code
            item['error_count'] = error_count
            error_list = list(map(lambda x: x.__dict__(), error_character_maplist))
            action_list = list(map(lambda x: x.__dict__(), action_maplist))
            item['error_character_maplist'] = error_list
            item['action_maplist'] = action_list
            que_write.put(item)
        else:
            item['try_count'] += 1
            if item['try_count'] <= error_max_count:
                err_count += 1
                que_read.put(item)
            else:
                fail_count += 1
                que_write.put(None)

    preprocess_logger.info("Process {} | count: {} | error_count: {} | fail_count: {} | repeat_count: {}".format(ind, count, err_count, fail_count,  repeat_count))
    preprocess_logger.info('End Make Fake Code Process {}'.format(ind))


def save_fake_code(que:mp.Queue, all_data_count):
    create_table(db_full_path=local_token_code_db, table_name=COMMON_ERROR_TOKEN_CODE_RECORDS)
    preprocess_logger.info('Start Save Fake Code Process')
    count = 0
    error_count = 0
    param = []
    while True:
        if not que.empty() and count < all_data_count:
            try:
                item = que.get()
            except TypeError as e:
                print('Save get Type Error')
                error_count += 1
                print(error_count, count)
                continue
            count += 1
            if count % 10000 == 0:
                preprocess_logger.info('Total receive records: {}'.format(count))
            if not item:
                continue
            param.append(item)
            if len(param) > 1000:
                preprocess_logger.info('Save {} recode. Total record: {}'.format(len(param), count))
                insert_items(db_full_path=local_token_code_db, table_name=COMMON_ERROR_TOKEN_CODE_RECORDS, params=dict_to_list(param))
                param = []
        elif que.empty() and count >= all_data_count:
            break
        else:
            time.sleep(10)
    preprocess_logger.info('Save {} recode. Total record: {}'.format(len(param), count))
    insert_items(db_full_path=local_token_code_db, table_name=COMMON_ERROR_TOKEN_CODE_RECORDS, params=dict_to_list(param))
    preprocess_logger.info('End Save Fake Code Process')


def dict_to_list(param):
    param_list = []
    for pa in param:
        item = []
        item.append(pa['id'])
        item.append(pa['submit_id'])
        item.append(pa['problem_id'])
        item.append(pa['user_id'])
        item.append(pa['ac_code'])
        item.append(pa['code'])
        item.append(pa['error_count'])
        # error_list = list(map(lambda x: x.__dict__(), pa['error_character_maplist']))
        # action_list = list(map(lambda x: x.__dict__(), pa['action_character_list']))
        error_list = json.dumps(pa['error_character_maplist'])
        action_list = json.dumps(pa['action_maplist'])
        item.append(error_list)
        item.append(action_list)
        param_list.append(item)
    return param_list


def preprocess_code(code, cpp_file_path=cpp_tmp_path):
    if not compile_code(code, cpp_file_path):
        return None, None, None, None, None
    code = remove_blank(code)
    code = remove_r_char(code)
    code = remove_comments(code)
    code = remove_blank_line(code)
    if not compile_code(code, cpp_file_path):
        return None, None, None, None, None
    before_code = code
    after_code = before_code
    error_count_range = (1, 5)

    count = 0
    action_maplist = []
    error_character_maplist = []
    error_count = -1
    while compile_code(after_code, cpp_file_path):
        cod = before_code
        # cod = remove_blank(cod)
        # cod = remove_comments(cod)
        # cod = remove_blank_line(cod)
        count += 1
        # before_code = cod
        before_code, after_code, action_maplist, error_character_maplist, error_count = create_error_code(cod, error_count_range=error_count_range)
        if count > 10:
            return None, None, None, None, None

    return before_code, after_code, action_maplist, error_character_maplist, error_count

CHANGE = 0
INSERT = 1
DELETE = 2
STAY = 3
FILL = 4

def fill_blank_to_error_code(error_character_maplist, ac_i, err_i):
    item = ERROR_CHARACTER_MAPITEM(act_type=FILL, from_char=' ', err_pos=err_i, ac_pos=ac_i)
    error_character_maplist.append(item)
    return error_character_maplist


def create_error_code(code, error_type_list=(5, 1, 4), error_count_range=(1, 1)):
    error_count = random.randint(*error_count_range)
    action_maplist = create_multi_error(code, error_type_list, error_count)
    action_mapposlist = list(map(lambda x: x.get_ac_pos(), action_maplist))
    error_character_maplist = []
    ac_code_list = list(code)

    ac_i = 0
    err_i = 0

    def get_action(act_type, ac_pos):
        for i in action_maplist:
            if act_type == i.act_type and ac_pos == i.ac_pos:
                return i
        return None

    # for ac_i in range(len(ac_code_list)):
    while ac_i < len(ac_code_list):
        if ac_i in action_mapposlist and get_action(act_type=DELETE, ac_pos=ac_i) != None:
            action = get_action(act_type=DELETE, ac_pos=ac_i)
            error_character_maplist = fill_blank_to_error_code(error_character_maplist, ac_i, err_i)
            err_i += 1

            action.err_pos = err_i
            ac_i += len(action.from_char)

            error_character_maplist = fill_blank_to_error_code(error_character_maplist, ac_i, err_i)
            err_i += 1
            continue

        if ac_i in action_mapposlist and get_action(act_type=INSERT, ac_pos=ac_i) != None:
            action = get_action(act_type=INSERT, ac_pos=ac_i)
            error_character_maplist = fill_blank_to_error_code(error_character_maplist, ac_i, err_i)
            err_i += 1

            action.err_pos = err_i
            for i in range(len(action.to_char)):
                err_item = ERROR_CHARACTER_MAPITEM(act_type=INSERT, from_char=action.to_char[i], err_pos=err_i, ac_pos=ac_i)
                error_character_maplist.append(err_item)
                err_i += 1

            error_character_maplist = fill_blank_to_error_code(error_character_maplist, ac_i, err_i)
            err_i += 1

        if ac_i in action_mapposlist and get_action(act_type=CHANGE, ac_pos=ac_i) != None:
            action = get_action(act_type=CHANGE, ac_pos=ac_i)
            error_character_maplist = fill_blank_to_error_code(error_character_maplist, ac_i, err_i)
            err_i += 1

            action.err_pos = err_i
            for i in range(len(action.to_char)):
                err_item = ERROR_CHARACTER_MAPITEM(act_type=CHANGE, from_char=action.to_char[i], err_pos=err_i, to_char=action.from_char, ac_pos=ac_i)
                err_i += 1
                error_character_maplist.append(err_item)
            ac_i += len(action.from_char)

            error_character_maplist = fill_blank_to_error_code(error_character_maplist, ac_i, err_i)
            err_i += 1

        else:
            err_item = ERROR_CHARACTER_MAPITEM(act_type=STAY, from_char=code[ac_i], err_pos=err_i, to_char=code[ac_i],
                                               ac_pos=ac_i)
            err_i += 1
            error_character_maplist.append(err_item)
            ac_i += 1

    if ac_i in action_mapposlist and get_action(act_type=INSERT, ac_pos=ac_i) != None:
        action = get_action(act_type=INSERT, ac_pos=ac_i)
        error_character_maplist = fill_blank_to_error_code(error_character_maplist, ac_i, err_i)
        err_i += 1

        action.err_pos = err_i
        for i in range(len(action.to_char)):
            err_item = ERROR_CHARACTER_MAPITEM(act_type=INSERT, from_char=action.to_char[i], err_pos=err_i, ac_pos=ac_i)
            error_character_maplist.append(err_item)
            err_i += 1

        error_character_maplist = fill_blank_to_error_code(error_character_maplist, ac_i, err_i)
        err_i += 1

    error_code = ''.join(list(map(lambda x: x.from_char, error_character_maplist)))

    return code, error_code, action_maplist, error_character_maplist, error_count


def create_multi_error(code, error_type_list=(5, 1, 4), error_count=1):
    # code_len = len(code)
    if len(error_type_list) != 3:
        return []
    try:
        code_tokens = tokenize(code)
    except Exception as e:
        # print('tokenize code error.')
        return []

    action_maplist = []
    token_pos_list = []
    try_count = 0
    while len(action_maplist) < error_count and try_count < failed_max_count:
        error_action_fn = create_error_action_fn()
        # act_type, pos, token_pos, from_char, to_char = error_action_fn(code, code_tokens)
        action_tuple_list = error_action_fn(code, code_tokens)
        if action_tuple_list == None:
            try_count += 1
            continue
        without_insert_pos_list = [i[2] if i[0] != INSERT else -1 for i in action_tuple_list]
        token_pos_tmp_list = filter(lambda x: x != -1, without_insert_pos_list)
        # token_pos_tmp_list = [i[2] for i in action_tuple_list]
        while len(set(token_pos_tmp_list) & set(token_pos_list)) > 0 and try_count < failed_max_count:
            action_tuple_list = error_action_fn(code, code_tokens)
            if action_tuple_list == None:
                try_count += 1
                continue
            without_insert_pos_list = [i[2] if i[0] != INSERT else -1 for i in action_tuple_list]
            token_pos_tmp_list = filter(lambda x: x != -1, without_insert_pos_list)
            # token_pos_tmp_list = [i[2] for i in action_tuple_list]
            try_count += 1

        if try_count >= failed_max_count:
            break
        token_pos_list.extend(token_pos_tmp_list)
        for act in action_tuple_list:
            act_type, pos, token_pos, from_char, to_char = act
            action_item = ACTION_MAPITEM(act_type=act_type, ac_pos=pos, token_pos=token_pos, from_char=from_char, to_char=to_char)
            action_maplist.append(action_item)

    if try_count >= failed_max_count:
        return []

    if len(action_maplist) > 5:
        return []

    return action_maplist


def create_identifier_set(tokens, keyword_set=pre_defined_cpp_token):
    tokens_value = [tok.value for tok in tokens]
    tokens_value_set = set(filter(lambda x: not isinstance(x, list), tokens_value))
    identify_set = tokens_value_set - keyword_set
    return identify_set


def create_keyword_weight(special_keyword_weight:dict=None):
    token_weight_dict = dict((name, 1) for name in pre_defined_cpp_token)
    if special_keyword_weight is not None:
        for k in special_keyword_weight.keys():
            token_weight_dict[k] = special_keyword_weight[k]
    return token_weight_dict

# def create_error(code, error_type_list=(1, 1, 1), error_count=1):
#     code_len = len(code)
#     new_code = code
#     if len(error_type_list) != 3:
#         return code, -1
#     res = random.uniform(0, sum(error_type_list))
#     act_type = 0
#     act_type = act_type + 1 if res > error_type_list[0] else act_type
#     act_type = act_type + 1 if res > error_type_list[1] else act_type
# 
#     act_pos = -1
#     act_cha_sign = -1
#     if act_type == 0:
#         pos = random.randint(0, code_len-1)
#         new_code = new_code[:pos] + new_code[(pos + 1):]
#         act_pos = pos * 2
#         if code[pos] not in char_sign_dict.keys():
#             return None, None, None, None, None
#         act_cha_sign = char_sign_dict[code[pos]]
#     elif act_type == 1:
#         pos = random.randint(0, code_len)
#         cha = sign_char_dict[random.randint(0, len(sign_char_dict)-2)]
#         new_code = new_code[:pos] + cha +new_code[pos:]
#         act_pos = pos * 2 + 1
#         act_cha_sign = 96
#     elif act_type == 2:
#         pos = random.randint(0, code_len-1)
#         cha = sign_char_dict[random.randint(0, len(sign_char_dict)-2)]
#         new_code = new_code[:pos] + cha +new_code[(pos + 1):]
#         act_pos = pos * 2 + 1
#         if code[pos] not in char_sign_dict.keys():
#             return None, None, None, None, None
#         act_cha_sign = char_sign_dict[code[pos]]
# 
#     return code, new_code, act_type, act_pos, act_cha_sign


def compile_code(code, file_path) -> bool:
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    f = open(file_path, 'w')
    f.write(code)
    f.flush()
    f.close()
    # res = os.system('g++ -O0 {} >/dev/null 2>&1'.format(file_path))
    # res = os.system(r'E: & cd E:\Program Files (x86)\Microsoft Visual Studio 12.0\VC & vcvarsall.bat & cl {} /Fet.exe /nologo /EHsc /c> nul 2>&1'.format(file_path))
    res = os.system(r'E: & cd E:\Program Files (x86)\Microsoft Visual Studio 12.0\VC & vcvarsall.bat & cl {} /Fet.exe /nologo /EHsc /c'.format(file_path))
    if res == 0:
        return True
    return False


if __name__ == '__main__':
    preprocess()

