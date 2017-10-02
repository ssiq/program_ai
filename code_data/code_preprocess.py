import re
from code_data.read_data import read_cpp_code_list
from code_data.constants import cpp_tmp_dir, cpp_tmp_path, char_sign_dict, sign_char_dict
from database.error_code_database import insert_fake_error, find_submit_by_problem_user
from scripts.scripts_util import initLogging
import logging
import random
import multiprocessing as mp
import queue
import os
import time


def preprocess():
    initLogging()
    logging.info("Start Read Code Data")
    code_df = read_cpp_code_list()
    logging.info("Code Data Read Finish. Total: {}".format(code_df.shape[0]))
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
        item['submitid'] = row['id']
        item['problemid'] = row['problem_id']
        item['userid'] = row['user_id']
        item['originalcode'] = row['code']
        items.append(item)

        ids.append(item['id'])

        if len(ids) == 10000:
            push_code_to_queue(que_read, ids, items)
            logging.info('Total Preprocess {}'.format(count))
            ids = []
            items = []

    push_code_to_queue(que_read, ids, items)
    logging.info('Total Preprocess {}'.format(count))

    for p in pros:
        p.join()
    save_pro.join()


def push_code_to_queue(que, ids, items):
    ids = ["'" + t + "'" for t in ids]
    result_list = find_submit_by_problem_user(ids)
    ids_repeat = [row[0] for row in result_list]
    count = 0
    for it in items:
        if it['id'] not in ids_repeat:
            count += 1
            que.put(it)
        else:
            que.put(None)
    logging.info('Preprocess {} code in {}'.format(count, len(ids)))

def make_fake_code(que_read:mp.Queue, que_write:mp.Queue, ind:int):
    logging.info('Start Make Fake Code Process {}'.format(ind))
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
            logging.info("Process {} | count: {} | error_count: {} | fail_count: {}".format(ind, count, err_count, fail_count))

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

        item['originalcode'] = item['originalcode'].replace('\ufeff', '').replace('\u3000', ' ')

        before_code, after_code, act_type, act_pos, act_sign, error_count = preprocess_code(item['originalcode'], cpp_file_path=tmp_code_file_path)
        count += 1
        if before_code:
            success_count += 1
            item['code'] = after_code
            item['errorcount'] = error_count
            item['actiontype'] = act_type
            item['actionpos'] = act_pos
            item['actionsign'] = act_sign

            que_write.put(item)
        else:
            item['try_count'] += 1
            if item['try_count'] <= 3:
                err_count += 1
                que_read.put(item)
            else:
                fail_count += 1
                que_write.put(None)

    logging.info("Process {} | count: {} | error_count: {} | fail_count: {}".format(ind, count, err_count, fail_count))
    logging.info('End Make Fake Code Process {}'.format(ind))

def save_fake_code(que:mp.Queue, all_data_count):
    logging.info('Start Save Fake Code Process')
    count = 0
    param = []
    while True:
        if not que.empty() and count < all_data_count:
            item = que.get()
            count += 1
            if not item:
                continue
            param.append(item)
            if len(param) > 1000:
                logging.info('Save {} recode. Total record: {}'.format(len(param), count))
                insert_fake_error(dict_to_list(param))
                param = []
        elif que.empty() and count >= all_data_count:
            break
        else:
            time.sleep(10)
    logging.info('Save {} recode. Total record: {}'.format(len(param), count))
    insert_fake_error(dict_to_list(param))
    logging.info('End Save Fake Code Process')


def dict_to_list(param):
    param_list = []
    for pa in param:
        item = []
        item.append(pa['id'])
        item.append(pa['submitid'])
        item.append(pa['problemid'])
        item.append(pa['userid'])
        item.append(pa['originalcode'])
        item.append(pa['code'])
        item.append(pa['errorcount'])
        item.append(pa['actiontype'])
        item.append(pa['actionpos'])
        item.append(pa['actionsign'])
        param_list.append(item)
    return param_list


def preprocess_code(code, error_count=1, cpp_file_path=cpp_tmp_path):
    if not compile_code(code, cpp_file_path):
        return None, None, None, None, None, None
    before_code = code
    after_code = before_code
    act_type = -1
    act_pos = -1
    act_sign = -1

    count = 0
    while compile_code(after_code, cpp_file_path):
        cod = before_code
        cod = remove_blank(cod)
        cod = remove_comments(cod)
        cod = remove_blank_line(cod)
        count += 1
        cod, after_code, act_type, act_pos, act_sign = create_error(cod)
        if count > 10:
            return None, None, None, None, None, None

    return before_code, after_code, act_type, act_pos, act_sign, error_count


def save_code(problem_id, user_id, before_code, after_code, act_type, act_pos, act_sign):
    id = problem_id + '_' + user_id
    pass


def create_error(code, error_type_list=(1, 1, 1)):
    code_len = len(code)
    new_code = code
    if len(error_type_list) != 3:
        return code, -1
    res = random.uniform(0, sum(error_type_list))
    act_type = 0
    act_type = act_type + 1 if res > error_type_list[0] else act_type
    act_type = act_type + 1 if res > error_type_list[1] else act_type

    act_pos = -1
    act_cha_sign = -1
    if act_type == 0:
        pos = random.randint(0, code_len-1)
        new_code = new_code[:pos] + new_code[(pos + 1):]
        act_pos = pos * 2
        if code[pos] not in char_sign_dict.keys():
            return None, None, None, None, None
        act_cha_sign = char_sign_dict[code[pos]]
    elif act_type == 1:
        pos = random.randint(0, code_len)
        cha = sign_char_dict[random.randint(0, len(sign_char_dict)-2)]
        new_code = new_code[:pos] + cha +new_code[pos:]
        act_pos = pos * 2 + 1
        act_cha_sign = 96
    elif act_type == 2:
        pos = random.randint(0, code_len-1)
        cha = sign_char_dict[random.randint(0, len(sign_char_dict)-2)]
        new_code = new_code[:pos] + cha +new_code[(pos + 1):]
        act_pos = pos * 2 + 1
        if code[pos] not in char_sign_dict.keys():
            return None, None, None, None, None
        act_cha_sign = char_sign_dict[code[pos]]

    return code, new_code, act_type, act_pos, act_cha_sign


def compile_code(code, file_path) -> bool:
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    f = open(file_path, 'w')
    f.write(code)
    f.flush()
    f.close()
    res = os.system('g++ -O0 -fsyntax-only {} >/dev/null 2>&1'.format(file_path))
    if res == 0:
        return True
    return False


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


def remove_blank(code):
    pattern = re.compile('''('.*?'|".*?"|[^ \t\r\f\v"']+)''')
    mat = re.findall(pattern, code)
    processed_code = ' '.join(mat)
    return processed_code


if __name__ == '__main__':
    preprocess()

