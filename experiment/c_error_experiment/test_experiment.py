from common.analyse_include_util import replace_include_with_blank, extract_include
from common.c_code_tokenize import tokenize_by_clex_fn
from experiment.c_error_experiment.experiment_util_for_c import create_c_embedding
from experiment.experiment_util import parse_xy_token_level_without_iscontinue, create_embedding, action_list_sorted
from common.test_supervision_util import create_test_experiment
from code_data.read_data import read_test_experiment_by_experiment_name
from common.beam_search_util import find_copy_input_position
from scripts.scripts_util import do_tokenize
from code_data.constants import local_c_test_experiment_db, TEST_EXPERIMENT_RECORDS
from common import util
from database.database_util import run_sql_statment
from scripts.student_build_closest_text import levenshtenin_distance, equal_fn
from scripts.scripts_util import init_code
from common.beam_search_util import metrics_output_directly

import pandas as pd
import random
import copy
from ply.lex import LexToken
import os
import json
import multiprocessing


def recovery_one_action_tokens(action, tokens, identifier_mask, placeholder_token, id_to_word_fn):
    position, is_copy, keyword_id, copy_id = action
    position, is_copy, keyword_id, copy_id = int(position), int(is_copy), int(keyword_id), int(copy_id)
    next_inputs = tokens, identifier_mask
    code_length = len(tokens)

    if position % 2 == 1 and is_copy == 0 and keyword_id == placeholder_token:
        # delete
        position = int(position / 2)
        if position >= code_length:
            # action position error
            print('delete action position error', position, code_length)
            return next_inputs
        tokens = tokens[0:position] + tokens[position + 1:]
        if identifier_mask is not None:
            identifier_mask = identifier_mask[0:position] + identifier_mask[position + 1:]
    else:
        if is_copy:
            # print(copy_id, identifier_mask)
            copy_position_id = find_copy_input_position(identifier_mask, copy_id)
            # copy_position_id = copy_id
            if copy_position_id >= code_length:
                # copy position error
                print('copy position error', copy_position_id, code_length)
                print('details:', position, is_copy, keyword_id, copy_position_id, code_length)
                return next_inputs
            word_token = LexToken()
            word_token.type = tokens[copy_position_id].type
            word_token.value = tokens[copy_position_id].value
            word_token.lineno = -1
            word_token.lexpos = -1
            if identifier_mask is not None:
                iden_mask = identifier_mask[copy_position_id]
        else:
            word = id_to_word_fn(keyword_id)
            if word == None:
                # keyword id error
                print('keyword id error', keyword_id)
                return next_inputs
            word_token = LexToken()
            word_token.value = word
            word_token.type = ''
            word_token.lineno = -1
            word_token.lexpos = -1
            if identifier_mask is not None:
                iden_mask = [0 for i in range(len(identifier_mask[0]))]

        if position % 2 == 0:
            # insert
            position = int(position / 2)
            if position > code_length:
                # action position error
                print('insert action position error', position, code_length)
                return next_inputs
            tokens = tokens[0:position] + [word_token] + tokens[position:]
            if identifier_mask is not None:
                identifier_mask = identifier_mask[0:position] + [iden_mask] + identifier_mask[position:]
        elif position % 2 == 1:
            # change
            position = int(position / 2)
            if position >= code_length:
                # action position error
                print('change action position error', position, code_length)
                return next_inputs
            tokens[position] = word_token
            if identifier_mask is not None:
                identifier_mask[position] = iden_mask
    next_inputs = tokens, identifier_mask
    return next_inputs


def recovery_tokens(actions, tokens, identifiers, key_val, output_length=None):
    plh_token_id = key_val.word_to_id(key_val.placeholder_label)
    id_to_word_fn = key_val.id_to_word
    i = 0
    # identifiers = None
    for position, is_copy, keyword_id, copy_id in actions:
        if output_length != None and i == output_length:
            break
        i += 1
        action = [position, is_copy, keyword_id, copy_id]
        tokens, identifiers = recovery_one_action_tokens(action, tokens, identifiers, plh_token_id, id_to_word_fn)
    return tokens


convert_code_count = 0
def convert_token_to_code(tokens, includes=None):
    lastLine = 1
    code = ''
    lastInclude = False
    last_token = None
    for i in range(len(tokens)):
        token = tokens[i]
        # print(token.lineno)
        if lastInclude or (lastLine != token.lineno and token.lineno != -1):
            code += '\n'
        val = token.value
        if isinstance(token.value, list):
            if not lastInclude:
                code += '\n'
            val = ''.join(token.value)
            lastInclude = True
        else:
            lastInclude = False
        skip_val = ' '
        if last_token is not None and last_token.type == 'TOK_INT_LITERAL' and token.type == 'TOK_FLOAT_LITERAL':
            skip_val = ''
        elif last_token is not None and last_token.type == 'TOK_INT_LITERAL' and token.type == 'ID':
            skip_val = ''
        if last_token is not None and last_token.type == 'TOK_INT_LITERAL' and token.type == 'ID' and (token.value == 'E' or token.value == 'e'):
            skip_val = ''
        if last_token is not None and (last_token.value == 'E' or last_token.value == 'e') and (token.type == 'TOK_INT_LITERAL' or token.value == '+' or token.value == '-'):
            skip_val = ''
        if last_token is not None and (last_token.value == '+' or last_token.value == '-') and token.type == 'TOK_INT_LITERAL':
            skip_val = ''
        code += skip_val
        code += val
        if token.lineno != -1:
            lastLine = token.lineno
        last_token = token
    if includes is not None:
        for inc in includes:
            code = inc + '\n' +code

    global convert_code_count
    if convert_code_count % 100 == 0:
        print(code)
    convert_code_count += 1
    return code


count = 0
def check_result(args):
    global count
    count += 1
    one, key_val = args

    current = multiprocessing.current_process()
    print('iteration {} in process {} {}: '.format(count, current.pid, current.name))
    file_name = 'test'+str(current.pid)+'.c'
    file_path = os.path.join(compile_file_path, file_name)

    beam_actions, identifier_list, output_length, tokens, ac_tokens, outputs, includes = init_one_records_with_include(one)
    if beam_actions is None:
        return [], [], -1, [], '', -1

    print('iteration {} after tokenize in Process {} {}'.format(count, current.pid, current.name))

    file_path_list = [file_path for i in range(len(beam_actions))]
    includes_list = [includes for i in range(len(beam_actions))]
    identifier_list_list = [identifier_list for i in range(len(beam_actions))]
    key_val_list = [key_val for i in range(len(beam_actions))]
    tokens_list = [tokens for i in range(len(beam_actions))]
    output_length_list = [output_length for i in range(len(beam_actions))]
    output_list = [outputs for i in range(len(beam_actions))]
    print('iteration {} before map beam action in Process {} {}'.format(count, current.pid, current.name))
    returns = map(check_beam_actions, zip(file_path_list, identifier_list_list, key_val_list, tokens_list, beam_actions, output_length_list, output_list, includes_list))
    res_list, res_code_list, res_action_list = list(zip(*returns))
    # res, res_code, res_action = check_beam_actions(file_path, identifier_list, key_val, output_length, tokens, beam_actions)
    print('iteration {} after map beam action in Process {} {}'.format(count, current.pid, current.name))

    res_list = [1 if res else 0 for res in res_list]
    final_res = 0
    final_res_code = ''
    final_res_action = []
    final_res_id = -1
    for i in range(len(res_list)):
        res = res_list[i]
        res_code = res_code_list[i]
        res_action = res_action_list[i]
        if res == 1:
            final_res = res
            final_res_code = res_code
            final_res_action = res_action
            final_res_id = i
            break
    print('iteration {} compile result: {} in Process {} {}'.format(count, final_res, current.pid, current.name))
    # if not res:
    #     print('code: ')
    #     print(code)
    #     print('ac_code: ')
    #     print(ac_code)
    #     print('res_code:')
    #     print(res_code)
    return res_code_list, res_list, final_res, final_res_action, final_res_code, final_res_id


def check_result_distance(args):
    global count
    count += 1
    one, key_val = args

    current = multiprocessing.current_process()
    print('iteration {} in process {} {}: '.format(count, current.pid, current.name))

    beam_actions, identifier_list, output_length, tokens, ac_tokens, outputs, includes = init_one_records_with_include(one)
    if beam_actions is None:
        return [], -1, -1
    distance = one['distance']

    print('iteration {} after tokenize in Process {} {}'.format(count, current.pid, current.name))

    identifier_list_list = [identifier_list for i in range(len(beam_actions))]
    key_val_list = [key_val for i in range(len(beam_actions))]
    tokens_list = [tokens for i in range(len(beam_actions))]
    ac_tokens_list = [ac_tokens for i in range(len(beam_actions))]
    output_length_list = [output_length for i in range(len(beam_actions))]
    print('iteration {} before map beam action in Process {} {}'.format(count, current.pid, current.name))
    distance_list = list(map(check_beam_actions_distance, zip(identifier_list_list, key_val_list, tokens_list, ac_tokens_list, beam_actions, output_length_list)))
    print('iteration {} after map beam action in Process {} {}'.format(count, current.pid, current.name))

    print(distance_list)
    # success_id = int(one['success_id'])
    # if success_id != -1:
    #     print('success id: {}, success distance: {}'.format(success_id, distance_list[success_id]))
    min_distance = -1
    min_distance_id = -1
    for i in range(len(distance_list)):
        cur_distance = distance_list[i]
        if min_distance == -1 or (min_distance != -1 and cur_distance < min_distance):
            min_distance = cur_distance
            min_distance_id = i
    print('iteration {} min distance: {} actual distance: {} in Process {} {}'.format(count, min_distance, distance, current.pid, current.name))
    return distance_list, min_distance, min_distance_id


def init_one_records_with_include(one, tokenize_fn=tokenize_by_clex_fn()):
    code = init_code(one['code'])
    ac_code = init_code(one['ac_code'])
    inputs = one['input_list']
    outputs = one['output_list']
    predicts = one['predict_list']
    # output_length = len(inputs[0])
    output_length = None
    beam_actions = [[position_onebeam, is_copy_onebeam, keyword_id_onebeam, copy_id_onebeam] for
                    position_onebeam, is_copy_onebeam, keyword_id_onebeam, copy_id_onebeam in zip(*predicts)]
    # print('beam actions: ', beam_actions)
    tokens = list(tokenize_fn(code))
    ac_tokens = list(tokenize_fn(ac_code))
    if tokens is None or ac_tokens is None:
        return None, None, None, None, None, None, None
    # for tok in tokens:
    #     print(tok)
    identifier_list = inputs[-1][0]
    includes = one['code_includes']
    return beam_actions, identifier_list, output_length, tokens, ac_tokens, outputs, includes


def metrics_actions(actions, outputs):
    if len(actions[0]) != len(outputs[0]):
        return 0
    print(actions, outputs)
    res_mask = metrics_output_directly(actions, outputs)
    res = 1
    for a_res in res_mask:
        if a_res == 0:
            res = 0
    return res


def check_beam_actions(args):
    file_path, identifier_list, key_val, tokens, actions, output_length, outputs, includes = args
    new_tokens = copy.copy(tokens)
    identifier_list = copy.copy(identifier_list)
    # res = metrics_actions(actions, outputs)
    actions = list(zip(*actions))
    # print('actions in one beam: {}'.format(actions))
    new_tokens = recovery_tokens(actions, new_tokens, identifier_list, key_val, output_length)
    res_code = convert_token_to_code(new_tokens, includes=includes)
    # res = util.compile_c_code_by_gcc(res_code, r'G:\Project\program_ai\test.cpp')
    res = util.compile_c_code_by_gcc(res_code, file_path)
    if res:
        print('')
    return res, res_code, actions


def check_beam_actions_distance(args):
    identifier_list, key_val, tokens, ac_tokens, actions, output_length = args
    new_tokens = copy.copy(tokens)
    identifier_list = copy.copy(identifier_list)
    actions = list(zip(*actions))
    # print('actions in one beam: {}'.format(actions))
    new_tokens = recovery_tokens(actions, new_tokens, identifier_list, key_val, output_length)
    distance = levenshtenin_distance(ac_tokens, new_tokens, equal_fn=equal_fn)[0]
    return distance


def save_check_result(test_df, experiment_name, local_db_path=local_c_test_experiment_db):
    test_df['res_code_list'] = test_df['res_code_list'].map(json.dumps)
    test_df['res_list'] = test_df['res_list'].map(json.dumps)
    test_df['final_res_action'] = test_df['final_res_action'].map(json.dumps)
    test_df['final_res'] = test_df['final_res'].map(str)
    test_df['final_res_id'] = test_df['final_res_id'].map(str)

    store_list = [[*one] for one in zip(test_df['res_code_list'], test_df['res_list'], test_df['final_res'],
                                        test_df['final_res_action'], test_df['final_res_code'],
                                        test_df['final_res_id'], test_df['id'])]

    run_sql_statment(local_db_path, TEST_EXPERIMENT_RECORDS, 'update_predict_result', store_list, replace_table_name=experiment_name)


def save_check_result_distance(test_df, experiment_name, local_db_path=local_c_test_experiment_db):
    test_df['distance_list'] = test_df['distance_list'].map(json.dumps)
    test_df['min_distance'] = test_df['min_distance'].map(str)
    test_df['min_distance_id'] = test_df['min_distance_id'].map(str)

    store_list = [[*one] for one in zip(test_df['distance_list'], test_df['min_distance'], test_df['min_distance_id'], test_df['id'])]
    run_sql_statment(local_db_path, TEST_EXPERIMENT_RECORDS, 'update_distance_result', store_list, replace_table_name=experiment_name)


compile_file_path = 'compile_path/'


def parallel_map_check_distance(test_dict, key_val_list):
    returns = util.parallel_map(core_num=core_num, f=check_result_distance, args=list(zip(test_dict, key_val_list)))
    # returns = map(check_result_distance, list(zip(test_dict, key_val_list)))
    distance_list, min_distance, min_distance_id = list(zip(*returns))
    return distance_list, min_distance, min_distance_id


def parallel_map_check_result(test_dict, key_val_list):
    returns = util.parallel_map(core_num=core_num, f=check_result, args=list(zip(test_dict, key_val_list)))
    # returns = map(check_result, list(zip(test_dict, key_val_list)))
    res_code_list, res_list, final_res, final_res_action, final_res_code, final_res_id = list(zip(*returns))
    return res_code_list, res_list, final_res, final_res_action, final_res_code, final_res_id


def calculate_result(test_df):
    test_df['result'] = test_df['result'].map(int)
    import numpy as np
    success = np.sum(test_df['result'])
    total = len(test_df)
    print('total {} code, recovery success {}'.format(total, success))


if __name__ == '__main__':
    key_val, char_voc = create_c_embedding()

    # experiment_name = 'c_final_iterative_model_using_common_error_without_iscontinue'
    experiment_name = 'deepfix_test_final_iterative_model_using_common_error_without_iscontinue'
    core_num = 8

    test_df = read_test_experiment_by_experiment_name(local_c_test_experiment_db, experiment_name)
    # calculate_result(test_df)

    test_df['code_includes'] = test_df['code'].map(extract_include)
    test_df['code'] = test_df['code'].map(replace_include_with_blank)

    test_df['input_list'] = test_df['input_list'].map(json.loads)
    test_df['output_list'] = test_df['output_list'].map(json.loads)
    test_df['predict_list'] = test_df['predict_list'].map(json.loads)

    test_dict = test_df.to_dict('records')
    key_val_list = [key_val for i in range(len(test_dict))]

    import time
    start1 = time.time()
    print('start1: {}'.format(start1))

    test_df['res_code_list'], test_df['res_list'], test_df['final_res'], test_df['final_res_action'], \
    test_df['final_res_code'], test_df['final_res_id'] = parallel_map_check_result(test_dict, key_val_list)
    save_check_result(test_df, experiment_name, local_c_test_experiment_db)

    # # test_df['distance_list'], test_df['min_distance'], test_df['min_distance_id'] = parallel_map_check_distance(test_dict, key_val_list)
    # # save_check_result_distance(test_df, experiment_name, local_c_test_experiment_db)

    end1 = time.time()
    print('end1: {}'.format(end1))

    # # start2 = time.time()
    # # print('start2: {}'.format(start2))
    # # test_df['result2'] = list(map(check_result, zip(test_dict, key_val_list)))
    # # # test_df['result'] = test_df.apply(check_result, key_val=key_val, raw=True, axis=1)
    # # end2 = time.time()
    # # print('end2: {}'.format(end2))

    print(test_df['final_res'])
    err = 0
    scc = 0
    total = 0
    for res in test_df['final_res']:
        total += 1
        if res == 1:
            scc += 1
        else:
            err += 1
    print('total {} code, recovery success {}, failed {}'.format(total, scc, err))
    print('parallel map time: {}'.format(end1-start1))


