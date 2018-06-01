from code_data.constants import cache_data_path, pre_defined_c_tokens
from code_data.read_c_data import read_fake_common_c_error_dataset, read_fake_common_c_error_dataset_with_limit_length
from common import util
from common.analyse_include_util import replace_include_with_blank, extract_include
from common.beam_search_util import flat_list
from common.c_code_tokenize import tokenize_by_clex_fn
from experiment.experiment_util import create_error_list_by_token_actionmap, action_list_sorted, create_token_id_input, \
    create_token_identify_mask, create_character_id_input, create_full_output, find_copy_id_by_identifier_dict, \
    find_pos_by_identifier_mask, create_name_list_by_LexToken

import json



MAX_TOKEN_LENGTH = 500
CHANGE = 0
INSERT = 1
DELETE = 2

def create_c_embedding():
    from code_data.constants import char_sign_dict
    from embedding.wordembedding import load_vocabulary
    from embedding.character_embedding import load_character_vocabulary
    key_val = load_vocabulary('c_keyword', embedding_size=200)
    char_voc = load_character_vocabulary('bigru', n_gram=1, embedding_shape=100, token_list=char_sign_dict.keys())
    return key_val, char_voc


def convert_action_map_to_old_action(actions):

    def convert_one_action(one_action):
        if one_action['act_type'] == 4:
            one_action['act_type'] = CHANGE
        elif one_action['act_type'] == 3:
            one_action['act_type'] = DELETE
        elif one_action['act_type'] == 1:
            one_action['act_type'] = INSERT
        else:
            print('error_one_action', one_action)
            return None
        return one_action

    new_actions_obj = json.loads(actions)
    old_actions_obj = [convert_one_action(one_action) for one_action in new_actions_obj]
    old_actions_obj = list(filter(lambda x: x is not None, old_actions_obj))
    old_actions = json.dumps(old_actions_obj)
    return old_actions


def convert_c_code_fields_to_cpp_fields(df):
    filter_macro_fn = lambda code: not (code.find('define') != -1 or code.find('defined') != -1 or
                                        code.find('undef') != -1 or code.find('pragma') != -1 or
                                        code.find('ifndef') != -1 or code.find('ifdef') != -1 or
                                        code.find('endif') != -1)
    df['action_character_list'] = df['modify_action_list'].map(convert_action_map_to_old_action)
    df['code_includes'] = df['similar_code'].map(extract_include)
    df['similar_code_without_include'] = df['similar_code'].map(replace_include_with_blank)
    df['ac_code'] = df['similar_code_without_include']
    df = df[df['similar_code'].map(filter_macro_fn)]
    df['error_count'] = df['distance']
    return df


@util.disk_cache(basename='load_c_code_data_common_error_token_level_without_iscontinue', directory=cache_data_path)
def load_c_code_data_common_error_token_level_without_iscontinue(max_bug_number=1, min_bug_number=0):
    train, vaild, test = read_fake_common_c_error_dataset_with_limit_length(MAX_TOKEN_LENGTH)

    train = convert_c_code_fields_to_cpp_fields(train)
    test = convert_c_code_fields_to_cpp_fields(test)
    vaild = convert_c_code_fields_to_cpp_fields(vaild)

    tokenize_fn = tokenize_by_clex_fn()

    parse_xy_fn = parse_xy_c_code_token_level_without_iscontinue
    key_val, char_voc = create_c_embedding()
    parse_xy_param = [key_val, char_voc, max_bug_number, min_bug_number, action_list_sorted, tokenize_fn]
    flat_train_data = parse_xy_fn(train, 'flat_train', *parse_xy_param)
    # train_data = parse_xy_fn(train, 'train', *parse_xy_param, sample_size=50000)
    # train_data = get_part_of_train_data(train, parse_xy_param)
    test_data = parse_xy_fn(test, 'test', *parse_xy_param)
    vaild_data = parse_xy_fn(vaild, 'vaild', *parse_xy_param)
    return flat_train_data, test_data, vaild_data


# @util.disk_cache(basename='load_c_code_data_common_error_token_level_without_iscontinue_sample_500', directory=cache_data_path)
def load_c_code_data_common_error_token_level_without_iscontinue_sample(max_bug_number=1, min_bug_number=0):
    dfs = read_fake_common_c_error_dataset_with_limit_length(MAX_TOKEN_LENGTH)
    train, test, vaild = [df.sample(500) for df in dfs]

    train = convert_c_code_fields_to_cpp_fields(train)
    test = convert_c_code_fields_to_cpp_fields(test)
    vaild = convert_c_code_fields_to_cpp_fields(vaild)

    tokenize_fn = tokenize_by_clex_fn()

    parse_xy_fn = parse_xy_c_code_token_level_without_iscontinue
    key_val, char_voc = create_c_embedding()
    parse_xy_param = [key_val, char_voc, max_bug_number, min_bug_number, action_list_sorted, tokenize_fn]
    flat_train_data = parse_xy_fn(train, 'flat_train', *parse_xy_param)
    # train_data = parse_xy_fn(train, 'train', *parse_xy_param, sample_size=50000)
    # train_data = get_part_of_train_data(train, parse_xy_param)
    test_data = parse_xy_fn(test, 'test', *parse_xy_param)
    vaild_data = parse_xy_fn(vaild, 'vaild', *parse_xy_param)
    return flat_train_data, test_data, vaild_data


@util.disk_cache(basename='parse_xy_c_code_token_level_without_iscontinue', directory=cache_data_path)
def parse_xy_c_code_token_level_without_iscontinue(df, data_type: str, keyword_voc, char_voc, max_bug_number=1, min_bug_number=0, sort_fn=None, tokenize_fn=None, only_input=False, sample_size=None):
    print('start :', len(df.index))

    df['res'] = ''
    df['ac_code_obj'] = df['ac_code'].map(tokenize_fn)
    df = df[df['ac_code_obj'].map(lambda x: x is not None)].copy()
    df['ac_code_obj'] = df['ac_code_obj'].map(list)
    print('after tokenize: ', len(df.index))

    df = df.apply(create_error_list_by_token_actionmap, axis=1, raw=True, sort_fn=sort_fn)
    df = df[df['res'].map(lambda x: x is not None)].copy()
    print('after create error: ', len(df.index))

    df = df.apply(create_token_id_input, axis=1, raw=True, keyword_voc=keyword_voc)
    df = df[df['res'].map(lambda x: x is not None)].copy()
    print('after create token id: ', len(df.index))

    df = df.apply(create_token_identify_mask, axis=1, raw=True, pre_defined_token_set=pre_defined_c_tokens)
    df = df[df['res'].map(lambda x: x is not None)].copy()
    print('after create identify mask: ', len(df.index))

    df = df.apply(create_character_id_input, axis=1, raw=True, char_voc=char_voc)
    df = df[df['res'].map(lambda x: x is not None)].copy()
    print('after create charactid: ', len(df.index))

    df = df.apply(create_full_output, axis=1, raw=True, keyword_voc=keyword_voc, max_bug_number=max_bug_number,
                  min_bug_number=min_bug_number, find_copy_id_fn=find_copy_id_by_identifier_dict, find_identifier_mask_fn=find_pos_by_identifier_mask)
    df = df[df['res'].map(lambda x: x is not None)].copy()
    print('after create output: ', len(df.index))

    if sample_size is not None:
        df = df.sample(sample_size)

    returns = (df['token_id_list'], df['token_length_list'], df['character_id_list'], df['character_length_list'],
               df['token_identify_mask'], df['position_list'], df['is_copy_list'],
               df['keywordid_list'], df['copyid_list'])

    if data_type == 'flat_train':
        returns = [flat_list(ret) for ret in returns]

    return returns


@util.disk_cache(basename='parse_xy_c_code_token_level_without_iscontinue_only_input', directory=cache_data_path)
def parse_xy_c_code_token_level_without_iscontinue_only_input(df, data_type: str, keyword_voc, char_voc, tokenize_fn=None, sample_size=None):
    print('start :', len(df.index))

    df['res'] = ''
    df['ac_code_obj'] = df['ac_code'].map(tokenize_fn)
    df = df[df['ac_code_obj'].map(lambda x: x is not None)].copy()
    df['ac_code_obj'] = df['ac_code_obj'].map(list)
    print('after tokenize: ', len(df.index))

    create_name_list_fn = lambda x: [create_name_list_by_LexToken(x)]
    df['token_name_list'] = df['ac_code_obj'].map(create_name_list_fn)
    df['copy_name_list'] = df['token_name_list']

    df = df.apply(create_token_id_input, axis=1, raw=True, keyword_voc=keyword_voc)
    df = df[df['res'].map(lambda x: x is not None)].copy()
    print('after create token id: ', len(df.index))

    df = df.apply(create_token_identify_mask, axis=1, raw=True, pre_defined_token_set=pre_defined_c_tokens)
    df = df[df['res'].map(lambda x: x is not None)].copy()
    print('after create identify mask: ', len(df.index))

    df = df.apply(create_character_id_input, axis=1, raw=True, char_voc=char_voc)
    df = df[df['res'].map(lambda x: x is not None)].copy()
    print('after create charactid: ', len(df.index))

    if sample_size is not None:
        df = df.sample(sample_size)

    returns = (df['token_id_list'], df['token_length_list'], df['character_id_list'], df['character_length_list'],
               df['token_identify_mask'])

    if data_type == 'flat_train':
        returns = [flat_list(ret) for ret in returns]

    return returns