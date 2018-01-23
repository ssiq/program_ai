import more_itertools

from code_data.constants import cache_data_path, pre_defined_cpp_token
from code_data.read_data import read_cpp_fake_code_records_set, read_cpp_random_token_code_records_set
from common import util
from common.code_tokenize import GetTokens
from common.new_tokenizer import tokenize
from common.beam_search_util import flat_list
from ply.lex import LexToken

MAX_TOKEN_LENGTH = 300
CHANGE = 0
INSERT = 1
DELETE = 2

@util.disk_cache(basename='random_token_code_load_data', directory=cache_data_path)
def load_data_token_level(max_bug_number=1, min_bug_number=0):
    train, test, vaild = read_cpp_random_token_code_records_set()

    key_val, char_voc = create_embedding()
    parse_xy_param = [key_val, char_voc, max_bug_number, min_bug_number]
    flat_train_data = parse_xy_token_level(train, 'flat_train', *parse_xy_param)
    train_data = parse_xy_token_level(train, 'train', *parse_xy_param, sample_size=50000)
    # train_data = get_part_of_train_data(train, parse_xy_param)
    test_data = parse_xy_token_level(test, 'test', *parse_xy_param)
    vaild_data = parse_xy_token_level(vaild, 'vaild', *parse_xy_param)
    # import gc
    # del train, test, vaild
    # gc.collect()
    # train = None
    # test = None
    # vaild = None
    # gc.collect()
    return flat_train_data, train_data, test_data, vaild_data


@util.disk_cache(basename='random_token_code_load_data_sample_5000', directory=cache_data_path)
def load_data_token_level_sample(max_bug_number=1, min_bug_number=0):
    train, test, vaild = sample_on_random_token_code_records()

    key_val, char_voc = create_embedding()
    parse_xy_param = [key_val, char_voc, max_bug_number, min_bug_number]
    flat_train_data = parse_xy_token_level(train, 'flat_train', *parse_xy_param)
    # train_data = parse_xy_token_level(train, 'train', *parse_xy_param)
    train_data = parse_xy_token_level(train, 'train', *parse_xy_param)
    test_data = parse_xy_token_level(test, 'test', *parse_xy_param)
    vaild_data = parse_xy_token_level(vaild, 'vaild', *parse_xy_param)
    return flat_train_data, train_data, test_data, vaild_data


@util.disk_cache(basename='random_token_code_load_data_without_iscontinue', directory=cache_data_path)
def load_data_token_level_without_iscontinue(max_bug_number=1, min_bug_number=0):
    train, test, vaild = read_cpp_random_token_code_records_set()

    parse_xy_fn = parse_xy_token_level_without_iscontinue
    key_val, char_voc = create_embedding()
    parse_xy_param = [key_val, char_voc, max_bug_number, min_bug_number, action_list_sorted]
    flat_train_data = parse_xy_fn(train, 'flat_train', *parse_xy_param)
    # train_data = parse_xy_fn(train, 'train', *parse_xy_param, sample_size=50000)
    # train_data = get_part_of_train_data(train, parse_xy_param)
    test_data = parse_xy_fn(test, 'test', *parse_xy_param)
    vaild_data = parse_xy_fn(vaild, 'vaild', *parse_xy_param)
    # import gc
    # del train, test, vaild
    # gc.collect()
    # train = None
    # test = None
    # vaild = None
    # gc.collect()
    return flat_train_data, test_data, vaild_data


@util.disk_cache(basename='random_token_code_load_data_without_iscontinue_sample_5000', directory=cache_data_path)
def load_data_token_level__without_iscontinue_sample(max_bug_number=1, min_bug_number=0):
    train, test, vaild = sample_on_random_token_code_records()

    parse_xy_fn = parse_xy_token_level_without_iscontinue
    key_val, char_voc = create_embedding()
    parse_xy_param = [key_val, char_voc, max_bug_number, min_bug_number, action_list_sorted]
    flat_train_data = parse_xy_fn(train, 'flat_train', *parse_xy_param)
    # train_data = parse_xy_token_level(train, 'train', *parse_xy_param)
    # train_data = parse_xy_fn(train, 'train', *parse_xy_param)
    test_data = parse_xy_fn(test, 'test', *parse_xy_param)
    vaild_data = parse_xy_fn(vaild, 'vaild', *parse_xy_param)
    return flat_train_data, test_data, vaild_data
    

@util.disk_cache(basename='random_get_part_of_train_data_50000', directory=cache_data_path)
def get_part_of_train_data(train, parse_xy_param):
    train_data = parse_xy_token_level(train, 'train', *parse_xy_param, sample_size=50000)
    return train_data


# ---------------------- sample ---------------------------#

@util.disk_cache(basename='token_level_multirnn_on_fake_cpp_data_sample_5000', directory=cache_data_path)
def sample():
    train, test, vaild = read_cpp_fake_code_records_set()
    train = train.sample(5000, random_state=1)
    test = test.sample(5000, random_state=1)
    vaild = vaild.sample(5000, random_state=1)
    return (train, test, vaild)


@util.disk_cache(basename='random_token_code_token_level_multirnn_records_sample_5000', directory=cache_data_path)
def sample_on_random_token_code_records():
    train, test, vaild = read_cpp_random_token_code_records_set()
    train = train.sample(5000, random_state=1)
    test = test.sample(5000, random_state=1)
    vaild = vaild.sample(5000, random_state=1)
    return (train, test, vaild)


def create_embedding():
    from code_data.constants import char_sign_dict
    from embedding.wordembedding import load_vocabulary
    from embedding.character_embedding import load_character_vocabulary
    key_val = load_vocabulary('keyword', embedding_size=200)
    char_voc = load_character_vocabulary('bigru', n_gram=1, embedding_shape=100, token_list=char_sign_dict.keys())
    return key_val, char_voc


def error_count_create_condition_fn(error_count: int):
    def condition_fn(one):
        for x in one:
            if len(x) > error_count:
                return False
        return True
    return condition_fn

def no_condition_create_fn():
    def condition_fn(one):
        return True
    return condition_fn

def error_count_without_train_condition_fn(data_type, error_count:int):
    def condition_fn(one):
        for x in one:
            if len(x) > error_count:
                return False
        return True
    def train_condition_fn(one):
        return True

    if data_type == 'flat_train':
        return train_condition_fn
    return condition_fn


# -------------------------------- parse_xy method -------------------------------- #

@util.disk_cache(basename='identifier_mask_token_level_multirnn_on_fake_cpp_data_parse_xy', directory=cache_data_path)
def parse_xy_with_identifier_mask(df, data_type:str, keyword_voc, char_voc, max_bug_number=1, min_bug_number=0):

    df['res'] = ''
    df['ac_code_obj'] = df['ac_code'].map(get_token_list)
    df = df[df['ac_code_obj'].map(lambda x: x is not None)].copy()

    df = df.apply(create_error_list, axis=1, raw=True)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_token_id_input, axis=1, raw=True, keyword_voc=keyword_voc)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_token_identify_mask, axis=1, raw=True, pre_defined_token_set=pre_defined_cpp_token)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_character_id_input, axis=1, raw=True, char_voc=char_voc)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_full_output, axis=1, raw=True, keyword_voc=keyword_voc, max_bug_number=max_bug_number, min_bug_number=min_bug_number, find_copy_id_fn=find_copy_id_by_identifier_dict)
    # df = df.apply(create_full_output, axis=1, raw=True, keyword_voc=keyword_voc, max_bug_number=max_bug_number, min_bug_number=min_bug_number, find_copy_id_fn=find_token_name)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    returns = (df['token_id_list'], df['token_length_list'], df['character_id_list'], df['character_length_list'], df['token_identify_mask'], df['output_length'], df['position_list'], df['is_copy_list'], df['keywordid_list'], df['copyid_list'])

    # if data_type == 'train':
    #     returns = [flat_list(ret) for ret in returns]

    return returns


@util.disk_cache(basename='token_level_multirnn_on_fake_cpp_data_parse_xy', directory=cache_data_path)
def parse_xy(df, data_type:str, keyword_voc, char_voc, max_bug_number=1, min_bug_number=0):

    df['res'] = ''
    df['ac_code_obj'] = df['ac_code'].map(get_token_list)
    df = df[df['ac_code_obj'].map(lambda x: x is not None)].copy()

    df = df.apply(create_error_list, axis=1, raw=True)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_token_id_input, axis=1, raw=True, keyword_voc=keyword_voc)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_character_id_input, axis=1, raw=True, char_voc=char_voc)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_full_output, axis=1, raw=True, keyword_voc=keyword_voc, max_bug_number=max_bug_number, min_bug_number=min_bug_number, find_copy_id_fn=find_token_name)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    return df['token_id_list'], df['token_length_list'], df['character_id_list'], df['character_length_list'], df['output_length'], df['position_list'], df['is_copy_list'], df['keywordid_list'], df['copyid_list']


@util.disk_cache(basename='random_token_action_multirnn_on_fake_cpp_data_parse_xy', directory=cache_data_path)
def parse_xy_token_level(df, data_type:str, keyword_voc, char_voc, max_bug_number=1, min_bug_number=0, sample_size=None):

    df['res'] = ''
    df['ac_code_obj'] = df['ac_code'].map(do_new_tokenize)
    df = df[df['ac_code_obj'].map(lambda x: x is not None)].copy()

    df = df.apply(create_error_list_by_token_actionmap, axis=1, raw=True)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_token_id_input, axis=1, raw=True, keyword_voc=keyword_voc)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_token_identify_mask, axis=1, raw=True, pre_defined_token_set=pre_defined_cpp_token)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_character_id_input, axis=1, raw=True, char_voc=char_voc)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_full_output, axis=1, raw=True, keyword_voc=keyword_voc, max_bug_number=max_bug_number,
                  min_bug_number=min_bug_number, find_copy_id_fn=find_copy_id_by_identifier_dict)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    if sample_size is not None:
        df = df.sample(sample_size)

    returns = (df['token_id_list'], df['token_length_list'], df['character_id_list'], df['character_length_list'],
               df['token_identify_mask'], df['output_length'], df['position_list'], df['is_copy_list'],
               df['keywordid_list'], df['copyid_list'])

    if data_type == 'flat_train':
        returns = [flat_list(ret) for ret in returns]

    return returns


@util.disk_cache(basename='random_token_action_multirnn_on_fake_cpp_data_parse_xy_without_iscontinue', directory=cache_data_path)
def parse_xy_token_level_without_iscontinue(df, data_type: str, keyword_voc, char_voc, max_bug_number=1, min_bug_number=0, sort_fn=None, sample_size=None):
    df['res'] = ''
    df['ac_code_obj'] = df['ac_code'].map(do_new_tokenize)
    df = df[df['ac_code_obj'].map(lambda x: x is not None)].copy()

    df = df.apply(create_error_list_by_token_actionmap, axis=1, raw=True, sort_fn=sort_fn)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_token_id_input, axis=1, raw=True, keyword_voc=keyword_voc)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_token_identify_mask, axis=1, raw=True, pre_defined_token_set=pre_defined_cpp_token)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_character_id_input, axis=1, raw=True, char_voc=char_voc)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_full_output, axis=1, raw=True, keyword_voc=keyword_voc, max_bug_number=max_bug_number,
                  min_bug_number=min_bug_number, find_copy_id_fn=find_copy_id_by_identifier_dict)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    if sample_size is not None:
        df = df.sample(sample_size)

    returns = (df['token_id_list'], df['token_length_list'], df['character_id_list'], df['character_length_list'],
               df['token_identify_mask'], df['position_list'], df['is_copy_list'],
               df['keywordid_list'], df['copyid_list'])

    if data_type == 'flat_train':
        returns = [flat_list(ret) for ret in returns]

    return returns

# -------------------------------- parse_xy util method -------------------------------- #

def create_error_list(one):
    import random

    def has_repeat_action(code_obj, ac_cha_list):
        start_ac_pos = []
        for act in ac_cha_list:
            ac_pos = act['ac_pos']
            for obj in code_obj:
                if ac_pos >= obj.start and ac_pos < obj.end:
                    if ac_pos in start_ac_pos:
                        return True
                    start_ac_pos.append(ac_pos)
                    break
        return False

    import json
    action_character_list = json.loads(one['action_character_list'])
    ac_code_obj = one['ac_code_obj']

    if has_repeat_action(ac_code_obj, action_character_list):
        one['res'] = None
        return one

    token_name_list = []
    action_list = []
    for act in action_character_list:
        ac_pos = act['ac_pos']
        ac_type = act['act_type']
        i = 0
        while i < len(ac_code_obj):
            obj = ac_code_obj[i]

            if ac_pos >= obj.start and ac_pos < obj.end:
                if ac_type == INSERT:
                    import copy
                    tmp_obj = ac_code_obj[random.randint(0, len(ac_code_obj)-1)]
                    tmp_obj = copy.deepcopy(tmp_obj)
                    tmp_obj.start = -1
                    tmp_obj.end = -1
                    action = {'type': DELETE, 'pos': i * 2 + 1, 'token': tmp_obj.name}
                    ac_code_obj.insert(i, tmp_obj)
                    name_list = create_name_list(ac_code_obj)
                    action_list.insert(0, action)
                    token_name_list.insert(0, name_list)
                    break
                elif ac_type == DELETE:
                    ac_code_obj.pop(i)
                    action = {'type': INSERT, 'pos': i*2, 'token': obj.name}
                    name_list = create_name_list(ac_code_obj)
                    action_list.insert(0, action)
                    token_name_list.insert(0, name_list)
                    break
                elif ac_type == CHANGE:
                    name = obj.name
                    tmp_list = list(name)
                    tmp_list[ac_pos-obj.start] = act['to_char']
                    ac_code_obj[i].name = "".join(tmp_list)

                    action = {'type': CHANGE, 'pos': i*2+1, 'token': name}
                    name_list = create_name_list(ac_code_obj)
                    action_list.insert(0, action)
                    token_name_list.insert(0, name_list)
                    break
            i += 1

    if len(action_list) == 0:
        one['res'] = None
        return one
    one['token_name_list'] = token_name_list
    one['action_list'] = action_list
    one['copy_name_list'] = token_name_list
    return one


def create_error_list_by_token_actionmap(one, sort_fn=None):
    import json
    ac_code_obj = one['ac_code_obj']
    action_token_list = json.loads(one['action_character_list'])
    if sort_fn is not None:
        action_token_list = sort_fn(action_token_list)

    def cal_token_pos_bias(action_list, cur_action):
        bias = 0
        cur_token_pos = cur_action['token_pos']
        for act in action_list:
            if act['act_type'] == INSERT and (cur_action['act_type'] == DELETE or cur_action['act_type'] == CHANGE) and cur_token_pos >= act['token_pos']:
                bias += 1
            elif act['act_type'] == INSERT and cur_token_pos > act['token_pos']:
                bias += 1
            elif act['act_type'] == DELETE and cur_token_pos > act['token_pos']:
                bias -= 1
        return bias

    token_pos_list = [act['token_pos'] for act in action_token_list]
    has_repeat_action_fn = lambda x: len(set(x)) < len(x)
    if has_repeat_action_fn(token_pos_list):
        one['res'] = None
        return one

    token_bias_list = [cal_token_pos_bias(action_token_list[0:i], action_token_list[i]) for i in range(len(action_token_list))]

    token_name_list = []
    action_list = []
    for act, token_bias in zip(action_token_list, token_bias_list):
        # ac_pos = act['ac_pos']
        ac_type = act['act_type']
        ac_token_pos = act['token_pos']
        real_token_pos = ac_token_pos + token_bias

        if ac_type == INSERT:
            to_char = act['to_char']
            tok = LexToken()
            tok.value = to_char
            tok.lineno = -1
            tok.type = ""
            tok.lexpos = -1
            ac_code_obj = ac_code_obj[0:real_token_pos] + [tok] + ac_code_obj[real_token_pos:]
            action = {'type': DELETE, 'pos': real_token_pos * 2 + 1, 'token': to_char}
            name_list = create_name_list_by_LexToken(ac_code_obj)
            token_name_list = [name_list] + token_name_list
            action_list = [action] + action_list
        elif ac_type == DELETE:
            from_char = act['from_char']
            ac_code_obj = ac_code_obj[0: real_token_pos] + ac_code_obj[real_token_pos+1:]
            action = {'type': INSERT, 'pos': real_token_pos * 2, 'token': from_char}
            name_list = create_name_list_by_LexToken(ac_code_obj)
            token_name_list = [name_list] + token_name_list
            action_list = [action] + action_list
        elif ac_type == CHANGE:
            from_char = act['from_char']
            to_char = act['to_char']
            tok = LexToken()
            tok.value = to_char
            tok.lineno = -1
            tok.type = ""
            tok.lexpos = -1
            action = {'type': CHANGE, 'pos': real_token_pos * 2 + 1, 'token': from_char}
            ac_code_obj = ac_code_obj[0: real_token_pos] + [tok] +ac_code_obj[real_token_pos + 1:]
            name_list = create_name_list_by_LexToken(ac_code_obj)
            token_name_list = [name_list] + token_name_list
            action_list = [action] + action_list

    if len(action_list) == 0:
        one['res'] = None
        return one
    one['token_name_list'] = token_name_list
    one['action_list'] = action_list
    one['copy_name_list'] = token_name_list
    return one


def create_name_list(code_obj):
    name_list = []
    for obj in code_obj:
        name_list.append(obj.name)
    return name_list


def create_name_list_by_LexToken(code_obj_list):
    name_list = [''.join(obj.value) if isinstance(obj.value, list) else obj.value for obj in code_obj_list]
    return name_list


def create_identifier_mask_from_dict(name_list, iden_dict:dict, pre_defined_cpp_token):
    iden_list = []
    for token in name_list:
        iden = [0 for i in range(len(iden_dict.keys()))]
        if token not in iden_dict.keys() and token not in pre_defined_cpp_token:
            return None
        elif token in iden_dict.keys():
            iden[iden_dict[token]-1] = 1
        iden_list.append(iden)
    return iden_list


def create_token_identify_mask(one, pre_defined_token_set=pre_defined_cpp_token):
    token_name_list = one['token_name_list']
    _, iden_dict = create_identifier_mask(token_name_list[0], pre_defined_token_set)
    if len(iden_dict.keys()) > 30:
        one['res'] = None
        return one
    token_identify_mask = [create_identifier_mask_from_dict(name_list, iden_dict, pre_defined_token_set) for name_list in token_name_list]
    for one_identify in token_identify_mask:
        if one_identify == None:
            one['res'] = None
            return one
    one['token_identify_mask'] = token_identify_mask
    one['copy_name_list'] = [iden_dict for i in range(len(token_identify_mask))]
    return one


def create_token_id_input(one, keyword_voc):
    token_name_list = one['token_name_list']
    token_id_list = []
    len_list = []
    for name_list in token_name_list:
        id_list = keyword_voc.parse_text_without_pad([name_list], False)[0]
        if id_list == None:
            one['res'] = None
            return one
        len_list.append(len(id_list))
        token_id_list.append(id_list)
    one['token_id_list'] = token_id_list
    one['token_length_list'] = len_list
    return one


def create_character_id_input(one, char_voc):
    token_name_list = one['token_name_list']
    char_id_list = []
    len_list = []
    for name_list in token_name_list:
        one_len = []
        id_list = char_voc.parse_string_without_padding([name_list], token_position_label=False, character_position_label=True)[0]
        if id_list == None:
            one['res'] = None
            return one
        char_id_list.append(id_list)
        for i in range(len(id_list)):
            id_l = id_list[i]
            one_len.append(len(id_l))
        len_list.append(one_len)

    char_col = list(more_itertools.collapse(char_id_list, levels=1))
    for tok in char_col:
        if len(tok) > 30:
            print('character len is {}. more than 30'.format(len(tok)))
            one['res'] = None
            return one

    one['character_id_list'] = char_id_list
    one['character_length_list'] = len_list
    return one


def find_copy_id_by_identifier_dict(name, iden_token_id_dict:dict):
    if name in iden_token_id_dict.keys():
        copy_id = iden_token_id_dict[name]-1
        return copy_id
    return -1


def find_token_name(name, token_name_list):
    for k in range(len(token_name_list)):
        token = token_name_list[k]
        if name == token:
            return k
    return -1


def create_full_output(one, keyword_voc, max_bug_number, min_bug_number, find_copy_id_fn):
    action_list = one['action_list']
    # token_name_list = one['token_name_list']
    copy_name_list = one['copy_name_list']
    # identifier_mask = one['token_identify_mask']

    position_list = []
    is_copy_list = []
    keywordid_list = []
    copyid_list = []

    for i in range(len(action_list)):
        action = action_list[i]
        act_type = action['type']
        act_pos = action['pos']
        act_token = action['token']


        if act_type == INSERT or act_type == CHANGE:
            act_id = keyword_voc.word_to_id(act_token)
            if act_id != keyword_voc.word_to_id(keyword_voc.identifier_label) and act_id != keyword_voc.word_to_id(keyword_voc.placeholder_label):
                is_copy_list.append(0)
                keywordid_list.append(keyword_voc.word_to_id(act_token))
                copyid_list.append(0)
            else:
                # name_list = token_name_list[i]
                name_list = copy_name_list[i]
                copy_pos = find_copy_id_fn(act_token, name_list)
                if copy_pos >= 0:
                    is_copy_list.append(1)
                    keywordid_list.append(0)
                    copyid_list.append(copy_pos)
                else:
                    one['res'] = None
                    return one
        elif act_type == DELETE:
            is_copy_list.append(0)
            keywordid_list.append(keyword_voc.word_to_id(keyword_voc.placeholder_label))
            copyid_list.append(0)
        position_list.append(act_pos)

    if len(position_list) == 0 or len(position_list) > max_bug_number or len(position_list) < min_bug_number:
        one['res'] = None
        return one
    one['position_list'] = position_list
    one['is_copy_list'] = is_copy_list
    one['keywordid_list'] = keywordid_list
    one['copyid_list'] = copyid_list

    one['output_length'] = [1 for i in range(len(position_list)-1)] + [0]
    return one


def get_token_list(code):
    try:
        code = code.replace('\r', '')
        if code.find('define') != -1 or code.find('defined') != -1 or code.find('undef') != -1 or \
                        code.find('pragma') != -1 or code.find('ifndef') != -1 or \
                        code.find('ifdef') != -1 or code.find('endif') != -1:
            return None
        tokens = list(GetTokens(code))
        if len(tokens) >= MAX_TOKEN_LENGTH:
            return None
        if None in tokens:
            return None
    except RuntimeError as e:
        print('in runtime error')
        return None
    return tokens


def do_new_tokenize(code):
    try:
        code = code.replace('\r', '')
        if code.find('define') != -1 or code.find('defined') != -1 or code.find('undef') != -1 or \
                        code.find('pragma') != -1 or code.find('ifndef') != -1 or \
                        code.find('ifdef') != -1 or code.find('endif') != -1:
            return None
        code_tokens = tokenize(code)
        if len(code_tokens) >= MAX_TOKEN_LENGTH:
            return None
        if None in code_tokens:
            return None
    except RuntimeError as e:
        print('in runtime error')
        return None
    return code_tokens



def create_identifier_mask(tokens, keyword_set):
    token_set = set(tokens) - keyword_set
    token_set = sorted(token_set)
    id_token_dict = dict(enumerate(token_set, start=1))
    token_id_dict = util.reverse_dict(id_token_dict)
    def f(x):
         return [int(x==t)  for t in tokens]
    # res = list(util.parallel_map(core_num=10, f=f, args=token_set))
    res = list(map(f, token_set))
    res = list(map(list, zip(*res)))
    return res, token_id_dict


def create_identifier_category(tokens, keyword_set):
    token_set = set(tokens) - keyword_set
    token_id_dict = util.reverse_dict(dict(enumerate(token_set, start=1)))
    return [token_id_dict[t] if t in token_set else 0 for t in tokens], token_id_dict


def action_list_sorted(action_list):
    def sort_key(a):
        bias = 0.5 if a['act_type'] == INSERT else 0
        return a['token_pos'] - bias

    action_list = sorted(action_list, key=sort_key, reverse=True)
    return action_list
