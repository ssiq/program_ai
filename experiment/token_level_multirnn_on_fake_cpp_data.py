from common import util
from code_data.read_data import read_cpp_fake_code_records_set
from common.code_tokenize import GetTokens
from common.supervision_util_increment import create_supervision_experiment
from model.token_level_multirnn_model import TokenLevelMultiRnnModel
from train.random_search import random_parameters_generator
from code_data.constants import cache_data_path
import more_itertools
from common.util import initCustomerLogger
import logging
from code_data.constants import DEBUG_LOG_PATH, OUTPUT_LOG_PATH, debug_logger_name_list, output_logger_name_list

MAX_TOKEN_LENGTH = 300

def create_identifier_category(tokens, keyword_set):
    token_set = set(tokens) - keyword_set
    token_id_dict = util.reverse_dict(dict(enumerate(token_set, start=1)))
    return [token_id_dict[t] if t in token_set else 0 for t in tokens], token_id_dict


def create_identifier_mask(tokens, keyword_set):
    token_set = set(tokens) - keyword_set
    token_set = sorted(token_set)
    id_token_dict = dict(enumerate(token_set, start=1))
    token_id_dict = util.reverse_dict(id_token_dict)
    def f(x):
         return [int(x==t)  for t in tokens]
    res = util.parallel_map(core_num=10, f=f, args=token_set)
    return res, token_id_dict


def get_token_list(code):
    try:
        code = code.replace('\r', '')
        if code.find('define') != -1 or code.find('param') != -1 or code.find('ifndef') != -1 or code.find(
                'endif') != -1:
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


@util.disk_cache(basename='token_level_multirnn_on_fake_cpp_data_parse_xy', directory=cache_data_path)
def parse_xy(df, keyword_voc, char_voc, max_bug_number=1, min_bug_number=0):

    def create_full_output(one):
        action_list = one['action_list']
        token_name_list = one['token_name_list']

        position_list = []
        is_copy_list = []
        keywordid_list = []
        copyid_list = []

        def find_token_name(name, token_name_list):
            for k in range(len(token_name_list)):
                token = token_name_list[k]
                if name == token:
                    return k
            return -1

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
                    name_list = token_name_list[i]
                    copy_pos = find_token_name(act_token, name_list)
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

        one['output_length'] = [1] * (len(position_list)-1) + [0]
        return one

    def create_character_id_input(one):
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



    def create_token_id_input(one):
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

        def create_name_list(code_obj):
            name_list = []
            for obj in code_obj:
                name_list.append(obj.name)
            return name_list


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

        one['token_name_list'] = token_name_list
        one['action_list'] = action_list
        return one

    CHANGE = 0
    INSERT = 1
    DELETE = 2

    df['res'] = ''
    df['ac_code_obj'] = df['ac_code'].map(get_token_list)
    df = df[df['ac_code_obj'].map(lambda x: x is not None)].copy()

    df = df.apply(create_error_list, axis=1, raw=True)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_token_id_input, axis=1, raw=True)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_character_id_input, axis=1, raw=True)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_full_output, axis=1, raw=True)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    return df['token_id_list'], df['token_length_list'], df['character_id_list'], df['character_length_list'], df['output_length'], df['position_list'], df['is_copy_list'], df['keywordid_list'], df['copyid_list']


@util.disk_cache(basename='token_level_multirnn_on_fake_cpp_data_sample_5000', directory=cache_data_path)
def sample():
    train, test, vaild = read_cpp_fake_code_records_set()
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


def create_condition_fn(error_count: int):
    def condition_fn(one):
        for x in one:
            if len(x) > error_count:
                return False
        return True
    return condition_fn


if __name__ == '__main__':
    util.initLogging()
    util.set_cuda_devices(1)
    train, test, vaild = read_cpp_fake_code_records_set()
    # train, test, vaild = sample()
    # train = train.sample(300000)

    key_val, char_voc = create_embedding()
    parse_xy_param = [key_val, char_voc, 5, 1]


    # print(train)

    # res = parse_xy(train, *parse_xy_param)
    # print(res)
    # print(len(res))

    # test_data_iterator = util.batch_holder(*parse_xy(test, *parse_xy_param), batch_size=8)
    #
    # isPrint = 0
    # for i in test_data_iterator():
    #     if isPrint < 1:
    #         for t in i:
    #             x = np.array(t)
    #             print(x.shape)
    #     isPrint += 1

    modify_condition = [
                        ({'error_count': 1}, 0.6),
                        ({'error_count': 2}, 0.5),
                        ({'error_count': 3}, 0.4),
                        ({'error_count': 4}, 0.4),
                        ({'error_count': 5}, 1.0), ]

    MAX_ITERATOR_LEGNTH = 5

    # train_supervision = create_supervision_experiment(train, test, vaild, parse_xy, parse_xy_param, experiment_name='token_level_multirnn_model', batch_size=16)

    train_supervision = create_supervision_experiment(train, test, vaild, parse_xy, parse_xy_param, experiment_name='token_level_multirnn_model', batch_size=16, create_condition_fn=create_condition_fn, modify_condition=modify_condition)
    param_generator = random_parameters_generator(random_param={"learning_rate": [-4, -1]},
                                                  choice_param={ },
                                                  constant_param={"hidden_size": 100,
                                                                  'rnn_layer_number': 2,
                                                                  'keyword_number': len(key_val.word_id_map),
                                                                  # 'start_id': key_val.word_to_id(key_val.start_label),
                                                                  'end_token_id': key_val.word_to_id(key_val.end_label),
                                                                  'max_decode_iterator_num': MAX_ITERATOR_LEGNTH,
                                                                  'identifier_token': key_val.word_to_id(key_val.identifier_label),
                                                                  'placeholder_token': key_val.word_to_id(key_val.placeholder_label),
                                                                  'word_embedding_layer_fn': key_val.create_embedding_layer,
                                                                  'character_embedding_layer_fn': char_voc.create_embedding_layer,
                                                                  'id_to_word_fn': key_val.id_to_word,
                                                                  'parse_token_fn': char_voc.parse_token})

    train_supervision(TokenLevelMultiRnnModel, param_generator, 1, restore=False)
    # restore_param_generator = random_parameters_generator(random_param={ },
    #                                               choice_param={ },
    #                                               constant_param={"learning_rate": 0.000101561577686,
    #                                                               "hidden_size": 100,
    #                                                               'rnn_layer_number': 2,
    #                                                               'keyword_number': len(key_val.word_id_map),
    #                                                               # 'start_id': key_val.word_to_id(key_val.start_label),
    #                                                               'end_token_id': key_val.word_to_id(key_val.end_label),
    #                                                               'max_decode_iterator_num': MAX_ITERATOR_LEGNTH,
    #                                                               'identifier_token': key_val.word_to_id(key_val.identifier_label),
    #                                                               'placeholder_token': key_val.word_to_id(key_val.placeholder_label),
    #                                                               'word_embedding_layer_fn': key_val.create_embedding_layer,
    #                                                               'character_embedding_layer_fn': char_voc.create_embedding_layer,
    #                                                               'id_to_word_fn': key_val.id_to_word,
    #                                                               'parse_token_fn': char_voc.parse_token})
    # train_supervision(TokenLevelMultiRnnModel, restore_param_generator, 1, restore=True)

    # import tensorflow as tf
    # with tf.Session():
    #     for params in param_generator(1):
    #         params['word_embedding_layer_fn'] = params['word_embedding_layer_fn']()
    #         params['character_embedding_layer_fn'] = params['character_embedding_layer_fn']()
    #         model = TokenLevelMultiRnnModel(**params)
    #         for i, data in enumerate(test_data_iterator()):
    #             model.metrics_model(*data)




