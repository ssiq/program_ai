from common import util
from code_data.read_data import read_cpp_token_fake_code_set
from common.code_tokenize import GetTokens
from common.supervision_util import create_supervision_experiment
from model.seq2seq_model import Seq2SeqModel
from train.random_search import random_parameters_generator

MAX_TOKEN_LENGTH = 200
MAX_ITERATOR_LEGNTH = 250

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


def token_to_name_list(token_list):
    return [token.name for token in token_list]


def create_error_token(one):
    CHANGE = 0
    INSERT = 1
    DELETE = 2
    actiontype = one['actiontype']
    actionpos = int(one['actionpos'])
    actionsign = one['actionsign']
    original_name_list = one['parse_original_token_name']
    error_name_list = original_name_list.copy()

    if actiontype == INSERT:
        try:
            if len(error_name_list) < actionpos:
                return None
            error_name_list.insert(actionpos, actionsign)
        except IndexError as e:
            print('insert index error')
            return None
    elif actiontype == DELETE:
        try:
            if error_name_list[actionpos] != actionsign:
                return None
            error_name_list.pop(actionpos)
        except IndexError as e:
            print('delete index error')
            return None
    elif actiontype == CHANGE:
        error_str = one['code']
        i = 0
        for i in range(0, actionpos):
            error_str = error_str.strip()
            error_str = error_str[len(original_name_list[i]):]

        i = len(original_name_list)-1
        while i > actionpos:
            error_str = error_str.strip()
            error_str = error_str[:-len(original_name_list[i])]
            i -= 1
        error_str = error_str.strip()

        if error_str != '':
            error_name_list[actionpos] = error_str
        else:
            actiontype = DELETE
            # print('CHANGE TO DELETE')
            error_name_list.pop(actionpos)

    return error_name_list


def parse_xy(df, keyword_voc, char_voc):

    def token_to_name_list(token_list):
        return [token.name for token in token_list]

    def name_to_char_list(name_list):
        char_list = [list(x) for x in name_list]
        return char_list

    def charlist_to_len_list(char_list):
        len_list = [len(x) for x in char_list]
        return len_list

    def find_keyword_id(one):
        keywords = []
        is_copys = one['is_copy']
        actionpos = int(one['actionpos'])
        ori_len = one['parse_original_token_len']
        err_len = one['parse_error_token_len']
        while len(keywords) < ori_len:
            keywords.append(0)
        if ori_len >= err_len:
            if is_copys[actionpos] == 0:
                keywords[actionpos] = keyword_voc.word_to_id(one['actionsign'])
        keywords.insert(0, keyword_voc.word_to_id(keyword_voc.start_label))
        keywords.append(keyword_voc.word_to_id(keyword_voc.end_label))
        return keywords


    def find_copy_id(one):
        is_copys = one['is_copy'][1:-1]
        copys = []
        actionpos = int(one['actionpos'])
        ori_len = one['parse_original_token_len']
        err_len = one['parse_error_token_len']

        ori_i = 0
        err_i = 0

        while ori_i < ori_len:
            if ori_i != actionpos:
                copys.insert(ori_i, err_i)
            elif ori_i == actionpos:
                if ori_len >= err_len:
                    name_list = one['parse_error_token_name']
                    actionsign = one['actionsign']
                    pos = -1
                    for i in range(0, len(name_list)):
                        if actionsign == name_list[i]:
                            pos = i
                            break
                    if pos == -1 and is_copys[ori_i] == 1:
                        return None
                    copys.insert(ori_i, pos)
                    if ori_len > err_len:
                        err_i -= 1
                elif ori_len < err_len:
                    err_i += 1
                    copys.insert(ori_i, err_i)

            if is_copys[ori_i] == 0:
                copys[ori_i] = 0

            ori_i += 1
            err_i += 1

        copys.insert(0, 0)
        copys.append(0)

        return copys

    def create_iscopy(one):

        ori_len = one['parse_original_token_len']
        err_len = one['parse_error_token_len']
        actionpos = int(one['actionpos'])
        actionsign = one['actionsign']
        is_copys = []
        while len(is_copys) < ori_len:
            is_copys.append(1)

        if ori_len >= err_len:
            if keyword_voc.word_to_id(actionsign) != keyword_voc.word_to_id(keyword_voc.identifier_label):
                is_copys[actionpos] = 0
        is_copys.insert(0, 0)
        is_copys.append(0)
        return is_copys


    def ensure_action_type(one):
        CHANGE = 0
        INSERT = 1
        DELETE = 2
        error_len = one['parse_error_token_len']
        ori_len = one['parse_original_token_len']
        if error_len < ori_len:
            return INSERT
        elif error_len > ori_len:
            return DELETE
        return CHANGE

    # keyword_voc = load_vocabulary('keyword', embedding_size=300)

    df['parse_original_token_obj'] = df['originalcode'].map(get_token_list)
    df = df[df['parse_original_token_obj'].map(lambda x: x is not None)].copy()

    df['parse_original_token_name'] = df['parse_original_token_obj'].map(token_to_name_list)
    df['parse_original_token_id'] = keyword_voc.parse_text_without_pad(df['parse_original_token_name'], True)
    df = df[df['parse_original_token_id'].map(lambda x: x is not None)].copy()
    df['parse_original_token_len'] = df['parse_original_token_id'].map(len)

    df['parse_error_token_name'] = df.apply(create_error_token, axis=1, raw=True)
    df = df[df['parse_error_token_name'].map(lambda x: x is not None)].copy()
    df['parse_error_token_id'] = keyword_voc.parse_text_without_pad(df['parse_error_token_name'], True)
    df = df[df['parse_error_token_id'].map(lambda x: x is not None)].copy()
    df['parse_error_token_len'] = df['parse_error_token_id'].map(len)

    # char_voc = load_character_vocabulary('bigru', n_gram=2, embedding_shape=150, token_list=df['parse_error_token_name'])

    df['parse_error_char_list'] = char_voc.parse_string_without_padding(df['parse_error_token_name'])
    df = df[df['parse_error_char_list'].map(lambda x: x is not None)].copy()
    df['parse_error_char_len'] = df['parse_error_char_list'].map(charlist_to_len_list)

    # df['modify_action_type'] = df.apply(ensure_action_type, axis=1, raw=True)

    # df['output_length'] = df['parse_original_token_len']

    # df['sign_id'] = df['actionsign'].map(keyword_voc.word_to_id)
    df['is_copy'] = df.apply(create_iscopy, axis=1, raw=True)
    df['output_len'] = df['is_copy'].map(len)

    df['keyword_id'] = df.apply(find_keyword_id, axis=1, raw=True)
    df['copy_id'] = df.apply(find_copy_id, axis=1, raw=True)
    df = df[df['copy_id'].map(lambda x: x is not None)].copy()

    print('parse_xy_data_shape: {}'.format(df.shape))

    return df['parse_error_token_id'], df['parse_error_token_len'], df['parse_error_char_list'], df['parse_error_char_len'], df['output_len'], df['is_copy'], df['keyword_id'], df['copy_id']


from code_data.constants import cache_data_path
@util.disk_cache(basename='token_level_rnn_on_fake_cpp_data_sample', directory=cache_data_path)
def sample():
    train, test, vaild = read_cpp_token_fake_code_set()
    train = train.sample(100, random_state=1)
    test = test.sample(100, random_state=1)
    vaild = vaild.sample(100, random_state=1)
    return (train, test, vaild)


def create_embedding():
    from code_data.constants import char_sign_dict
    key_val = load_vocabulary('keyword', embedding_size=200)
    char_voc = load_character_vocabulary('bigru', n_gram=1, embedding_shape=100, token_list=char_sign_dict.keys())
    return key_val, char_voc

if __name__ == '__main__':
    FILE_PATH = '/home/lf/Project/program_ai/log/seq2seq.log'
    util.initlogging(FILE_PATH)
    util.set_cuda_devices(1)
    train, test, vaild = read_cpp_token_fake_code_set()
    # train = train.sample(200000)

    from embedding.wordembedding import load_vocabulary
    from embedding.character_embedding import load_character_vocabulary

    # train, test, vaild = sample()

    key_val, char_voc = create_embedding()

    parse_xy_param = [key_val, char_voc]

    # test_data_iterator = util.batch_holder(*parse_xy(test, *parse_xy_param), batch_size=2)
    #
    # isPrint = 0
    # for i in test_data_iterator():
    #     # if isPrint < 2:
    #     print(isPrint)
    #     for t in i:
    #         for k in t:
    #             if isinstance(k, list):
    #                 print(len(k))
    #                 for u in k:
    #                     if isinstance(u, list):
    #                         # print(len(u))
    #                         pass
    #     isPrint += 1

    train_supervision = create_supervision_experiment(train, test, vaild, parse_xy, parse_xy_param, experiment_name='seq2seq', batch_size=8)
    param_generator = random_parameters_generator(random_param={"learning_rate": [-5, 0]},
                                                  choice_param={ },
                                                  constant_param={"hidden_size": 100,
                                                                  'rnn_layer_number': 2,
                                                                  'keyword_number': len(key_val.word_id_map),
                                                                  'start_id': key_val.word_to_id(key_val.start_label),
                                                                  'end_token_id': key_val.word_to_id(key_val.end_label),
                                                                  'max_decode_iterator_num': MAX_ITERATOR_LEGNTH,
                                                                  'identifier_token': key_val.word_to_id(key_val.identifier_label),
                                                                  'word_embedding_layer_fn': key_val.create_embedding_layer,
                                                                  'character_embedding_layer_fn': char_voc.create_embedding_layer})

    train_supervision(Seq2SeqModel, param_generator)




