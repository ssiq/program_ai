from common import util
from code_data.read_data import read_cpp_token_fake_code_set
from embedding.wordembedding import load_vocabulary, Vocabulary
from embedding.character_embedding import load_character_vocabulary
from common.code_tokenize import GetTokens
from common.supervision_util import create_supervision_experiment
from model.seq2seq_model import Seq2SeqModel
from train.random_search import random_parameters_generator
import pandas as pd

def parse_xy(df):

    def get_token_list(code):
        try:
            code = code.replace('\r', '')
            if code.find('define') != -1 or code.find('param') != -1 or code.find('ifndef') != -1 or code.find('endif') != -1:
                return None
            tokens = list(GetTokens(code))
            if None in tokens:
                return None
        except RuntimeError as e:
            print('in runtime error')
            return None
        return tokens

    def token_to_name_list(token_list):
        return [token.name for token in token_list]

    def name_to_char_list(name_list):
        char_list = [list(x) for x in name_list]
        return char_list

    def charlist_to_len_list(char_list):
        len_list = [len(x) for x in char_list]
        return len_list

    def find_copy_id(one):
        is_copys = one['is_copy']
        copys = []
        for c in is_copys:
            if c == 1:
                name_list = one['parse_error_token_name']
                actionsign = one['actionsign']
                pos = -1
                for i in range(0, len(name_list)):
                    if actionsign == name_list[i]:
                        pos = i
                        break
                if pos == -1:
                    return None
                copys.append(pos)
            else:
                copys.append(-1)
        return copys

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

    keyword_voc = load_vocabulary('keyword', embedding_size=300)

    df['parse_original_token_obj'] = df['originalcode'].map(get_token_list)
    df = df[df['parse_original_token_obj'].map(lambda x: x is not None)].copy()

    df['parse_original_token_name'] = df['parse_original_token_obj'].map(token_to_name_list)
    df['parse_original_token_id'] = keyword_voc.parse_text_without_pad(df['parse_original_token_name'], True)
    df['parse_original_token_len'] = df['parse_original_token_id'].map(len)

    df['parse_error_token_name'] = df.apply(create_error_token, axis=1, raw=True)
    df = df[df['parse_error_token_name'].map(lambda x: x is not None)].copy()
    df['parse_error_token_id'] = keyword_voc.parse_text_without_pad(df['parse_error_token_name'], True)
    df['parse_error_token_len'] = df['parse_error_token_id'].map(len)

    char_voc = load_character_vocabulary('bigru', n_gram=2, embedding_shape=300, token_list=df['parse_error_token_name'])

    df['parse_error_char_list'] = char_voc.parse_string_without_padding(df['parse_error_token_name'])
    df['parse_error_char_len'] = df['parse_error_char_list'].map(charlist_to_len_list)

    df['modify_action_type'] = df.apply(ensure_action_type, axis=1, raw=True)

    df['output_length'] = 1

    df['sign_id'] = df['actionsign'].map(keyword_voc.word_to_id)
    df['is_copy'] = df['actionsign'].map(lambda x: [1] if (keyword_voc.word_to_id(x) == (len(keyword_voc.word_id_map)-1))else [0])

    df['keyword_id'] = df['actionsign'].map(lambda x: [-1] if (keyword_voc.word_to_id(x) == (len(keyword_voc.word_id_map)-1))else [keyword_voc.word_to_id(x)])
    df['copy_id'] = df.apply(find_copy_id, axis=1, raw=True)
    df = df[df['copy_id'].map(lambda x: x is not None)].copy()

    return df['parse_error_token_id'], df['parse_original_token_len'], df['parse_error_char_list'], df['parse_error_char_len'], df['output_length'], df['is_copy'], df['keyword_id'], df['copy_id']



if __name__ == '__main__':
    util.set_cuda_devices()
    train, test, vaild = read_cpp_token_fake_code_set()

    train_supervision = create_supervision_experiment(train, test, vaild, parse_xy, experiment_name='seq2seq_test')
    param_generator = random_parameters_generator(random_param={"learning_rate": [-5, 0]},
                                                  choice_param={"state_size": [100]},
                                                  constant_param={ })

    train_supervision(Seq2SeqModel, param_generator)

    # isPrint = 0
    # for i in test_data_iterator():
    #     # if isPrint < 2:
    #     print(isPrint)
    #     for t in i:
    #         for k in t:
    #             if isinstance(k, list):
    #                 # print(len(k))
    #                 for u in k:
    #                     if isinstance(u, list):
    #                         print(len(u))
    #     isPrint += 1


