from code_data.constants import cache_data_path
from common import util
from common.beam_search_util import flat_list
from train.random_search import random_parameters_generator
from common.supervision_util_increment import create_supervision_experiment
from experiment.experiment_util import sample, create_embedding, error_count_without_train_condition_fn, create_error_list, \
    create_token_id_input, create_character_id_input, find_token_name, create_full_output, get_token_list, create_token_identify_mask, find_copy_id_by_identifier_dict, error_count_create_condition_fn
from model.masked_token_level_multirnn_model import MaskedTokenLevelMultiRnnModel
from code_data.read_data import read_cpp_fake_code_records_set


@util.disk_cache(basename='identifier_mask_token_level_multirnn_on_fake_cpp_data_parse_xy', directory=cache_data_path)
def parse_xy_with_identifier_mask(df, data_type:str, keyword_voc, char_voc, max_bug_number=1, min_bug_number=0):

    df['res'] = ''
    df['ac_code_obj'] = df['ac_code'].map(get_token_list)
    df = df[df['ac_code_obj'].map(lambda x: x is not None)].copy()

    df = df.apply(create_error_list, axis=1, raw=True)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_token_id_input, axis=1, raw=True, keyword_voc=keyword_voc)
    df = df[df['res'].map(lambda x: x is not None)].copy()

    df = df.apply(create_token_identify_mask, axis=1, raw=True)
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


if __name__ == '__main__':
    util.initLogging()
    util.set_cuda_devices(1)
    # train, test, vaild = read_cpp_fake_code_records_set()
    train, test, vaild = sample()
    # train = train.sample(300000)

    key_val, char_voc = create_embedding()
    parse_xy_param = [key_val, char_voc, 5, 1]


    # print(train)

    # res = parse_xy_with_identifier_mask(train, '', *parse_xy_param)
    # print(res[4].iloc[2])
    # print(res[9].iloc[2])

    # res = parse_xy(train, *parse_xy_param)
    # res = parse_xy_with_iden_mask(train, 'train', *parse_xy_param)
    # print('flat_len: ', ' '.join([str(len(x)) for x in res]))
    # res = parse_xy_with_iden_mask(train, '1', *parse_xy_param)
    # print('old_len: ', ' '.join([str(np.sum(list(map(len, x)))) for x in res]))

    # tmp = flat_list(res[0])
    # print(len(tmp), len(tmp[0]))
    # print(res[4])
    # print(len(res))
    # print(len(res[0]))
    # print(res[1].loc[197037][0], len(res[4].loc[197037][0]))

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
                        ({'error_count': 1}, 0.9),
                        ({'error_count': 2}, 0.8),
                        ({'error_count': 3}, 0.7),
                        ({'error_count': 4}, 0.7),
                        ({'error_count': 5}, 1.0), ]

    MAX_ITERATOR_LEGNTH = 5

    # train_supervision = create_supervision_experiment(train, test, vaild, parse_xy_with_iden_mask, parse_xy_param, experiment_name='token_level_multirnn_model', batch_size=16)

    train_supervision = create_supervision_experiment(train, test, vaild, parse_xy_with_identifier_mask, parse_xy_param, experiment_name='masked_token_level_multirnn_model', batch_size=16, create_condition_fn=error_count_create_condition_fn, modify_condition=modify_condition)
    param_generator = random_parameters_generator(random_param={"learning_rate": [-4, -1]},
                                                  choice_param={ },
                                                  constant_param={"hidden_size": 100,
                                                                  'rnn_layer_number': 2,
                                                                  'output_layer_num': 2,
                                                                  'decay_steps': 1000,
                                                                  'decay_rate': 0.96,
                                                                  'keyword_number': len(key_val.word_id_map),
                                                                  # 'start_id': key_val.word_to_id(key_val.start_label),
                                                                  'end_token_id': key_val.word_to_id(key_val.end_label),
                                                                  # 'max_decode_iterator_num': MAX_ITERATOR_LEGNTH,
                                                                  'identifier_token': key_val.word_to_id(key_val.identifier_label),
                                                                  'placeholder_token': key_val.word_to_id(key_val.placeholder_label),
                                                                  'word_embedding_layer_fn': key_val.create_embedding_layer,
                                                                  'character_embedding_layer_fn': char_voc.create_embedding_layer,
                                                                  'id_to_word_fn': key_val.id_to_word,
                                                                  'parse_token_fn': char_voc.parse_token})

    train_supervision(MaskedTokenLevelMultiRnnModel, param_generator, 1, debug=False, restore=False)
    # restore_param_generator = random_parameters_generator(random_param={ },
    #                                               choice_param={ },
    #                                               constant_param={"learning_rate": 0.000948072915975,
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
    # train_supervision(TokenLevelMultiRnnModel, restore_param_generator, 1, debug=False, restore=True)

    # import tensorflow as tf
    # with tf.Session():
    #     for params in param_generator(1):
    #         params['word_embedding_layer_fn'] = params['word_embedding_layer_fn']()
    #         params['character_embedding_layer_fn'] = params['character_embedding_layer_fn']()
    #         model = TokenLevelMultiRnnModel(**params)
    #         for i, data in enumerate(test_data_iterator()):
    #             model.metrics_model(*data)

