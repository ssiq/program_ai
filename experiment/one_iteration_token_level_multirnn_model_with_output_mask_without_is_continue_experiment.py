from code_data.read_data import read_cpp_fake_code_records_set, read_cpp_random_token_code_records_set
from common import util
from common.supervision_util_increment import create_supervision_experiment
from experiment.experiment_util import sample, create_embedding, error_count_create_condition_fn, \
    parse_xy_with_identifier_mask, parse_xy_token_level, sample_on_random_token_code_records, load_data_token_level, \
    load_data_token_level_sample, load_data_token_level__without_iscontinue_sample, load_data_token_level_without_iscontinue, \
    no_condition_create_fn
from model.masked_token_level_multirnn_model import MaskedTokenLevelMultiRnnModel
from model.one_iteration_token_level_multirnn_model_with_output_mask_without_is_continue import TokenLevelMultiRnnModel
from train.random_search import random_parameters_generator

if __name__ == '__main__':
    util.initLogging()
    util.set_cuda_devices(0)
    # train, test, vaild = read_cpp_random_token_code_records_set()
    load_data_fn = load_data_token_level_without_iscontinue
    # train, test, vaild = sample_on_random_token_code_records()
    # load_data_fn = load_data_token_level_sample
    # load_data_fn = load_data_token_level__without_iscontinue_sample

    key_val, char_voc = create_embedding()
    # parse_xy_param = [key_val, char_voc, 5, 1]
    load_data_param = [5, 1]


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
                        # ({'error_count': 1}, 0.9),
                        # ({'error_count': 2}, 0.8),
                        # ({'error_count': 3}, 0.7),
                        # ({'error_count': 4}, 0.7),
                        ({'error_count': 5}, 1.0), ]

    MAX_ITERATOR_LEGNTH = 5

    # train_supervision = create_supervision_experiment(train, test, vaild, parse_xy_with_iden_mask, parse_xy_param, experiment_name='token_level_multirnn_model', batch_size=16)

    train_supervision = create_supervision_experiment(load_data_fn, load_data_param, experiment_name='one_iteration_token_level_multirnn_model_without_iscontinue', batch_size=16, create_condition_fn=error_count_create_condition_fn, modify_condition=modify_condition)
    param_generator = random_parameters_generator(random_param={"learning_rate": [-4, -3]},
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

    train_supervision(TokenLevelMultiRnnModel, param_generator, 1, debug=False, restore=False)
    # restore_param_generator = random_parameters_generator(random_param={ },
    #                                               choice_param={ },
    #                                               constant_param={"learning_rate": 0.000678127906854,
    #                                                               "hidden_size": 100,
    #                                                               'rnn_layer_number': 2,
    #                                                               'output_layer_num': 2,
    #                                                               'decay_steps': 1000,
    #                                                               'decay_rate': 0.96,
    #                                                               'keyword_number': len(key_val.word_id_map),
    #                                                               # 'start_id': key_val.word_to_id(key_val.start_label),
    #                                                               'end_token_id': key_val.word_to_id(key_val.end_label),
    #                                                               # 'max_decode_iterator_num': MAX_ITERATOR_LEGNTH,
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
