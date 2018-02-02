from experiment.experiment_util import create_embedding
from train.random_search import random_parameters_generator


def load_model_and_params_by_name(experiment_name):
    load_fn = load_fn_dict[experiment_name]
    return load_fn()


def load_final_iterative_model_using_common_error_without_iscontinue_experiment():
    from model.final_iterative_model import TokenLevelMultiRnnModel
    key_val, char_voc = create_embedding()
    restore_param_generator = random_parameters_generator(random_param={},
                                                          choice_param={},
                                                          constant_param={"learning_rate": 0.000103202489861,
                                                                          "hidden_size": 100,
                                                                          'rnn_layer_number': 2,
                                                                          'output_layer_num': 2,
                                                                          'decay_steps': 1000,
                                                                          'decay_rate': 0.96,
                                                                          'keyword_number': len(key_val.word_id_map),
                                                                          # 'start_id': key_val.word_to_id(key_val.start_label),
                                                                          'end_token_id': key_val.word_to_id(
                                                                              key_val.end_label),
                                                                          # 'max_decode_iterator_num': MAX_ITERATOR_LEGNTH,
                                                                          'identifier_token': key_val.word_to_id(
                                                                              key_val.identifier_label),
                                                                          'placeholder_token': key_val.word_to_id(
                                                                              key_val.placeholder_label),
                                                                          'word_embedding_layer_fn': key_val.create_embedding_layer,
                                                                          'character_embedding_layer_fn': char_voc.create_embedding_layer,
                                                                          'id_to_word_fn': key_val.id_to_word,
                                                                          'parse_token_fn': char_voc.parse_token})
    return TokenLevelMultiRnnModel, restore_param_generator


def load_final_iterative_model_without_iscontinue_experiment():
    from model.final_iterative_model import TokenLevelMultiRnnModel
    key_val, char_voc = create_embedding()
    restore_param_generator = random_parameters_generator(random_param={},
                                                          choice_param={},
                                                          constant_param={"learning_rate": 0.000597488422048,
                                                                          "hidden_size": 100,
                                                                          'rnn_layer_number': 2,
                                                                          'output_layer_num': 2,
                                                                          'decay_steps': 1000,
                                                                          'decay_rate': 0.96,
                                                                          'keyword_number': len(key_val.word_id_map),
                                                                          # 'start_id': key_val.word_to_id(key_val.start_label),
                                                                          'end_token_id': key_val.word_to_id(
                                                                              key_val.end_label),
                                                                          # 'max_decode_iterator_num': MAX_ITERATOR_LEGNTH,
                                                                          'identifier_token': key_val.word_to_id(
                                                                              key_val.identifier_label),
                                                                          'placeholder_token': key_val.word_to_id(
                                                                              key_val.placeholder_label),
                                                                          'word_embedding_layer_fn': key_val.create_embedding_layer,
                                                                          'character_embedding_layer_fn': char_voc.create_embedding_layer,
                                                                          'id_to_word_fn': key_val.id_to_word,
                                                                          'parse_token_fn': char_voc.parse_token})
    return TokenLevelMultiRnnModel, restore_param_generator


def load_one_iteration_token_level_multirnn_model_using_common_error_with_output_mask_without_is_continue_experiment():
    from model.one_iteration_token_level_multirnn_model_with_output_mask_without_is_continue import TokenLevelMultiRnnModel
    key_val, char_voc = create_embedding()
    param_generator = random_parameters_generator(random_param={ },
                                                  choice_param={ },
                                                  constant_param={'learning_rate': 0.000623810360711,
                                                                  "hidden_size": 100,
                                                                  'rnn_layer_number': 2,
                                                                  'output_layer_num': 2,
                                                                  'decay_steps': 1000,
                                                                  'decay_rate': 0.96,
                                                                  'keyword_number': len(key_val.word_id_map),
                                                                  # 'start_id': key_val.word_to_id(key_val.start_label),
                                                                  'end_token_id': key_val.word_to_id(key_val.end_label),
                                                                  # 'max_decode_iterator_num': MAX_ITERATOR_LEGNTH,
                                                                  'identifier_token': key_val.word_to_id(
                                                                      key_val.identifier_label),
                                                                  'placeholder_token': key_val.word_to_id(
                                                                      key_val.placeholder_label),
                                                                  'word_embedding_layer_fn': key_val.create_embedding_layer,
                                                                  'character_embedding_layer_fn': char_voc.create_embedding_layer,
                                                                  'id_to_word_fn': key_val.id_to_word,
                                                                  'parse_token_fn': char_voc.parse_token})
    return TokenLevelMultiRnnModel, param_generator


def load_one_iteration_token_level_multirnn_model_with_output_mask_without_is_continue_experiment():
    from model.one_iteration_token_level_multirnn_model_with_output_mask_without_is_continue import TokenLevelMultiRnnModel
    key_val, char_voc = create_embedding()
    param_generator = random_parameters_generator(random_param={ },
                                                  choice_param={ },
                                                  constant_param={'learning_rate': 0.000929280128196,
                                                                  "hidden_size": 100,
                                                                  'rnn_layer_number': 2,
                                                                  'output_layer_num': 2,
                                                                  'decay_steps': 1000,
                                                                  'decay_rate': 0.96,
                                                                  'keyword_number': len(key_val.word_id_map),
                                                                  # 'start_id': key_val.word_to_id(key_val.start_label),
                                                                  'end_token_id': key_val.word_to_id(key_val.end_label),
                                                                  # 'max_decode_iterator_num': MAX_ITERATOR_LEGNTH,
                                                                  'identifier_token': key_val.word_to_id(
                                                                      key_val.identifier_label),
                                                                  'placeholder_token': key_val.word_to_id(
                                                                      key_val.placeholder_label),
                                                                  'word_embedding_layer_fn': key_val.create_embedding_layer,
                                                                  'character_embedding_layer_fn': char_voc.create_embedding_layer,
                                                                  'id_to_word_fn': key_val.id_to_word,
                                                                  'parse_token_fn': char_voc.parse_token})
    return TokenLevelMultiRnnModel, param_generator


def load_one_iteration_token_level_multirnn_model_without_is_continue_experiment():
    from model.one_iteration_token_level_multirnn_model_without_is_continue import TokenLevelMultiRnnModel
    key_val, char_voc = create_embedding()
    param_generator = random_parameters_generator(random_param={ },
                                                  choice_param={ },
                                                  constant_param={'learning_rate': 0.000935706050305,
                                                                  "hidden_size": 100,
                                                                  'rnn_layer_number': 2,
                                                                  'output_layer_num': 2,
                                                                  'decay_steps': 1000,
                                                                  'decay_rate': 0.96,
                                                                  'keyword_number': len(key_val.word_id_map),
                                                                  # 'start_id': key_val.word_to_id(key_val.start_label),
                                                                  'end_token_id': key_val.word_to_id(key_val.end_label),
                                                                  # 'max_decode_iterator_num': MAX_ITERATOR_LEGNTH,
                                                                  'identifier_token': key_val.word_to_id(
                                                                      key_val.identifier_label),
                                                                  'placeholder_token': key_val.word_to_id(
                                                                      key_val.placeholder_label),
                                                                  'word_embedding_layer_fn': key_val.create_embedding_layer,
                                                                  'character_embedding_layer_fn': char_voc.create_embedding_layer,
                                                                  'id_to_word_fn': key_val.id_to_word,
                                                                  'parse_token_fn': char_voc.parse_token})
    return TokenLevelMultiRnnModel, param_generator


load_fn_dict = {
    'final_iterative_model_without_iscontinue': load_final_iterative_model_without_iscontinue_experiment,
    'final_iterative_model_using_common_error_without_iscontinue': load_final_iterative_model_using_common_error_without_iscontinue_experiment,
    'one_iteration_token_level_multirnn_model_using_common_error_without_iscontinue': load_one_iteration_token_level_multirnn_model_using_common_error_with_output_mask_without_is_continue_experiment,
    'one_iteration_token_level_multirnn_model_without_iscontinue': load_one_iteration_token_level_multirnn_model_with_output_mask_without_is_continue_experiment,
    'final_iterative_model_using_common_error_without_iscontinue_without_beam_search': load_final_iterative_model_using_common_error_without_iscontinue_experiment,
    'one_iteration_token_level_multirnn_model_using_common_error_without_iscontinue_without_beam_search': load_one_iteration_token_level_multirnn_model_using_common_error_with_output_mask_without_is_continue_experiment,
    'one_iteration_token_level_multirnn_model_using_common_error_without_iscontinue_without_identifier_mask': load_one_iteration_token_level_multirnn_model_without_is_continue_experiment,
    'one_iteration_token_level_multirnn_model_using_common_error_without_iscontinue_without_sample_mask': load_one_iteration_token_level_multirnn_model_using_common_error_with_output_mask_without_is_continue_experiment,
}