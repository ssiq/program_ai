from code_data.read_c_data import read_fake_common_c_error_dataset_with_limit_length, read_deepfix_error_data
from code_data.read_data import read_local_test_code_records
from common.analyse_include_util import replace_include_with_blank
from common.c_code_tokenize import tokenize_by_clex_fn
from experiment.c_error_experiment.experiment_util_for_c import \
    load_c_code_data_common_error_token_level_without_iscontinue, convert_c_code_fields_to_cpp_fields, \
    parse_xy_c_code_token_level_without_iscontinue, create_c_embedding, \
    parse_xy_c_code_token_level_without_iscontinue_only_input
from experiment.c_error_experiment.load_experiment_util import load_model_and_params_by_name
from experiment.experiment_util import parse_xy_token_level_without_iscontinue_without_character, create_embedding, \
    action_list_sorted, parse_xy_without_iscontinue_without_identifier_mask, sample_on_common_error_token_code_records, parse_xy_token_level_without_iscontinue
from common.test_supervision_util import create_test_experiment
from database.database_util import create_table, run_sql_statment
from code_data.constants import local_c_test_experiment_db, TEST_EXPERIMENT_RECORDS
from common import util

import pandas as pd
import json


def save_records(test_df, experiment_name='test_experiment_default', local_db_path=local_c_test_experiment_db):
    create_table(local_db_path, TEST_EXPERIMENT_RECORDS, replace_table_name=experiment_name)
    test_df['input_list'] = test_df['input_list'].map(json.dumps)
    test_df['output_list'] = test_df['output_list'].map(json.dumps)
    test_df['predict_list'] = test_df['predict_list'].map(json.dumps)
    test_df['distance'] = test_df['distance'].map(str)
    test_df['build_result'] = test_df['build_result'].map(str)

    store_list = [[*one] for one in zip(test_df['id'], test_df['time'], test_df['build_start_time'],
                                    test_df['build_end_time'], test_df['solution_name'], test_df['project_name'],
                                    test_df['build_log_content'], test_df['compile_command'], test_df['files'],
                                    test_df['build_error_info'], test_df['build_result'], test_df['code'],
                                    test_df['ac_code'], test_df['action_character_list'], test_df['distance'],
                                    test_df['input_list'], test_df['output_list'], test_df['predict_list'])]
    run_sql_statment(local_c_test_experiment_db, TEST_EXPERIMENT_RECORDS, 'insert_ignore', store_list, replace_table_name=experiment_name)


def save_train_records(test_df, experiment_name='test_experiment_default', local_db_path=local_c_test_experiment_db):
    create_table(local_db_path, TEST_EXPERIMENT_RECORDS, replace_table_name=experiment_name)
    test_df['input_list'] = test_df['input_list'].map(json.dumps)
    test_df['output_list'] = test_df['output_list'].map(json.dumps)
    test_df['predict_list'] = test_df['predict_list'].map(json.dumps)
    test_df['distance'] = test_df['error_count'].map(str)
    blank_list = ['' for i in range(len(test_df))]

    store_list = [[*one] for one in zip(test_df['id'], blank_list, blank_list, blank_list, blank_list, blank_list,
                                        blank_list, blank_list, blank_list, blank_list, blank_list, test_df['code'],
                                        test_df['ac_code'], test_df['action_character_list'], test_df['distance'],
                                        test_df['input_list'], test_df['output_list'], test_df['predict_list'])]
    run_sql_statment(local_c_test_experiment_db, TEST_EXPERIMENT_RECORDS, 'insert_ignore', store_list,
                     replace_table_name=experiment_name)



def deepfix_main():
    tokenize_fn = tokenize_by_clex_fn()
    test_df = read_deepfix_error_data()

    test_df['id'] = test_df['code_id']
    test_df['ac_code'] = test_df['code'].map(replace_include_with_blank)
    test_df['action_character_list'] = '[]'
    test_df['error_count'] = test_df['errorcount']

    parse_xy_fn = parse_xy_c_code_token_level_without_iscontinue_only_input
    key_val, char_voc = create_c_embedding()
    parse_xy_param = [key_val, char_voc, tokenize_fn]

    res = parse_xy_fn(test_df, '', *parse_xy_param)
    test_df = test_df.loc[res[0].index]
    print('in main df length: ', len(test_df))

    experiment_name = 'deepfix_test_final_iterative_model_using_common_error_without_iscontinue'

    model_name = 'c_final_iterative_model_using_common_error_without_iscontinue'

    return test_df, parse_xy_fn, parse_xy_param, experiment_name, model_name


if __name__ == '__main__':
    MAX_TOKEN_LENGTH = 500
    util.set_cuda_devices(1)

    # tokenize_fn = tokenize_by_clex_fn()
    #
    # _, _, test_df = read_fake_common_c_error_dataset_with_limit_length(MAX_TOKEN_LENGTH)
    # # test_df = test_df.sample(50)
    # test_df = convert_c_code_fields_to_cpp_fields(test_df)
    #
    # parse_xy_fn = parse_xy_c_code_token_level_without_iscontinue
    # key_val, char_voc = create_c_embedding()
    # max_bug_number = 10
    # min_bug_number = 1
    # parse_xy_param = [key_val, char_voc, max_bug_number, min_bug_number, action_list_sorted, tokenize_fn]
    #
    # res = parse_xy_fn(test_df, '', *parse_xy_param)
    # test_df = test_df.loc[res[0].index]
    # print('in main df length: ', len(test_df))
    #
    # experiment_name = 'c_final_iterative_model_using_common_error_without_iscontinue'
    # experiment_name = 'deepfix_test_final_iterative_model_using_common_error_without_iscontinue'
    #
    # model_name = 'c_final_iterative_model_using_common_error_without_iscontinue'

    test_df, parse_xy_fn, parse_xy_param, experiment_name, model_name = deepfix_main()
    # test_df = test_df.sample(50)

    test_model_fn = create_test_experiment(test_df, parse_xy_fn, parse_xy_param,
                                           experiment_name=model_name,
                                           batch_size=16, input_length=5)

    model, restore_param_generator = load_model_and_params_by_name(experiment_name)
    input_list, output_list, predict_list = test_model_fn(model, restore_param_generator)

    print(predict_list)

    test_df['input_list'] = pd.Series(list(input_list)).values
    if len(output_list) != len(test_df):
        test_df['output_list'] = [[] for i in range(len(test_df))]
    else:
        test_df['output_list'] = pd.Series(list(output_list)).values
    test_df['predict_list'] = pd.Series(list(predict_list)).values
    # print('final_predict_list: {}'.format(test_df['predict_list'].iloc[0]))
    print('total predict list: {}'.format(len(test_df['predict_list'])))

    # print(test_df)
    # save_records(test_df, experiment_name=experiment_name)
    save_train_records(test_df, experiment_name=experiment_name)
