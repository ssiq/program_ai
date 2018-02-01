from code_data.read_data import read_local_test_code_records
from experiment.experiment_util import parse_xy_token_level_without_iscontinue, create_embedding, action_list_sorted
from common.test_supervision_util import create_test_experiment
from experiment.load_experiment_util import load_model_and_params_by_name
from database.database_util import create_table, run_sql_statment
from code_data.constants import local_test_experiment_db, TEST_EXPERIMENT_RECORDS
from common import util

import pandas as pd
import json


def save_records(test_df, experiment_name='test_experiment_default', local_db_path=local_test_experiment_db):
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
    run_sql_statment(local_test_experiment_db, TEST_EXPERIMENT_RECORDS, 'insert_ignore', store_list, replace_table_name=experiment_name)


if __name__ == '__main__':
    util.set_cuda_devices(1)
    test_df = read_local_test_code_records()
    # test_df = test_df.sample(10)

    parse_xy_fn = parse_xy_token_level_without_iscontinue
    key_val, char_voc = create_embedding()
    max_bug_number = 5
    min_bug_number = 1
    parse_xy_param = [key_val, char_voc, max_bug_number, min_bug_number, action_list_sorted]

    # res = parse_xy_token_level_without_iscontinue(test_df, '', *parse_xy_param)

    # experiment_name = 'final_iterative_model_using_common_error_without_iscontinue'
    # experiment_name = 'final_iterative_model_without_iscontinue'
    # experiment_name = 'one_iteration_token_level_multirnn_model_without_iscontinue'
    experiment_name = 'one_iteration_token_level_multirnn_model_using_common_error_without_iscontinue'

    test_model_fn = create_test_experiment(test_df, parse_xy_fn, parse_xy_param,
                                           experiment_name=experiment_name,
                                           batch_size=16, input_length=5)

    model, restore_param_generator = load_model_and_params_by_name(experiment_name)
    input_list, output_list, predict_list = test_model_fn(model, restore_param_generator)
    test_df['input_list'] = pd.Series(list(input_list)).values
    test_df['output_list'] = pd.Series(list(output_list)).values
    test_df['predict_list'] = pd.Series(list(predict_list)).values
    # print('final_predict_list: {}'.format(test_df['predict_list'].iloc[0]))
    print('total predict list: {}'.format(len(test_df['predict_list'])))

    save_records(test_df, experiment_name=experiment_name)
