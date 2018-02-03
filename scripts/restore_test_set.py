import json
from scripts.filter_student_code_action import get_effective_student_test_set
from database.database_util import create_table, insert_items
from code_data.constants import local_student_db_path, TEST_CODE_RECORDS


def restore_test_records():
    test_df = get_effective_student_test_set()
    test_df['files'] = test_df['files'].map(json.dumps)
    test_df['build_error_info'] = test_df['build_error_info'].map(json.dumps)
    test_dict = test_df.to_dict('records')
    convert_df_to_tuple_fn = lambda x: (x['id'], str(x['time']), x['build_start_time'], x['build_end_time'],
                                        x['solution_name'], x['project_name'], x['build_log_content'],
                                        x['compile_command'], x['files'], x['build_error_info'],
                                        x['build_result'], x['file_content'], x['ac_code'],
                                        x['action_character_list'], x['distance'])
    test_info = [convert_df_to_tuple_fn(i) for i in test_dict]
    print('test info len: ', len(test_info))
    create_table(local_student_db_path, TEST_CODE_RECORDS)
    insert_items(local_student_db_path, TEST_CODE_RECORDS, test_info)


if __name__ == '__main__':
    restore_test_records()
