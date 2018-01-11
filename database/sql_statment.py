from code_data.constants import FAKE_CODE_RECORDS, STUDENT_BUILD_INFO, BUILD_ERROR_STAT, RANDOM_TOKEN_CODE_RECORDS

CREATE_FAKE_CODE_RECORDS = r'''CREATE TABLE IF NOT EXISTS fake_code_records (
  id TEXT PRIMARY KEY,
  submit_id TEXT,
  problem_id TEXT,
  user_id TEXT,
  ac_code TEXT,
  code TEXT,
  error_count INTEGER,
  error_character_maplist TEXT,
  action_character_list TEXT
)'''

CREATE_RANDOM_TOKEN_CODE_RECORDS = r'''CREATE TABLE IF NOT EXISTS random_token_code_records (
  id TEXT PRIMARY KEY,
  submit_id TEXT,
  problem_id TEXT,
  user_id TEXT,
  ac_code TEXT,
  code TEXT,
  error_count INTEGER,
  error_character_maplist TEXT,
  action_character_list TEXT
)'''

CREATE_STUDENT_BUILD_INFO = r'''CREATE TABLE IF NOT EXISTS student_build_info (
  id TEXT PRIMARY KEY,
  time TEXT, 
  build_start_time TEXT, 
  build_end_time TEXT, 
  solution_name TEXT, 
  project_name TEXT, 
  build_log_content TEXT, 
  compile_command TEXT, 
  files TEXT, 
  build_error_info TEXT, 
  build_result INTEGER, 
  file_content TEXT
)'''

CREATE_BUILD_ERROR_STAT = r'''CREATE TABLE IF NOT EXISTS build_error_stat (
  id TEXT PRIMARY KEY,
  count INTEGER,
  error_codes TEXT, 
  error_content TEXT, 
  percent FLOAT
)'''

INSERT_IGNORE_FAKE_CODE_RECORDS = r'''INSERT OR IGNORE INTO fake_code_records (id, submit_id, problem_id, user_id, ac_code, code, error_count, error_character_maplist, action_character_list) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'''
INSERT_IGNORE_RANDOM_TOKEN_CODE_RECORDS = r'''INSERT OR IGNORE INTO random_token_code_records (id, submit_id, problem_id, user_id, ac_code, code, error_count, error_character_maplist, action_character_list) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'''
INSERT_IGNORE_STUDENT_BUILD_INFO = r'''INSERT OR IGNORE INTO student_build_info (id, time, build_start_time, build_end_time, solution_name, project_name, build_log_content, compile_command, files, build_error_info, build_result, file_content) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
INSERT_IGNORE_BUILD_ERROR_STAT = r'''INSERT OR IGNORE INTO build_error_stat (id, count, error_codes, error_content, percent) VALUES (?, ?, ?, ?, ?)'''

FIND_IDS_BY_USER_PROBLEM_ID_FAKE_CODE_RECORDS = r'''SELECT id FROM fake_code_records WHERE id in ( {} )'''
FIND_IDS_BY_USER_PROBLEM_ID_RANDOM_TOKEN_CODE_RECORDS = r'''SELECT id FROM random_token_code_records WHERE id in ( {} )'''

sql_dict = {FAKE_CODE_RECORDS: {'create': CREATE_FAKE_CODE_RECORDS, 'insert_ignore': INSERT_IGNORE_FAKE_CODE_RECORDS,
                                'find_ids_by_user_problem_id': FIND_IDS_BY_USER_PROBLEM_ID_FAKE_CODE_RECORDS},
            RANDOM_TOKEN_CODE_RECORDS: {'create': CREATE_RANDOM_TOKEN_CODE_RECORDS, 'insert_ignore': INSERT_IGNORE_RANDOM_TOKEN_CODE_RECORDS,
                                'find_ids_by_user_problem_id': FIND_IDS_BY_USER_PROBLEM_ID_RANDOM_TOKEN_CODE_RECORDS},
            STUDENT_BUILD_INFO: {'create': CREATE_STUDENT_BUILD_INFO, 'insert_ignore': INSERT_IGNORE_STUDENT_BUILD_INFO},
            BUILD_ERROR_STAT: {'create': CREATE_BUILD_ERROR_STAT, 'insert_ignore': INSERT_IGNORE_BUILD_ERROR_STAT}}