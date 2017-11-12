from code_data.constants import FAKE_CODE_RECORDS

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

INSERT_IGNORE_FAKE_CODE_RECORDS = r'''INSERT OR IGNORE INTO fake_code_records (id, submit_id, problem_id, user_id, ac_code, code, error_count, error_character_maplist, action_character_list) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'''

FIND_IDS_BY_USER_PROBLEM_ID = r'''SELECT id FROM fake_code_records WHERE id in ( {} )'''

sql_dict = {FAKE_CODE_RECORDS: {'create': CREATE_FAKE_CODE_RECORDS, 'insert_ignore': INSERT_IGNORE_FAKE_CODE_RECORDS,
                                'find_ids_by_user_problem_id': FIND_IDS_BY_USER_PROBLEM_ID}}