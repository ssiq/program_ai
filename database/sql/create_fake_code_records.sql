CREATE TABLE IF NOT EXISTS fake_code_records (
  id TEXT PRIMARY KEY,
  submit_id TEXT,
  problem_id TEXT,
  user_id TEXT,
  ac_code TEXT,
  code TEXT,
  error_count INTEGER,
  error_character_maplist TEXT,
  action_character_list TEXT
)