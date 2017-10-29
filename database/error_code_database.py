import sqlite3
from code_data.constants import FAKE_ERROR_CODE, local_db_path, FAKE_ERROR_TOKEN_CODE

def initFakeCodeTable():
    conn = sqlite3.connect(local_db_path)
    checkExist(conn)
    return conn


def initFakeTokenCodeTable():
    conn = sqlite3.connect(local_db_path)
    checkTokenExist(conn)
    return conn


def close(conn):
    conn.close()


def checkExist(conn):
    com = '''CREATE TABLE IF NOT EXISTS '''+FAKE_ERROR_CODE+''' 
    (id TEXT primary key,
    submitid TEXT, 
    problemid TEXT, 
    userid TEXT, 
    originalcode TEXT, 
    code TEXT, 
    errorcount INTEGER, 
    actiontype INTEGER, 
    actionpos TEXT,
    actionsign TEXT)'''
    conn.execute(com)
    conn.commit()


def checkTokenExist(conn):
    com = '''CREATE TABLE IF NOT EXISTS '''+FAKE_ERROR_TOKEN_CODE+''' 
    (id TEXT primary key,
    submitid TEXT, 
    problemid TEXT, 
    userid TEXT, 
    originalcode TEXT, 
    code TEXT, 
    errorcount INTEGER, 
    actiontype INTEGER, 
    actionpos TEXT,
    actionsign TEXT)'''
    conn.execute(com)
    conn.commit()



def insert_fake_error(param):
    conn = initFakeCodeTable()
    cur = conn.cursor()
    cmd = 'INSERT OR IGNORE INTO ' + FAKE_ERROR_CODE + ' (id, submitid, problemid, userid, originalcode, code, errorcount, actiontype, actionpos, actionsign)' \
                                                 'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) '
    cur.executemany(cmd, param)
    conn.commit()
    conn.close()


def insert_fake_token_error(param):
    conn = initFakeTokenCodeTable()
    cur = conn.cursor()
    cmd = 'INSERT OR IGNORE INTO ' + FAKE_ERROR_TOKEN_CODE + ' (id, submitid, problemid, userid, originalcode, code, errorcount, actiontype, actionpos, actionsign)' \
                                                 'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) '
    cur.executemany(cmd, param)
    conn.commit()
    conn.close()


def find_submit_by_problem_user(param_problem_user):
    conn = initFakeCodeTable()
    cur = conn.cursor()
    cmd = 'SELECT id FROM ' + FAKE_ERROR_CODE + ' WHERE id in ( '+ ",".join(param_problem_user) +' )'
    cur.execute(cmd)
    res = cur.fetchall()
    conn.close()
    return res


def find_token_submit_by_problem_user(param_problem_user):
    conn = initFakeTokenCodeTable()
    cur = conn.cursor()
    cmd = 'SELECT id FROM ' + FAKE_ERROR_TOKEN_CODE + ' WHERE id in ( '+ ",".join(param_problem_user) +' )'
    cur.execute(cmd)
    res = cur.fetchall()
    conn.close()
    return res
