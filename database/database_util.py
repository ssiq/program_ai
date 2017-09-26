import sqlite3
from code_data.constants import DATABASE_PATH, EPISODES, STEP_INFO, BACKUP_PATH
import os
import shutil
import time


def initSubmit():
    conn = sqlite3.connect(DATABASE_PATH)
    checkExist(conn)
    return conn


def close(conn):
    conn.close()


def checkExist(conn):
    com = '''CREATE TABLE IF NOT EXISTS '''+EPISODES+''' 
    (episodeid TEXT primary key,
    starttime TEXT, 
    endtime TEXT, 
    totalstep INTEGER default 0, 
    totalreward FLOAT default 0, 
    resolved INTEGER, 
    codeid TEXT, 
    originalcode TEXT,
    endcode TEXT)'''
    conn.execute(com)
    conn.commit()

    com = '''CREATE TABLE IF NOT EXISTS '''+STEP_INFO+''' 
    (id INTEGER primary key AUTOINCREMENT,
    episodeid TEXT, 
    stepid TEXT, 
    actionpos INTEGER,
    actioncha TEXT,
    reward FLOAT,
    done INTEGER)'''
    conn.execute(com)
    conn.commit()


def insertEpisodes(conn, episodeid, starttime, endtime, totalstep, totalreward, resolved, codeid, originalcode, endcode):
#    if not conn:
    conn = initSubmit()

    originalcode = originalcode.replace("'", "''")
    endcode = endcode.replace("'", "''")
    cmd = "INSERT OR IGNORE INTO "+EPISODES+" (episodeid, starttime, endtime, totalstep, totalreward, resolved, codeid, originalcode, endcode) " \
          "VALUES ('"+episodeid+"', '"+starttime+"', '"+endtime+"', "+totalstep+", "+totalreward+", "+resolved+", '"+codeid+"', '''"+originalcode+"''', '''"+endcode+"''')"
    conn.execute(cmd)
    conn.commit()
    conn.close()


def updateEpisodes(conn, episodeid, endtime, totalstep, totalreward, resolved, endcode):
#    if not conn:
    conn = initSubmit()
    endcode = endcode.replace("'", "''")
    cmd = "UPDATE "+EPISODES+" SET endtime = '"+endtime+"', totalstep = "+totalstep+", totalreward = "+totalreward+", resolved = "+resolved+", endcode = '''" + endcode + "''' WHERE episodeid = '" + episodeid + "' "
    conn.execute(cmd)
    conn.commit()
    conn.close()


def insertStepInfoMany(param):
    conn = initSubmit()

    cur = conn.cursor()
    print("len", len(param))
    cmd = 'INSERT OR IGNORE INTO '+STEP_INFO+' (episodeid, stepid, actionpos, actioncha, reward, done)' \
        'VALUES (?, ?, ?, ?, ?, ?) '
    cur.executemany(cmd, param)
    conn.commit()
    conn.close()


def find(conn, table_name, key, value):
#    if not conn:
    conn = initSubmit()

    cmd = "SELECT * FROM "+table_name+" WHERE "+key+"='"+value+"' "
    cur = conn.execute(cmd)
    return cur.fetchall()

def findIdExist(ids, table_name, key):
    conn = initSubmit()

    cur = conn.cursor()
    idstr = ",".join(ids)
    cmd = "SELECT "+key+" FROM "+ table_name+" WHERE "+key+" in ( "+ idstr + " )"
    cur.execute(cmd)
    return cur.fetchall()

def runCommand(conn, cmd):
    #if not conn:
    conn = initSubmit()

    return conn.execute(cmd)


def backup():
    if os.path.exists(DATABASE_PATH):
        ti = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        back_dir = os.path.join(BACKUP_PATH, ti)
        if not os.path.exists(back_dir):
            os.makedirs(back_dir)
        shutil.copyfile(DATABASE_PATH, os.path.join(back_dir, 'train.db'))
        os.remove(DATABASE_PATH)

