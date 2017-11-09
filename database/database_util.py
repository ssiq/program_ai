import sqlite3

from code_data.constants import local_db_path
from database.sql_statment import sql_dict


def with_connect():
    def wrapper(func):
        def sub_wrapper(*args, **kwargs):
            db_path = None
            if len(args) > 0:
                db_path = args[0]
            if 'db_full_path' in kwargs.keys():
                db_path = kwargs['db_full_path']
            con = sqlite3.connect(db_path)
            kwargs['con'] = con
            res = func(*args, **kwargs)
            con.close()
            return res
        return sub_wrapper
    return wrapper

@with_connect()
def create_table(db_full_path, table_name, **kwargs):
    assert 'con' in kwargs.keys()

    sql = sql_dict[table_name]['create']
    con = kwargs['con']
    con.execute(sql)
    con.commit()

@with_connect()
def insert_items(db_full_path, table_name, params, **kwargs):
    assert 'con' in kwargs.keys()

    sql = sql_dict[table_name]['insert_ignore']
    con = kwargs['con']
    con.executemany(sql, params)
    con.commit()


@with_connect()
def find_ids_by_user_problem_id(db_full_path, table_name, ids, **kwargs):
    assert 'con' in kwargs.keys()

    sql = sql_dict[table_name]['find_ids_by_user_problem_id']
    con = kwargs['con']
    cur = con.cursor()
    cur.execute(sql.format(",".join(ids)))
    res = cur.fetchall()
    return res


if __name__ == '__main__':
    from code_data.constants import FAKE_CODE_RECORDS
    create_table(local_db_path, table_name=FAKE_CODE_RECORDS)
    # res = ['a', 'b', 'c', 'd', 'e', 'f', 2, 'h', 'i']
    # insert_items('db_path', table_name=FAKE_CODE_RECORDS, params=[res])
    find_ids_by_user_problem_id(local_db_path, FAKE_CODE_RECORDS, ids=['1', '2'])

