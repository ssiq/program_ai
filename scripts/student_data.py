import sqlite3
import pandas as pd
import re
import json
import os
import cytoolz
import more_itertools

from code_data.constants import local_student_db_path, STUDENT_BUILD_INFO
from database.database_util import insert_items, create_table
from scripts.scripts_util import scan_dir


def read_build_and_text_info(conn:sqlite3.Connection):
    build_infos = pd.read_sql('select * from {}'.format('build_project_info'), conn)
    text_infos = pd.read_sql('select * from {}'.format('command_text'), conn)
    build_infos['time'] = pd.to_datetime(build_infos['time'])
    # build_infos['buildid'] = pd.to_datetime(build_infos['buildid'])
    # build_infos['buildstarttime'] = pd.to_datetime(build_infos['buildstarttime'])
    # build_infos['buildendtime'] = pd.to_datetime(build_infos['buildendtime'])
    build_infos['type'] = 'build'
    build_infos = build_infos.sort_values(['time'])

    text_infos['time'] = pd.to_datetime(text_infos['time'])
    text_infos = text_infos[text_infos['action'].map(lambda x: x == 'Save')].copy()
    text_infos['type'] = 'text'
    text_infos = text_infos.sort_values(['time'])
    return build_infos, text_infos


def do_merge_student_code(dir_path):
    pattern = lambda x: x[-6:] == 'log.db'
    build_list = []
    create_table(db_full_path=local_student_db_path, table_name=STUDENT_BUILD_INFO)

    file_count = 0
    for db_path in scan_dir(dir_path, pattern):
        file_count += 1
        print('merge db file {}, total build file {}'.format(db_path, file_count))
        # one_list = merge_build_text_info(db_path, local_student_db_path)
        build_list = build_list + merge_build_text_info(db_path)
        print('total build list {}'.format(len(build_list)))
    build_list = cytoolz.unique(build_list, key=lambda x: x[-1])

    total = 0
    for data in more_itertools.chunked(build_list, 10000):
        total += len(data)
        print('save {} data. total {}'.format(len(data), total))
        insert_items(db_full_path=local_student_db_path, table_name=STUDENT_BUILD_INFO, params=data)


def merge_build_text_info(db_path):
    conn = sqlite3.connect("file:{}?mode=ro".format(db_path), uri=True)
    dir_name = os.path.basename(os.path.dirname(db_path))
    try:
        build_df, text_df = read_build_and_text_info(conn)
    except Exception as e:
        return []

    if len(build_df.index) <= 0 or len(text_df.index) <= 0:
        return []

    id_fn = lambda x: dir_name + '_' + str(x)
    build_df['id'] = build_df['id'].map(id_fn)

    build_df = filter_build_records(build_df)
    if len(build_df.index) <= 0:
        return []

    build_list = find_closest_file(build_df, text_df)
    build_list = dict_to_list(build_list)

    return build_list


def filter_build_records(build_df:pd.DataFrame):

    build_df['files'] = build_df['buildlogcontent'].map(deal_files)
    build_df = build_df[build_df['files'].map(lambda x: len(x) == 1)].copy()
    if len(build_df.index) <= 0:
        return build_df

    # build_df['compile_command'] = build_df['content'].map(deal_build_command)
    build_df = build_df[build_df['compilercommand'].map(lambda x: x != '' and x != None)].copy()
    if len(build_df.index) <= 0:
        return build_df

    build_df['error_infos'] = build_df['buildlogcontent'].map(deal_error)
    build_df['build_result'] = build_df['buildlogcontent'].map(deal_result)

    return build_df


def dict_to_list(build_dict_lists):

    def item_to_list(item):
        return item['id'], item['time'].strftime("%Y-%m-%d %H:%M:%S"), item['buildstarttime'], \
                item['buildendtime'], item['solutionname'], item['projectname'], \
                item['buildlogcontent'], item['compilercommand'], json.dumps(item['files']), \
                json.dumps(item['error_infos']), item['build_result'], item['file_content']

    build_list = [list(item_to_list(di)) for di in build_dict_lists]
    return build_list

def find_closest_file(build_df:pd.DataFrame, text_df:pd.DataFrame):
    build_list = build_df.to_dict('records')
    text_list = text_df.to_dict('records')
    merge_data = build_list + text_list
    merge_data = sorted(merge_data, key=lambda x: x['time'])

    projects = {}
    build_res = []
    for rec in merge_data:
        if rec['type'] == 'text':
            project_name = rec['project']
            file_name = rec['name']
            code = rec['content']

            if project_name not in projects.keys():
                projects[project_name] = {}
            if file_name not in projects[project_name].keys():
                projects[project_name][file_name] = ''
            projects[project_name][file_name] = code

        if rec['type'] == 'build':
            files = rec['files']
            project_name = rec['projectname']

            rec['file_content'] = None
            if project_name not in projects.keys():
                continue
            file_name = None
            if len(files) == 1:
                file_name = files[0]
            if file_name and file_name in projects[project_name].keys():
                rec['file_content'] = projects[project_name][file_name]
            if rec['file_content']:
                build_res.append(rec)
    return build_res



# ------------------- util ------------------ #

def deal_build_command(build_output):
    lines = build_output.split("\n")
    for temp in lines:
        temp = temp.strip()
        temps = temp.split(">")
        if len(temps) > 1:
            line = temp[2:]
        else:
            line = temp
        pattern = re.compile(r"^cl(.*)\.cpp\"?$")
        match = pattern.match(line)
        if match != None:
            return line
    return None


def deal_error(content):
    errs = []
    lines=content.split("\n")
    for temp in lines:
        temp=temp.strip()
        temps=temp.split(">")
        if len(temps)>1:
            line=temp[2:]
        else:
            line=temp
        pattern=re.compile(r"^(.*): (fatal |)error (\w*): (.*)$")
        match=pattern.search(line)
        if match:
            position=match.group(1)
            code=match.group(3)
            message=match.group(4)
            err = {'line': line, 'position': position, 'code': code, 'message': message}
            errs.append(err)
            #print(position+"\t"+code+"\t"+message)
    return errs


def deal_files(content):
    files = []
    lines = content.split("\n")
    for temp in lines:
        temp = temp.strip()
        temps = temp.split(">")
        if len(temps) > 1:
            line = temp[2:]
        else:
            line = temp
        pattern = re.compile(r"^cl(.*)\.cpp\"?$")
        match = pattern.search(line)
        if match:
            keylist = match.group(0).split(" ")
            for keyname in keylist:
                if keyname[0:1] == '"':
                    keyname = keyname[1:len(keyname)-1]
                    #print(keyname)
                if keyname[len(keyname)-4:len(keyname)] != '.cpp':
                    continue
                files.append(keyname)
    return files


def deal_result(content):
    lines=content.split("\n")
    for temp in lines:
        temp=temp.strip()
        temps=temp.split(">")
        if len(temps)>1:
            line=temp[2:]
        else:
            line=temp
        pattern = re.compile(r"^(Build succeeded|生成成功).*")
        match = pattern.search(line)
        if match:
            return 1
        pattern = re.compile(r"^(Build FAILED|生成失败).*")
        match = pattern.search(line)
        if match:
            return 0
    return 0



if __name__ == '__main__':
    # code_path = r'/home/lf/server_student_data/unzip_data/monitor/2017/10/25/41_monitor_16X5VDN'
    code_path = r'/home/lf/server_student_data/unzip_data/monitor/2017/'
    do_merge_student_code(code_path)


