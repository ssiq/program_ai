import os
import zipfile
from scripts.scripts_util import scan_dir

def unZip(dir, file, target_dir, zip_pattern=None):
    path = os.path.join(dir, file)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    f = zipfile.ZipFile(path, 'r')

    for filename in f.namelist():
        if zip_pattern and zip_pattern(filename):
            f.extract(filename, target_dir)


def unZipData(datadir, target_dir, pattern=None, zip_pattern=None, level=-1):
    for file in scan_dir(datadir, pattern, level):
        if os.path.isfile(file):
            (filepath, tempfilename) = os.path.split(file)
            (shortname, extension) = os.path.splitext(tempfilename)
            rel_path = os.path.relpath(filepath, datadir)
            if extension == '.zip':
                tar_dir = os.path.join(target_dir, rel_path)
                tar_dir = os.path.join(tar_dir, shortname)
                print('unzip file {} {} to {}'.format(filepath, tempfilename, tar_dir))
                try:
                    unZip(filepath, tempfilename, tar_dir, zip_pattern)
                except Exception as e:
                    print('unzip file failed {} {}'.format(filepath, tempfilename))


def monitor_pattern(file_name:str):
    strs = file_name.split('_')
    if strs[1] == 'monitor':
        return True
    return False


def zip_pattern(file_name:str):
    if file_name[0:4] == 'File':
        return False
    return True

if __name__ == '__main__':
    data_root_path = r'/home/lf/server_student_data/raw_data/2017/'
    target_root_path = r'/home/lf/server_student_data/unzip_data/monitor/2017/'
    unZipData(data_root_path, target_root_path, monitor_pattern, zip_pattern)
