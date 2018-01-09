from scripts.scripts_util import scan_dir
import os

exam_ids = [36, 41, 46, 47, 52, 53, 64, 67, 68]

def filter_raw_zip_file(dirpath, exam_id_list):
    def remove_pattern(file_name:str):
        file_name_split = file_name.split('_')
        if int(file_name_split[0]) in exam_id_list and file_name[-4:] == '.zip':
            return False
        return True
    for file_path in scan_dir(dirpath, remove_pattern):
        print('remove file {}'.format(file_path))
        os.remove(file_path)


def remove_empety_dir(dirpath):
    files = os.listdir(dirpath)
    count = 0
    for file in files:
        file = os.path.join(dirpath, file)
        if os.path.isdir(file):
            count += remove_empety_dir(file)
            if len(os.listdir(file)) <= 0:
                print('remove file {}'.format(file))
                os.rmdir(file)
                count += 1
    return count


if __name__ == '__main__':
    filter_raw_zip_file(r'/home/lf/server_student_data/raw_data/2017', exam_ids)
    total = remove_empety_dir(r'/home/lf/server_student_data/raw_data/2017')
    print('total remove {}'.format(total))
