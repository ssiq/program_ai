from experiment.experiment_util import create_embedding, parse_xy_token_level

def stat_change_insert_delete_count(res, plh_id):
    position_df = res[6]
    is_copy_df = res[7]
    keyword_df = res[8]
    copy_df = res[9]
    total = 0
    ins_copy = 0
    ins_keyword = 0
    change_copy = 0
    change_keyword = 0
    delete = 0
    for rid in position_df.index:
        position_list = position_df.loc[rid]
        is_copy_list = is_copy_df.loc[rid]
        keyword_list = keyword_df.loc[rid]
        copy_list = copy_df.loc[rid]

        for i in range(len(position_list)):
            total += 1
            position = position_list[i]
            is_copy = is_copy_list[i]
            keyword = keyword_list[i]
            copy_id = copy_list[i]

            if position % 2 == 1 and is_copy == 0 and keyword == plh_id:
                delete += 1
            if position % 2 == 1 and is_copy == 0 and keyword != plh_id:
                change_keyword += 1
            if position % 2 == 1 and is_copy == 1:
                change_copy += 1
            if position % 2 == 0 and is_copy == 0 and keyword != plh_id:
                ins_keyword += 1
            if position % 2 == 0 and is_copy == 1:
                ins_copy += 1
    print(total, ins_keyword, ins_copy, change_keyword, change_copy, delete)
    return total, ins_keyword, ins_copy, change_keyword, change_copy, delete


if __name__ == '__main__':
    from code_data.read_data import read_cpp_random_token_code_records_set

    train, test, vaild = read_cpp_random_token_code_records_set()
    key_val, char_voc = create_embedding()
    parse_xy_param = [key_val, char_voc, 5, 1]
    plh_id = key_val.word_to_id(key_val.placeholder_label)
    res = parse_xy_token_level(train, 'train', *parse_xy_param)