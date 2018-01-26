from collections import Counter

from scripts.build_type_analyse import sort_dict_value
from scripts.scripts_util import get_student_test_set


def split_include(one):
    tokens = one['tokenize']
    inc_list = []
    for token in tokens:
        if isinstance(token.value, list):
            inc_str = ''.join(token.value)
            inc_str = inc_str.replace(' ', '')
            inc_list.append(inc_str)

    one['include_list'] = inc_list
    return one


def stat_include(build_df):
    total_list = []
    for inc in build_df['include_list']:
        total_list += inc

    cou_dict = dict(Counter(total_list))
    return cou_dict


def get_effective_student_test_set():
    from experiment.experiment_util import create_embedding, parse_xy_token_level_without_iscontinue, action_list_sorted
    import json
    key_val, char_voc = create_embedding()
    parse_xy_param = [key_val, char_voc, 5, 1, action_list_sorted]
    test_df = get_student_test_set()
    print('start: ', len(test_df.index))
    test_df['ac_code'] = test_df['similar_code']
    test_df['action_character_list'] = test_df['modify_action_list'].map(json.dumps)
    res = parse_xy_token_level_without_iscontinue(test_df, 'test', *parse_xy_param)
    print(len(res.index))
    return res

if __name__ == '__main__':
    # build_df = get_student_test_set()
    build_df = get_effective_student_test_set()
    from scripts.scripts_util import include_error_count, from_char_error_count
    print('include error: {}  from_char error: {}'.format(include_error_count, from_char_error_count))
    print(len(build_df.index))

    build_df = build_df.apply(split_include, raw=True, axis=1)

    print('------------- include stat -------------')
    inc_dict = stat_include(build_df)
    inc_sorted_list = sort_dict_value(inc_dict)
    for item in inc_sorted_list:
        print('{}, {}'.format(item[0], item[1]))
    print(len(inc_sorted_list))

    print('------------- error stat -------------')
    error_count_dict = {}
    for dis in build_df['distance']:
        key = str(dis)
        error_count_dict[key] = error_count_dict.get(key, 0) + 1
    print(error_count_dict)




