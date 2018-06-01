from code_data.constants import cache_data_path
from code_data.read_data import read_deepfix_records
from common.analyse_include_util import replace_include_with_blank
from common.c_code_tokenize import tokenize_by_clex_fn
from common.util import disk_cache
from scripts.scripts_util import init_code


def filter_frac(data_df, frac):
    user_count = len(data_df.groupby('user_id').size())
    print('user_count: {}'.format(user_count))
    user_id_list = data_df['user_id'].sample(int(user_count * frac)).tolist()
    print('user_id_list: {}'.format(len(user_id_list)))
    split_df = data_df[data_df.apply(lambda x: x['user_id'] in user_id_list, axis=1, raw=True)]
    print('split_df: {}'.format(len(split_df)))
    # drop_df = data_df[data_df.apply(lambda x: x['user_id'] in user_id_list, axis=1, raw=True)]
    main_df = data_df.drop(split_df.index)
    print('main_df: {}'.format(len(main_df)))
    return main_df, split_df


def filter_length(df, limit_length, tokenize_fn, key='similar_code'):
    df['tokens'] = df[key].map(tokenize_fn)
    df = df[df['tokens'].map(lambda x: x is not None and len(x) < limit_length)].copy()
    df = df.drop(columns=['tokens'], axis=1)
    return df


@disk_cache(basename='read_fake_common_c_error_dataset', directory=cache_data_path)
def read_fake_common_c_error_dataset():
    pass
    # test_df = read_distinct_problem_user_c_records()
    # test_df = test_df[test_df['distance'].map(lambda x: 0 < x < 10)]
    # data_df = read_distinct_problem_user_fake_c_common_records()
    # train_df, valid_df = filter_frac(data_df, 0.1)
    # print('train df size: {}, valid df size: {}, test df size: {}'.format(len(train_df), len(valid_df), len(test_df)))
    # return train_df, valid_df, test_df



@disk_cache(basename='read_fake_random_c_error_dataset', directory=cache_data_path)
def read_fake_random_c_error_dataset():
    pass
    # test_df = read_distinct_problem_user_c_records()
    # test_df = test_df[test_df['distance'].map(lambda x: 0 < x < 10)]
    # data_df = read_distinct_problem_user_fake_c_random_records()
    # train_df, valid_df = filter_frac(data_df, 0.1)
    # print('train df size: {}, valid df size: {}, test df size: {}'.format(len(train_df), len(valid_df), len(test_df)))
    # return train_df, valid_df, test_df


@disk_cache(basename='read_fake_common_c_error_dataset_with_limit_length', directory=cache_data_path)
def read_fake_common_c_error_dataset_with_limit_length(limit_length=500):
    dfs = read_fake_common_c_error_dataset()
    tokenize_fn = tokenize_by_clex_fn()

    train, valid, test = [filter_length(df, limit_length, tokenize_fn) for df in dfs]
    return train, valid, test


@disk_cache(basename='read_fake_random_c_error_dataset_with_limit_length', directory=cache_data_path)
def read_fake_random_c_error_dataset_with_limit_length(limit_length=500):
    dfs = read_fake_random_c_error_dataset()
    tokenize_fn = tokenize_by_clex_fn()

    train, valid, test = [filter_length(df, limit_length, tokenize_fn) for df in dfs]
    return train, valid, test


def read_deepfix_error_data():
    test_df = read_deepfix_records()
    test_df = test_df[test_df['errorcount'].map(lambda x: x > 0)]
    test_df['code'] = test_df['code'].map(init_code)
    # test_df['code'] = test_df['code'].map(replace_include_with_blank)
    print('original length: {}'.format(len(test_df)))
    return test_df


if __name__ == '__main__':
    tokenize_fn = tokenize_by_clex_fn()
    # train, valid, test = read_fake_random_c_error_dataset_with_limit_length(500)
    # _, _, test = read_fake_common_c_error_dataset_with_limit_length(500)
    # print(len(test))
    # test['similar_code'] = test['similar_code'].map(replace_include_with_blank)
    # test['res'] = test['similar_code'].map(tokenize_fn)
    # test = test[test['res'].map(lambda x: x is not None)]
    # print(len(test))
    #
    # test['code'] = test['code'].map(replace_include_with_blank)
    # test['res'] = test['code'].map(tokenize_fn)
    # test = test[test['res'].map(lambda x: x is not None)]
    # print(len(test))

    test_df = read_deepfix_error_data()
    print('test len: {}'.format(len(test_df)))

    test_df['tokens'] = test_df['code'].map(tokenize_fn)
    # failed_df = test_df[test_df['tokens'].map(lambda x: x is None)]
    test_df = test_df[test_df['tokens'].map(lambda x: x is not None)]
    print('before filter length: {}'.format(len(test_df)))
    test_df = test_df[test_df['tokens'].map(lambda x: len(x) < 500)].copy()
    print('after filter length: {}'.format(len(test_df)))

    test_df['length'] = test_df['tokens'].map(len)
    # print('after tokenize: {}'.format(len(test_df)))
    # print(failed_df.iloc[10]['code_id'])
    # print(failed_df.iloc[10]['code'])

    # print(test_df[test_df['code_id'].map(lambda x: x == 'prog18168')].iloc[0]['code'])
    # tokens = tokenize_fn(test_df[test_df['code_id'].map(lambda x: x == 'prog18168')].iloc[0]['code'])
    # print(tokens)


