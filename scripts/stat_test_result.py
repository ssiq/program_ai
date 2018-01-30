from code_data.read_data import read_test_experiment_by_experiment_name
from code_data.constants import local_test_experiment_db


if __name__ == '__main__':
    experiment_name = 'final_iterative_model_using_common_error_without_iscontinue'
    # experiment_name = 'final_iterative_model_without_iscontinue'
    test_df = read_test_experiment_by_experiment_name(local_test_experiment_db, experiment_name)

    success_fn = lambda x: x['result'] == '1'
    success_df = test_df[test_df['result'] == '1'].copy()
    print('total {}, success {}'.format(len(test_df), len(success_df)))
