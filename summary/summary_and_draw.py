import json
import pandas as pd
import more_itertools
import cytoolz as toolz
import numpy as np
from toolz.sandbox.core import unzip
from collections import OrderedDict
import plotly.offline as py
import plotly.graph_objs as go

import os

from code_data.constants import local_test_experiment_db
from code_data.read_data import read_test_experiment_by_experiment_name

experiment_name = 'final_iterative_model_using_common_error_without_iscontinue'
# experiment_name = 'final_iterative_model_without_iscontinue'
# experiment_name = 'one_iteration_token_level_multirnn_model_without_iscontinue'
# experiment_name = 'one_iteration_token_level_multirnn_model_using_common_error_without_iscontinue'


def read_data():
    db_path = local_test_experiment_db
    return read_test_experiment_by_experiment_name(db_path, experiment_name)


def plot_error_and_success_rate(df: pd.DataFrame, accuracy_fn, name):
    data = [dict(t) for (_, t) in df.iterrows()]
    error_tuple_list = more_itertools.flatten([[(t['message'], d) for t in json.loads(d["build_error_info"])] for d in data])
    error_tuple_list = toolz.groupby(lambda x: x[0], error_tuple_list)

    def key_map_fn(code_list):
        code_list = list(code_list)
        distance_l = [float(t[1]['min_distance']) for t in filter(lambda x: x[1]['success_id'] == '-1', code_list)]
        return np.mean([accuracy_fn(t[1]) for t in code_list]), \
               np.mean(distance_l) if distance_l else "No failed",\
               len(code_list)

    error_tuple_dict = toolz.valmap(key_map_fn, error_tuple_list)
    error_list = list(error_tuple_dict.keys())
    accuracy_list, distance_list, length_list = [list(t) for t in unzip(list(error_tuple_dict.values()))]
    out_df = pd.DataFrame(
        OrderedDict({
            "error": error_list,
            "accuracy": accuracy_list,
            "distance": distance_list,
            "number": length_list
        })
    )
    out_df.to_csv(os.path.join("summary", "graph_and_csv", "error_type", "{}.csv".format(name)))

    # plot_data = [
    #     go.Bar(
    #         x=list(error_tuple_list.keys()),
    #         y=list(error_tuple_list.values())
    #     )
    # ]
    # py.plot(
    #     {
    #         'data': plot_data,
    #     },
    #     image='svg', image_filename=name,
    #     output_type='file',
    #     auto_open=False, image_width=800, image_height=600,
    # )


def exact_accuracy(d):
    return float(d['result'])


def plot_accuracy_with_different_length(df: pd.DataFrame, accuracy_fn, name):
    data = [dict(t) for (_, t) in df.iterrows()]
    # different_length_accuracy = [(len(json.loads(d['output_list'])[0]), d) for d in data]
    different_length_accuracy = [(0, d) for d in data]
    different_length_accuracy = toolz.groupby(lambda x: x[0], different_length_accuracy)

    def key_map_fn(code_list):
        code_list = list(code_list)
        distance_l = [float(t[1]['min_distance']) for t in filter(lambda x: x[1]['success_id'] == '-1', code_list)]
        return np.mean([accuracy_fn(t[1]) for t in code_list]), \
               np.mean(distance_l) if distance_l else "No failed"

    error_tuple_dict = toolz.valmap(key_map_fn, different_length_accuracy)
    out_df = pd.DataFrame(
        OrderedDict(
            {
                "error_number": list(error_tuple_dict.keys()),
                "accuracy": list(error_tuple_dict.values())
            }
        )
    )
    out_df.to_csv(os.path.join("summary", "graph_and_csv", "action_number", "{}.csv".format(name)))


if __name__ == '__main__':
    df = read_data()
    print(df.head(10))
    print()
    print(os.getcwd())
    plot_error_and_success_rate(df, exact_accuracy, "{}_em_accuracy".format(experiment_name))
    plot_accuracy_with_different_length(df, exact_accuracy, "{}_em_accuracy".format(experiment_name))
