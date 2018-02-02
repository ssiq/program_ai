import json
import pandas as pd
import more_itertools
import cytoolz as toolz
import numpy as np
from toolz.sandbox.core import unzip
from collections import OrderedDict, Counter
import plotly.offline as py
import plotly.graph_objs as go

import os

from code_data.constants import local_test_experiment_finish_db
from code_data.read_data import read_test_experiment_by_experiment_name

experiment_name_list = ['final_iterative_model_using_common_error_without_iscontinue',
                   'one_iteration_token_level_multirnn_model_without_iscontinue',
                   'one_iteration_token_level_multirnn_model_using_common_error_without_iscontinue',
                   'one_iteration_token_level_multirnn_model_using_common_error_without_iscontinue_without_beam_search']


def read_data(experiment_name):
    db_path = local_test_experiment_finish_db
    return read_test_experiment_by_experiment_name(db_path, experiment_name)


def get_dm_list(df: pd.DataFrame):
    return [float(t) for t in df[df['success_id'] == "-1"]['min_distance']]


if __name__ == '__main__':
    print(os.getcwd())
    dm_list = [get_dm_list(read_data(name)) for name in experiment_name_list]
    dm_counter_list = [Counter(dm) for dm in dm_list]
    plot_data = [
        go.Bar(x=list(dm.keys()), y=[t/sum(dm.values()) for t in list(dm.values())],  name=name) for dm, name in zip(dm_counter_list, experiment_name_list)
    ]
    py.plot(
        {
            'data': plot_data,
        },
        image='svg', image_filename=os.path.join("summary", "dm_distribution.svg"),
        output_type='file',
        auto_open=False, image_width=800, image_height=600, filename=os.path.join("summary", "dm_distribution.html")
    )