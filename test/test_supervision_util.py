from experiment.bi_rnn_on_fake_cpp_data import parse_xy
from code_data.constants import char_sign_dict
from code_data.read_data import read_cpp_fake_code_set
from common import util
from model.bi_rnn import BiRnnClassifyModel
from train.random_search import random_parameters_generator

from common.supervision_util import create_supervision_experiment


if __name__ == '__main__':
    util.set_cuda_devices()
    train, test, vaild = read_cpp_fake_code_set()
    param_generator = random_parameters_generator(random_param={"learning_rate": [-5, 0]},
                                                  choice_param={"state_size": [100]},
                                                  constant_param={"character_number": len(char_sign_dict),
                                                                  "action_number": len(char_sign_dict)})

    exper_train = create_supervision_experiment(train, test, vaild, parse_xy, experiment_name='test_supervision', batch_size=8)
    exper_train(BiRnnClassifyModel, param_generator, generator_times=6)
