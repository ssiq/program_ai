import tensorflow as tf
import numpy as np

import util
from common import tf_util
from code_data.read_data import read_cpp_fake_code_set
from train.random_search import random_parameters_generator
from code_data.constants import char_sign_dict
from model.bi_rnn import BiRnnClassifyModel


def parse_xy(df):
    action_number = len(char_sign_dict)
    X = df['code']
    df['parsed_code'] = df['code'].map(util.get_sign_list)
    df = df[df['parsed_code'].map(lambda x: x is not None)]
    X = list(df['parsed_code'])
    y = list(action_number*df['actionpos'].astype(int)+df['actionsign'].astype(int))
    return X, y


def train_model(model_fn,
                train_data_iterator,
                validation_data,
                test_data,
                model_parameters,
                ):
    sess = tf_util.get_session()
    train_writer = tf.summary.FileWriter(
        logdir='./graphs/bi_rnn_on_fake_cpp_data_model/{}'.format(util.format_dict_to_string(model_parameters) + "_train"),
        graph=sess.graph)
    validation_writer = tf.summary.FileWriter(
        logdir='./graphs/bi_rnn_on_fake_cpp_data_model/{}'.format(util.format_dict_to_string(model_parameters) + "_validation"),
        graph=sess.graph)
    save_steps = 1000
    skip_steps = 100
    print_skip_step = 100
    losses = []
    accuracies = []
    model = model_fn(**model_parameters)
    saver = tf.train.Saver()
    validation_data_itr = validation_data()
    util.make_dir('checkpoints', 'bi_rnn_on_fake_cpp_data_model_{}'.format(util.format_dict_to_string(model_parameters)))
    for i, data in enumerate(train_data_iterator()):
        loss, accuracy, _ = model.train(*data)
        losses.append(loss)
        accuracies.append(accuracy)
        if i % skip_steps == 0:
            train_summary = model.summary(*data)
            train_writer.add_summary(train_summary, global_step=model.global_step)
            validation_summary = model.summary(*next(validation_data_itr))
            validation_writer.add_summary(validation_summary, global_step=model.global_step)
        if i % print_skip_step == 0:
            print("iteration {} with loss {} and accuracy {}".format(i, np.mean(losses), np.mean(accuracies)))
            losses = []
            accuracies = []
        if i % save_steps == 0:
            saver.save(sess, 'checkpoints/bi_rnn_on_fake_cpp_data_model_{}/bi_rnn'.format(util.format_dict_to_string(model_parameters)),
                       model.global_step)
    saver.save(sess,
               'checkpoints/bi_rnn_on_fake_cpp_data_model_{}/bi_rnn'.format(util.format_dict_to_string(model_parameters)),
               model.global_step)

    return np.mean([model.accuracy(*p) for p in test_data()])


if __name__ == '__main__':
    util.set_cuda_devices()
    train, test, vaild = read_cpp_fake_code_set()
    train_batch_iterator = util.batch_holder(*parse_xy(train), batch_size=4)
    # validation_data = util.dataset_holder(*list(map(util.padded_code, parse_xy(vaild))))
    # test_data = util.dataset_holder(*list(map(util.padded_code, parse_xy(test))))
    validation_data = util.batch_holder(*parse_xy(vaild), batch_size=4)
    test_data = util.batch_holder(*parse_xy(test), batch_size=4, epoches=1)
    param_generator = random_parameters_generator(random_param={"learning_rate": [-5, -1]},
                                                  choice_param={"state_size": [100]},
                                                  constant_param={"character_number": len(char_sign_dict),
                                                                  "action_number": len(char_sign_dict)})
    best_parameter = None
    best_accuracy = None
    for model_parm in param_generator(6):
        tf.reset_default_graph()
        with tf.Session():
            print("parameter:{}".format(util.format_dict_to_string(model_parm)))
            accuracy = train_model(BiRnnClassifyModel, train_batch_iterator, validation_data, test_data, model_parm)
            if best_accuracy is None:
                best_accuracy = accuracy
                best_parameter = model_parm
            elif best_accuracy < accuracy:
                best_accuracy = accuracy
                best_parameter = model_parm
            print("param:{}, accuracy:{}".format(util.format_dict_to_string(model_parm), accuracy))

    print("best accuracy:{}, best parameter:{}".format(best_accuracy, best_parameter))
