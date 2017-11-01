import tensorflow as tf
import numpy as np

from common import util
from common import tf_util


def create_supervision_experiment(train, test, vaild, parse_xy_fn, batch_complete_fn, experiment_name='default', batch_size=64):
    '''
    create a supervision experiment method. you can train supervision in model by calling the returned methon.
    :param train: all of train data
    :param test: all of test data
    :param vaild: all of vaild data
    :param parse_xy_fn: parse data to (X, y) tunple method
    :param experiment_name:  experiment name
    :param batch_size: batch size
    :return: train method. you can call the method to start a supervision experiment
    '''

    def train_supervision(model_fn, param_generator, generator_times=6):
        best_parameter = None
        best_accuracy = None
        for model_parm in param_generator(generator_times):
            tf.reset_default_graph()
            train_model_fn = create_model_train_fn(model_fn, model_parm)
            with tf.Session():
                accuracy = train_model_fn(train_batch_iterator, validation_data_iterator, test_data_iterator, batch_complete_fn, experiment_name)
                if best_accuracy is None:
                    best_accuracy = accuracy
                    best_parameter = model_parm
                elif best_accuracy < accuracy:
                    best_accuracy = accuracy
                    best_parameter = model_parm
                print("param:{}, accuracy:{}".format(util.format_dict_to_string(model_parm), accuracy))

        print("best accuracy:{}, best parameter:{}".format(best_accuracy, best_parameter))

    train_batch_iterator = util.batch_holder(*parse_xy_fn(train), batch_size=batch_size)
    validation_data_iterator = util.batch_holder(*parse_xy_fn(vaild), batch_size=batch_size)
    test_data_iterator = util.batch_holder(*parse_xy_fn(test), batch_size=batch_size, epoches=1)

    return train_supervision


def create_model_train_fn(model_fn, model_parameters):
    '''
    create a model with special paramenters.
    :param model_fn: Model class
    :param model_parameters: special model parameters
    :return: the train method of the model
    '''

    def train_model(train_data_iterator,
                    validation_data_iterator,
                    test_data_iterator,
                    batch_complete_fn,
                    experiment_name='default'
                    ):
        print("parameter:{}".format(util.format_dict_to_string(model_parameters)))
        model = model_fn(**model_parameters)
        sess = tf_util.get_session()
        train_writer = tf.summary.FileWriter(
            logdir='./graphs/{}/{}'.format(
                experiment_name+'_model', util.format_dict_to_string(model_parameters) + "_train"),
            graph=sess.graph)
        validation_writer = tf.summary.FileWriter(
            logdir='./graphs/{}/{}'.format(
                experiment_name + '_model', util.format_dict_to_string(model_parameters) + "_validation"),
            graph=sess.graph)
        save_steps = 1000
        skip_steps = 100
        print_skip_step = 100
        losses = []
        accuracies = []
        saver = tf.train.Saver()
        validation_data_itr = validation_data_iterator()
        util.make_dir('checkpoints', '{}_{}'.format(
            experiment_name + '_model', util.format_dict_to_string(model_parameters)))
        for i, data in enumerate(train_data_iterator()):
            data = batch_complete_fn(*data)
            loss, accuracy, _ = model.train(*data)
            losses.append(loss)
            accuracies.append(accuracy)
            if i % skip_steps == 0:
                train_summary = model.summary(*data)
                train_writer.add_summary(train_summary, global_step=model.global_step)
                validation_summary = model.summary(*batch_complete_fn(*next(validation_data_itr)))
                validation_writer.add_summary(validation_summary, global_step=model.global_step)
            if i % print_skip_step == 0:
                print("iteration {} with loss {} and accuracy {}".format(i, np.mean(losses), np.mean(accuracies)))
                losses = []
                accuracies = []
            if i % save_steps == 0:
                saver.save(sess, 'checkpoints/{}_{}/bi_rnn'.format(
                    experiment_name + '_model', util.format_dict_to_string(model_parameters)),
                           model.global_step)
        saver.save(sess, 'checkpoints/{}_{}/bi_rnn'.format(
                    experiment_name + '_model', util.format_dict_to_string(model_parameters)),
                   model.global_step)

        return np.mean([model.accuracy(*batch_complete_fn(*p)) for p in test_data_iterator()])

    return train_model
