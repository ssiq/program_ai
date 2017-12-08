import tensorflow as tf
import numpy as np
import gc

from common import util
from common import tf_util
import logging
from common.debug_tool import record_memory, show_growth, show_diff_length_fn


def create_supervision_experiment(train, test, vaild, parse_xy_fn, parse_xy_param, experiment_name='default', batch_size=32):
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

    def train_supervision(model_fn, param_generator, generator_times=6, debug=False):
        print('create supervision data start')
        best_parameter = None
        best_accuracy = None
        for model_parm in param_generator(generator_times):
            with tf_util.summary_scope():
                tf.reset_default_graph()
                model_parm['word_embedding_layer_fn'] = model_parm['word_embedding_layer_fn']()
                model_parm['character_embedding_layer_fn'] = model_parm['character_embedding_layer_fn']()

                print('create train model')
                train_model_fn = create_model_train_fn(model_fn, model_parm, debug)
                print('create train model finish')
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                with tf.Session(config=config):
                    print('in tf.session')
                    with tf_util.summary_scope():
                        print('train model')
                        accuracy = train_model_fn(train_batch_iterator, validation_data_iterator, test_data_iterator, experiment_name)
                    if best_accuracy is None:
                        best_accuracy = accuracy
                        best_parameter = model_parm
                    elif best_accuracy < accuracy:
                        best_accuracy = accuracy
                        best_parameter = model_parm
                    print("param:{}, accuracy:{}".format(util.format_dict_to_string(model_parm), accuracy))

        print("best accuracy:{}, best parameter:{}".format(best_accuracy, best_parameter))

    print('create supervision experiment.')
    train_data = parse_xy_fn(train, *parse_xy_param)
    vaild_data = parse_xy_fn(vaild, *parse_xy_param)
    test_data = parse_xy_fn(test, *parse_xy_param)
    print('train_data_length: {}'.format(len(train_data[0].index)))
    print('vaild_data_length: {}'.format(len(vaild_data[0].index)))
    print('test_data_length: {}'.format(len(test_data[0].index)))
    train_batch_iterator = util.batch_holder(*train_data, batch_size=batch_size)
    validation_data_iterator = util.batch_holder(*vaild_data, batch_size=batch_size, epoches=None)
    test_data_iterator = util.batch_holder(*test_data, batch_size=batch_size, epoches=1)

    return train_supervision


def create_model_train_fn(model_fn, model_parameters, debug=False):
    '''
    create a model with special paramenters.
    :param model_fn: Model class
    :param model_parameters: special model parameters
    :return: the train method of the model
    '''

    def train_model(train_data_iterator,
                    validation_data_iterator,
                    test_data_iterator,
                    experiment_name='default'
                    ):
        print("parameter:{}".format(util.format_dict_to_string(model_parameters)))
        model = model_fn(**model_parameters)
        sess = tf_util.get_session()
        train_writer = tf.summary.FileWriter(
            logdir='./graphs/{}/{}'.format(
                experiment_name+'_model', util.format_dict_to_string(model_parameters) + "_train"),
                # experiment_name+'_model',  "_train"),
            graph=sess.graph)
        validation_writer = tf.summary.FileWriter(
            logdir='./graphs/{}/{}'.format(
                experiment_name + '_model', util.format_dict_to_string(model_parameters) + "_validation"),
                # experiment_name + '_model',  "_validation"),
            graph=sess.graph)
        save_steps = 1000
        skip_steps = 100
        print_skip_step = 100
        debug_steps = 10

        recordloggername = 'record'
        growthloggername = 'growth'
        objectlengthloggername = 'objectlength'
        show_diff_length = show_diff_length_fn(recordloggername, objectlengthloggername)

        losses = []
        accuracies = []
        saver = tf.train.Saver()
        validation_data_itr = validation_data_iterator()
        util.make_dir('checkpoints', '{}_{}'.format(
            experiment_name + '_model', util.format_dict_to_string(model_parameters)))
        for i, data in enumerate(train_data_iterator()):
            try:
                # log_data_shape(*data, recordloggername=recordloggername)
                loss, metrics, _ = model.train_model(*data)
                losses.append(loss)
                accuracies.append(metrics)
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
                    saver.save(sess, 'checkpoints/{}_{}/bi_rnn'.format(
                        experiment_name + '_model', util.format_dict_to_string(model_parameters)),
                               model.global_step)
                if i % debug_steps == 0 and debug:
                    record_memory(recordloggername)
                    show_growth(recordloggername, growthloggername)
                    show_diff_length(i=i)
            except Exception as e:
                import traceback
                mess = traceback.format_exc()
                print(mess)
                logging.error(mess)
        saver.save(sess, 'checkpoints/{}_{}/bi_rnn'.format(
                    experiment_name + '_model', util.format_dict_to_string(model_parameters)),
                   model.global_step)

        return np.mean([model.metrics_model(*p) for p in test_data_iterator()])

    return train_model


def log_data_shape(*data, recordloggername):
    import logging
    logger = logging.getLogger(recordloggername)
    for d in data:
        arr = np.array(d)
        logger.debug('data shape: {}'.format(arr.shape))
