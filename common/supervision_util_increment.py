import tensorflow as tf
import numpy as np
import gc

from common import util
from common import tf_util
import logging
from common.debug_tool import record_memory, show_growth, show_diff_length_fn
from common.iterator_condition import Condition


def create_supervision_experiment(load_data_fn, load_data_param, experiment_name='default', batch_size=32, create_condition_fn=None, modify_condition=[]):
    '''
    create a supervision experiment method. you can train supervision in model by calling the returned methon.
    :param parse_xy_fn: parse data to (X, y) tunple method
    :param experiment_name:  experiment name
    :param batch_size: batch size
    :return: train method. you can call the method to start a supervision experiment
    '''

    def train_supervision(model_fn, param_generator, generator_times=6, debug=False, restore=None):
        print('create supervision data start')
        best_parameter = None
        best_accuracy = None
        for model_parm in param_generator(generator_times):
            with tf_util.summary_scope():
                tf.reset_default_graph()
                model_parm['word_embedding_layer_fn'] = model_parm['word_embedding_layer_fn']()
                model_parm['character_embedding_layer_fn'] = model_parm['character_embedding_layer_fn']()

                change_count = 0
                condition_train_fn = create_condition_fn(**modify_condition[change_count][0])
                condition = Condition(condition_train_fn)
                print('start condition: {}'.format(str(modify_condition[change_count][0])))
                flat_train_batch_iterator = util.batch_holder_with_condition(*flat_train_data, batch_size=batch_size, condition=None)
                train_batch_iterator = util.batch_holder_with_condition(*train_data, batch_size=batch_size, condition=condition, epoches=None)
                validation_data_iterator = util.batch_holder_with_condition(*vaild_data, batch_size=batch_size, epoches=None, condition=condition)
                test_data_iterator = util.batch_holder_with_condition(*test_data, batch_size=batch_size, epoches=1, condition=condition)

                print('create train model')
                train_model_fn = create_model_train_fn(model_fn, model_parm, debug, restore)
                print('create train model finish')
                config = tf.ConfigProto()
                # config.gpu_options.allow_growth = True
                with tf.Session(config=config):
                    print('in tf.session')
                    with tf_util.summary_scope():
                        print('train model')
                        param_metrics = None
                        for metrics in train_model_fn(flat_train_batch_iterator, train_batch_iterator, validation_data_iterator, test_data_iterator, experiment_name):
                            param_metrics = metrics

                            if metrics > modify_condition[change_count][1] and (len(modify_condition)-1) > change_count:
                                change_count += 1
                                print('condition modified to {}, metrics condition from {} to {} with metrics {}'.format(str(modify_condition[change_count][0]), modify_condition[change_count-1][1], modify_condition[change_count][1], metrics))
                                condition.condition_fn = create_condition_fn(**modify_condition[change_count][0])
                                condition.modify = True

                            if best_accuracy is None:
                                best_accuracy = metrics
                                best_parameter = model_parm
                            elif best_accuracy < metrics:
                                best_accuracy = metrics
                                best_parameter = model_parm
                        print("param:{}, accuracy:{}".format(util.format_dict_to_string(model_parm), param_metrics))

        print("best accuracy:{}, best parameter:{}".format(best_accuracy, best_parameter))

    print("begin load_data")
    # train_data, test_data, vaild_data = load_data_fn(*load_data_param)
    loaded_data = load_data_fn(*load_data_param)
    if len(loaded_data) == 3:
        train_data, test_data, vaild_data = loaded_data
        flat_train_data = train_data
    else:
        flat_train_data, train_data, test_data, vaild_data = loaded_data

    print('flat_train_data_length: {}'.format(len(flat_train_data[0])))
    print('train_data_length: {}'.format(len(train_data[0])))
    print('vaild_data_length: {}'.format(len(vaild_data[0])))
    print('test_data_length: {}'.format(len(test_data[0])))

    return train_supervision


def create_model_train_fn(model_fn, model_parameters, debug=False, restore=None):
    '''
    create a model with special paramenters.
    :param model_fn: Model class
    :param model_parameters: special model parameters
    :return: the train method of the model
    '''

    def train_model(flat_train_batch_iterator,
                    train_data_iterator,
                    validation_data_iterator,
                    test_data_iterator,
                    experiment_name='default',
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
        skip_steps = 10
        print_skip_step = 100
        debug_steps = 10
        metrics_steps = 10

        recordloggername = 'record'
        growthloggername = 'growth'
        objectlengthloggername = 'objectlength'
        show_diff_length = show_diff_length_fn(recordloggername, objectlengthloggername)

        losses = []
        accuracies = []
        metrics_list = []
        saver = tf.train.Saver()
        if restore:
            restore_dir = 'checkpoints/{}_{}/'.format(experiment_name + '_model', util.format_dict_to_string(model_parameters))
            util.load_check_point(restore_dir, sess, saver)
            print('model restore from {}'.format(restore_dir))
        validation_data_itr = validation_data_iterator()
        train_data_itr = train_data_iterator()
        util.make_dir('checkpoints', '{}_{}'.format(
            experiment_name + '_model', util.format_dict_to_string(model_parameters)))
        print('start train enumerate')
        for i, data in enumerate(flat_train_batch_iterator()):
            try:
                current_step = model.global_step
                # log_data_shape(*data, recordloggername=recordloggername)
                loss, accuracy, _ = model.train_model(*data)
                accuracies.append(accuracy)
                losses.append(loss)
                # accuracies.append(metrics)
                # print("iteration {} with loss {} and metrics {}".format(current_step, loss, metrics))
                if current_step % metrics_steps == 0:
                    metrics = model.metrics_model(*next(train_data_itr))
                    # metrics = model.metrics_model(*next(validation_data_itr))
                    metrics_list.append(metrics)
                if current_step % skip_steps == 0:
                    train_summary = model.summary(*next(train_data_itr))
                    train_writer.add_summary(train_summary, global_step=model.global_step)
                    validation_summary = model.summary(*next(validation_data_itr))
                    validation_writer.add_summary(validation_summary, global_step=model.global_step)
                    pass
                if current_step % print_skip_step == 0:
                    loss_mean = np.mean(losses)
                    accuracy_mean = np.mean(accuracies)
                    metrics_mean = np.mean(metrics_list)
                    valid = model.metrics_model(*next(validation_data_itr))
                    print("iteration {} with loss {} and accuracy {} and metrics {} and validation metrics {}".format(current_step, loss_mean, accuracy_mean, metrics_mean, valid))
                    yield metrics_mean
                    losses = []
                    accuracies = []
                if current_step % save_steps == 0:
                    saver.save(sess, 'checkpoints/{}_{}/bi_rnn'.format(
                        experiment_name + '_model', util.format_dict_to_string(model_parameters)),
                               model.global_step)
                if current_step % debug_steps == 0 and debug:
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
