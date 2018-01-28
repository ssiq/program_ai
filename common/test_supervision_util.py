import tensorflow as tf

from common import tf_util, util


def restore_model(model_fn, model_params, experiment_name):
    model = model_fn(**model_params)
    sess = tf_util.get_session()
    restore_dir = 'checkpoints/{}_{}/'.format(experiment_name + '_model', util.format_dict_to_string(model_params))
    saver = tf.train.Saver()
    util.load_check_point(restore_dir, sess, saver)
    return model, sess


def create_test_experiment(test_df, parse_xy_fn, parse_params, experiment_name='default', batch_size=16, input_length=5):

    def test_model(model_fn, param_generator):

        input_list = test_data[0:input_length]
        output_list = test_data[input_length:]


        model_params = None
        for param in param_generator(1):
            model_params = param
            model_params['word_embedding_layer_fn'] = model_params['word_embedding_layer_fn']()
            model_params['character_embedding_layer_fn'] = model_params['character_embedding_layer_fn']()

        with tf_util.summary_scope():
            # tf.reset_default_graph()
            config = tf.ConfigProto()
            with tf.Session(config=config):
                with tf_util.summary_scope():
                    model, sess = restore_model(model_fn, model_params, experiment_name)

                    # input_list = [[] for i in range(input_length)]
                    # output_list = [[] for i in range(len(test_data)-input_length)]
                    predict_list = [[] for i in range(len(test_data)-input_length)]

                    for i, data in enumerate(test_batch_iterator()):
                        print('in {} enumerate'.format(i))
                        input_data = data[0:input_length]
                        output_data = data[input_length:]
                        print('{}: after get input and output'.format(i))
                        # predict_data = model.predict_model(*input_data, )
                        predict_data = [[[one_bat] for one_bat in out] for out in output_data]
                        print('{}: after predict'.format(i))

                        # input_list = [inp_list + inp_data for inp_list, inp_data in zip(input_list, input_data)]
                        # output_list = [out_list + out_data for out_list, out_data in zip(output_list, output_data)]
                        predict_list = [pre_list + pre_data for pre_list, pre_data in zip(predict_list, predict_data)]
        input_list = list(zip(*input_list))
        output_list = list(zip(*output_list))
        predict_list = list(zip(*predict_list))
        print(len(input_list))
        print(len(output_list))
        print(len(predict_list))
        return input_list, output_list, predict_list

    test_data = parse_xy_fn(test_df, 'test', *parse_params)
    print('test_data length: {}'.format(len(test_data[0])))
    test_batch_iterator = util.batch_holder_with_condition(*test_data, batch_size=batch_size, condition=None, shuffle=False, epoches=1)
    return test_model

if __name__ == '__main__':
    from code_data.read_data import read_local_test_code_records
    test_df = read_local_test_code_records()
    print(len(test_df))

