import typing

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

from code_data.preprocess import CharacterSet, StringSeq, IdList
from code_data.read_data import read_code_list
from model.lm import CharacterLanguageModel, build_model
from train.random_search import random_parameters_generator
import util

def get_right_C_code(df: pd.DataFrame) -> pd.Series:
    return (df.language == 1) & (df.status != 7)

def create_character_set(code_list: StringSeq) -> CharacterSet:
    character_set = CharacterSet(code_list)
    return character_set

def data(code_set:IdList, character_set: CharacterSet, batch_size:int=32, epoches:int=10) -> typing.Iterator[IdList]:
    for i in range(epoches):
        code_set = sklearn.utils.shuffle(code_set)
        for t in range(0, len(code_set), batch_size):
            yield character_set.align_texts_with_same_length(code_set[t:t+batch_size])

def train(model_parameters, train_data, val_data, character_set, epoches=15, skip_steps=5, print_skip_step=100, save_steps=1000):
    print("model_parameter:{}".format(model_parameters))
    model, train_op, X_input, global_step = build_model(**model_parameters)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    del model_parameters['Model']
    train_writer = tf.summary.FileWriter(logdir='./graphs/{}'.format(util.format_dict_to_string(model_parameters)+"_train"),
                                   graph=sess.graph)
    validation_writer = tf.summary.FileWriter(logdir='./graphs/{}'.format(util.format_dict_to_string(model_parameters)+"_validation"),
                                   graph=sess.graph)
    losses = []
    val_data = character_set.align_texts_with_same_length(val_data)
    for i, codes in enumerate(data(train_data, character_set, epoches=epoches)):
        # print('codes shape:{}'.format(np.array(codes).shape))
        _, loss = sess.run([train_op, model.loss_op], feed_dict={X_input: codes})
        losses.append(loss)
        if i % skip_steps == 0:
            summary = sess.run(model.summary_op, feed_dict={X_input: codes})
            train_writer.add_summary(summary, global_step=global_step.eval(sess))
            summary = sess.run(model.summary_op, feed_dict={X_input: val_data})
            validation_writer.add_summary(summary, global_step=global_step.eval(sess))
        if i % print_skip_step == 0:
            print("step {}: loss {}".format(i, np.mean(losses)))
            losses = []
        if i % save_steps == 0:
            saver.save(sess, 'checkpoints/rnn_language_model', global_step.eval(sess))

    train_writer.close()
    validation_writer.close()

if __name__ == '__main__':
    code_list = read_code_list(get_right_C_code, 100)
    character_set = create_character_set(code_list)
    param_generator = random_parameters_generator(random_param={'learning_rate': (-5, -2),
                                                                'max_grad_norm': (0, 2)},
                                                  choice_param={'hidden_size': (64, 128, 256),
                                                                'embedding_size': (128, 256)},
                                                  constant_param={'character_size': character_set.character_set_size,
                                                                  'Model': CharacterLanguageModel,
                                                                  })
    code_list = character_set.parse_text(code_list)
    train_data, val_data = train_test_split(code_list, shuffle=False, test_size=0.1)
    for param in param_generator(1):
        tf.reset_default_graph()
        train(param, train_data, val_data, character_set)




