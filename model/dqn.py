import tensorflow as tf
import numpy as np
import typing
from model.lm import build_model, CharacterLanguageModel

from model import util

class DQNModel:

    def __init__(self, n_actions, n_features = 2, learning_rate=0.01, e_greedy=0.95, gamma=0.9, batch_size=10, memory_size=500, replace_network_iter = 20):
        self.n_actions = n_actions
        self.n_features = n_features

        self.observation = ""
        self.learning_rate = learning_rate
        self.e_greedy = e_greedy
        self.gamma = gamma

        self.replace_network_iter = replace_network_iter

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.memory_counter = 0
        self.learn_step_counter = 0

        network, train_op, x_input, global_step = build_model(learning_rate, 100, 100, 100, CharacterLanguageModel, 10)

        self.network = network
        self.network_train_op = train_op
        self.network_x_input = x_input
        self.network_global_step = global_step


        self.sess = tf.Session()
        tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())


    def choose_action(self, observation):

        pass

    def store_transition(self, observation, action, reward, observation_):
        if not self.memory_counter:
            self.memory_counter = 0
        transition = (observation, action, reward, observation_)
        index = self.learn_step_counter % self.memory_size
        self.memory[index:] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.replace_network_iter == 0:
            self.sess.run(self.replace_network_iter)

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        self.sess.run(self.network_train_op, feed_dict={self.network_x_input: batch_memory})

        self.learn_step_counter += 1

