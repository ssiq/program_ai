import os

import numpy as np
import tensorflow as tf

from code_data.constants import char_sign_dict, LOG_PATH, CHECKPOINT_PATH
from model.bi_rnn import build_bi_rnn
from common.tf_util import get_shape


class DQNModel(object):

    @staticmethod
    def init_para():

        flags = tf.app.flags
        flags.DEFINE_float('dqn_learning_rate', 0.001, '')
        flags.DEFINE_integer('dqn_e_greedy_start', 1, '')
        flags.DEFINE_integer('dqn_e_greedy_end', 0.0, '')
        flags.DEFINE_integer("dqn_e_greedy_update_iter", 50, '')
        flags.DEFINE_float('dqn_e_greedy_update_step', 0.001, '')
        flags.DEFINE_float('dqn_gamma', 0.75, '')
        flags.DEFINE_integer('dqn_batch_size', 32, '')
        flags.DEFINE_integer('dqn_memory_size', 50000, '')
        flags.DEFINE_integer('dqn_replace_network_iter', 50, '')
        flags.DEFINE_integer('dqn_no_train_step', 10, '')
        flags.DEFINE_integer('char_dict_len', 97, '')
        flags.DEFINE_integer('hidden_state', 64, '')

        flags.DEFINE_string('summary_dir', LOG_PATH, '')


    def __init__(self):
        FLAGS = tf.app.flags.FLAGS

        self.learning_rate = FLAGS.dqn_learning_rate
        self.e_greedy = FLAGS.dqn_e_greedy_start
        self.e_greedy_end = FLAGS.dqn_e_greedy_end
        self.e_greedy_update_iter = FLAGS.dqn_e_greedy_update_iter
        self.e_greedy_update_step = FLAGS.dqn_e_greedy_update_step
        self.gamma = FLAGS.dqn_gamma
        self.batch_size = FLAGS.dqn_batch_size
        self.memory_size = FLAGS.dqn_memory_size
        self.replace_network_iter = FLAGS.dqn_replace_network_iter
        self.no_train_step = FLAGS.dqn_no_train_step
        self.hidden_state = FLAGS.hidden_state

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        self._build_network()
        self.sess = tf.Session()

        self.init_record()

        self.sess.run(tf.global_variables_initializer())

        self.memory = []
        self.memory_count = 0

        self.reset_count = 0
        self.step_num = 0
        self.step_count = 0
        self.merged = tf.summary.merge_all()

        ALREADY_INITIALIZED = set()
        new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
        self.sess.run(tf.variables_initializer(new_variables))
        ALREADY_INITIALIZED.update(new_variables)

        self.update_target()

    def init_record(self):
        f = tf.flags.FLAGS
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(f.summary_dir, self.sess.graph)
        self.test_writer = tf.summary.FileWriter(f.summary_dir)

    def reset(self):
        self.reset_count += 1
        self.step_num = 0


    def store_transition(self, obs, act, rew, don, new_obs):
        if not self.memory_count:
            self.memory_count = 0
        ind = self.memory_count % self.memory_size
        da = (list(obs), act, rew, don, list(new_obs))
        if self.memory_count < self.memory_size:
            self.memory.append(da)
        else:
            self.memory[ind] = da
        self.memory_count += 1

    def choose_action(self, obs):
        obs = np.array(obs)[None]
        if np.random.uniform() > self.e_greedy:
            q_eval = self.sess.run(self.nn_q_eval, feed_dict={self.obs_ph: obs})
            ind = np.argmax(q_eval)
        else:
            ind = np.random.randint(0, obs.shape[1]*2)

        # dict_len = len(char_sign_dict)
        # action = [int(ind/dict_len), ind%dict_len]
        if self.e_greedy > self.e_greedy_end and self.step_count % self.e_greedy_update_iter == 0 and self.step_num == 0:
            self.e_greedy -= self.e_greedy_update_step
        self.step_count += 1
        return ind

    def train(self):
        if self.step_count % self.replace_network_iter == 0:
            self.update_target()

        if self.step_count < self.no_train_step:
            return

        batch_obs, batch_act, batch_rew, batch_don, batch_new_obs = self.random_memory()
        self.sess.run(self.nn_train, feed_dict={self.obs_ph: batch_obs,
                                                 self.action_ph: batch_act,
                                                 self.reward_ph: batch_rew,
                                                 self.done_ph: batch_don,
                                                 self.new_obs_ph: batch_new_obs})
        if self.step_count % 100 == 0:
            summ = self.sess.run(self.merged, feed_dict={self.obs_ph: batch_obs,
                                                    self.action_ph: batch_act,
                                                    self.reward_ph: batch_rew,
                                                    self.done_ph: batch_don,
                                                    self.new_obs_ph: batch_new_obs})
            self.test_writer.add_summary(summ, self.step_count)
            self.test_writer.flush()

        if self.step_count % 1000 == 0:
            saver = tf.train.Saver()
            saver.save(self.sess, os.path.join(CHECKPOINT_PATH, 'dqn'), global_step=self.global_step)

    def random_memory(self):
        '''
        random sample in memory. sample size is batch_size
        :return: obs, act, rew, don, new_obs array
        '''
        if self.memory_count > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_count, size=self.batch_size)

        from common.util import padded_code

        batch_obs = []
        batch_act = []
        batch_rew = []
        batch_don = []
        batch_new_obs = []
        for i in sample_index:
            (obs, act, rew, don, new_obs) = self.memory[i]
            batch_obs.append(obs)
            batch_act.append(np.array(act))
            batch_rew.append(np.array(rew))
            batch_don.append(np.array(don))
            batch_new_obs.append(new_obs)
        padded_code(batch_obs)
        padded_code(batch_new_obs)

        batch_obs = np.array(batch_obs)
        batch_act = np.array(batch_act)
        batch_rew = np.array(batch_rew)
        batch_don = np.array(batch_don)
        batch_new_obs = np.array(batch_new_obs)
        return batch_obs, batch_act, batch_rew, batch_don, batch_new_obs

    @staticmethod
    def padded_code(batch_obs):
        max_obs_len = 0
        for obs in batch_obs:
            max_obs_len = len(obs) if (len(obs) > max_obs_len) else max_obs_len
        for o in batch_obs:
            while len(o) < max_obs_len:
                o.append(-1)
        return batch_obs

    def update_target(self):
        self.sess.run(self.nn_update)

    def _build_network(self):
        self.obs_ph = tf.placeholder(tf.int32, shape=[None, None], name='obs')
        self.new_obs_ph = tf.placeholder(tf.int32, shape=[None, None], name='new_obs')
        self.action_ph = tf.placeholder(tf.int32, shape=[None], name='action')
        self.reward_ph = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.done_ph = tf.placeholder(tf.float32, shape=[None], name='done')


        with tf.variable_scope('build_q_eval_network'):
            self.weight_eval = tf.random_normal(shape=[len(char_sign_dict), 1000], dtype=tf.float32)
            self.nn_q_eval = build_bi_rnn(self.obs_ph, self.hidden_state, self.weight_eval, len(char_sign_dict))
            q_eval_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)

        with tf.variable_scope('build_q_target_network'):
            self.weight_target = tf.random_normal(shape=[len(char_sign_dict), 1000], dtype=tf.float32)
            self.nn_q_target = build_bi_rnn(self.new_obs_ph, self.hidden_state, self.weight_target, len(char_sign_dict))
            q_target_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)

        with tf.variable_scope('loss'):
            act_shape = get_shape(self.nn_q_eval)
            action_mask = tf.one_hot(self.action_ph, depth=act_shape[1])
            q_eval_masked_action = tf.reduce_sum(self.nn_q_eval * action_mask, 1)

            act_new_shape = get_shape(self.nn_q_target)
            q_tp1_best_using_online_net = tf.arg_max(self.nn_q_target, 1)
            q_max_target = tf.reduce_sum(self.nn_q_target * tf.one_hot(q_tp1_best_using_online_net, depth=act_new_shape[1]), 1)

            done_mask = 1.0 - self.done_ph
            q_max_target_masked_done = done_mask * q_max_target
            q_bellman_target = self.reward_ph + self.gamma * q_max_target_masked_done

            td_error = q_eval_masked_action - tf.stop_gradient(q_bellman_target)

            delta = 1.0
            t = tf.where(tf.abs(td_error) < delta, tf.square(td_error) * 0.5, delta * (tf.abs(td_error) - 0.5 * delta))
            self.nn_loss = tf.reduce_mean(t)
            tf.summary.scalar('loss', self.nn_loss)

        with tf.variable_scope('train'):
            gradients = tf.train.AdamOptimizer(learning_rate=self.learning_rate).compute_gradients(self.nn_loss, var_list=q_eval_var)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, 10), var)
            self.nn_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).apply_gradients(gradients, global_step=self.global_step)

        with tf.variable_scope('update'):
            update = []
            for var, var_target in zip(sorted(q_eval_var, key=lambda v: v.name),
                                       sorted(q_target_var, key=lambda v: v.name)):
                update.append(var_target.assign(var))
            self.nn_update = tf.group(*update)



class MLPModel:

    def __init__(self, n_features, n_actions, hiddens=[64], learning_rate=0.001, gamma=1.0):
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.n_actions = n_actions
        self.hiddens = hiddens
        self.gamma = gamma


    def build_network(self):
        with tf.variable_scope('deepq'):
            self.obs_ph = tf.placeholder(tf.float32, shape=[None, self.n_features], name='obs')
            self.new_obs_ph = tf.placeholder(tf.float32, shape=[None, self.n_features], name='new_obs')
            # self.action_ph = tf.placeholder(tf.int32, shape=[None], name='action')
            # self.reward_ph = tf.placeholder(tf.float32, shape=[None], name='reward')
            # self.done_ph = tf.placeholder(tf.float32, shape=[None], name='done')


            self.q_eval = self.build_mlp(self.obs_ph, hiddens=self.hiddens, scope='q_eval_net')
            q_eval_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name+'/'+'q_eval_net')

            self.q_target = self.build_mlp(self.new_obs_ph, hiddens=self.hiddens, scope='q_target_net')
            q_target_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name+'/'+'q_target_net')

            update = []
            for var, var_target in zip(sorted(q_eval_var, key=lambda v: v.name),
                                       sorted(q_target_var, key=lambda v: v.name)):
                update.append(var_target.assign(var))
            self.update = tf.group(*update)
        return self.q_eval, q_eval_var, self.q_target, self.update

    def build_mlp(self, input_ph, hiddens, scope, reuse=False, activation_fn=None):
        import tensorflow.contrib.layers as layers
        with tf.variable_scope(scope, reuse=reuse):
            out = input_ph
            for hidden in hiddens:
                out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
                out = tf.nn.relu(out)
            out = layers.fully_connected(out, num_outputs=self.n_actions, activation_fn=None)
            return out


