"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.
The Cartpole example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.8.0
gym 0.10.5
"""

import multiprocessing
from telnetlib import SE
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
import wandb


# GAME = 'CartPole-v0'
# OUTPUT_GRAPH = True
# LOG_DIR = './log'
# N_WORKERS = multiprocessing.cpu_count()
# MAX_GLOBAL_EP = 1000
GLOBAL_NET_SCOPE = 'Global_Net'
# UPDATE_GLOBAL_ITER = 10
# GAMMA = 0.9
# ENTROPY_BETA = 0.001
# LR_A = 0.001    # learning rate for actor
# LR_C = 0.001    # learning rate for critic
# GLOBAL_RUNNING_R = []
# GLOBAL_EP = 0

# env = gym.make(GAME)
# N_S = env.observation_space.shape[0]
# N_A = env.action_space.n


class ACNet(object):
    def __init__(self, scope, env, h_parameter, opt_a, opt_c, sess, globalAC=None):
        self.env = env
        self.h_parameter = h_parameter
        self.sess = sess

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.env.n_s], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None,  self.env.n_s], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(
                    tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(
                    scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(tf.clip_by_value(self.a_prob, 1e-20, 0.999999)) * tf.one_hot(
                        self.a_his,  self.env.n_a, dtype=tf.float32), axis=1, keepdims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(tf.clip_by_value(self.a_prob, 1e-20, 0.999999)),
                                             axis=1, keepdims=True)  # encourage exploration
                    self.exp_v = self.h_parameter["ENTROPY_BETA"] * \
                        entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(
                        self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(
                        self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = opt_a.apply_gradients(
                        zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = opt_c.apply_gradients(
                        zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6,
                                  kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(
                l_a,  self.env.n_a, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6,
                                  kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(
                l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        # local grads applies to global net
        self.sess.run(
            [self.update_a_op, self.update_c_op], feed_dict)
        a_loss, c_loss = self.sess.run([self.a_loss, self.c_loss], feed_dict)
        wandb.log({
            "a_loss": a_loss,
            "c_loss": c_loss
        })
        print(a_loss)

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = self.sess.run(self.a_prob, feed_dict={
                                     self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.flatten())  # select action w.r.t the actions prob
        return action


class Worker(object):
    def __init__(self, name, globalAC, env, h_parameter, g_r, g_e, opt_a, opt_c, coord, sess):
        self.env = env
        self.h_parameter = h_parameter
        self.name = name
        self.AC = ACNet(name, env, h_parameter, opt_a, opt_c, sess, globalAC)
        self.g_r = g_r
        self.g_e = g_e
        self.coord = coord
        self.sess = sess

    def work(self):
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not self.coord.should_stop() and self.g_e[0] < self.h_parameter["MAX_GLOBAL_EP"]:
            s = self.env.reset()
            ep_r = 0
            delays = []
            hits = []
            while self.env.step_cnt < len(self.env.trace):
                # if self.name == 'W_0':
                #     self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                delays.append(info["delay"])
                hits.append(info["hit"])
                if done:
                    r = -5
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                # update global and assign to local net
                if total_step % self.h_parameter["UPDATE_GLOBAL_ITER"] == 0 or done:
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = self.sess.run(
                            self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + self.h_parameter["GAMMA"] * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(
                        buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(self.g_r) == 0:  # record running episode reward
                        self.g_r.append(ep_r)
                    else:
                        self.g_r.append(0.99 * self.g_r[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", self.g_e[0],
                        "| Ep_r: %s" % self.g_r[-1],
                        "| A_D: %s" % np.average(delays)
                    )
                    wandb.log({
                        "Ep": self.g_e[0],
                        "Ep_r": self.g_r[-1],
                        "Proposed average delay": np.average(delays),
                        "Proposed hit ratio": np.average(hits)
                    }, step=self.g_e[0])
                    delays.clear()
                    hits.clear()
                    self.g_e[0] += 1
                    break


def test_a3c(env, h_parameter, train=True):
    OUTPUT_GRAPH = True
    LOG_DIR = './log'
    N_WORKERS = multiprocessing.cpu_count()
    # N_WORKERS = 2
    GLOBAL_NET_SCOPE = 'Global_Net'
    GLOBAL_RUNNING_R = []
    GLOBAL_EP = [0]

    SESS = tf.Session()
    COORD = tf.train.Coordinator()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(h_parameter["LR_A"], name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(h_parameter["LR_C"], name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE, env, h_parameter,
                          OPT_A, OPT_C, SESS)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(
                i_name,
                GLOBAL_AC,
                env.copy(),
                h_parameter,
                GLOBAL_RUNNING_R,
                GLOBAL_EP,
                OPT_A,
                OPT_C,
                COORD,
                SESS))

    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        def job(): return worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
