"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.

The Cartpole example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.8.0
gym 0.10.5
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import os
import shutil
import wandb
from environment import EdgeCacheEnvironment
from utils import build_cmd_parser
import numpy as np


OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 10000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

N_S = 0
N_A = 0

np.random.seed(0)

class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(
                    tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(
                    scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(
                        self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
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
                    self.update_a_op = OPT_A.apply_gradients(
                        zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(
                        zip(self.c_grads, globalAC.c_params))
        self.params_saver = tf.train.Saver(self.a_params + self.c_params)

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6,
                                  kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(
                l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
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
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)
        a_loss, c_loss = SESS.run([self.a_loss, self.c_loss], feed_dict)
        wandb.log({
            "a_loss": a_loss,
            "c_loss": c_loss
        })

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = SESS.run(self.a_prob, feed_dict={
                                self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action



class Worker(object):
    def __init__(self, name, globalAC, config):
        self.env = EdgeCacheEnvironment.make(config)
        self.name = name
        self.AC = ACNet(name, globalAC)
        self.ep_delays = []
        self.ep_hits = []

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []    
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            delays = []
            hits = []
            while True:
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

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(
                            self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
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
                    ep_delay = np.average(delays)
                    ep_hit = np.average(hits)
                    self.ep_delays.append(ep_delay)
                    self.ep_hits.append(ep_hit)
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(
                            0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                        "| Ep_delay: %s" % ep_delay
                    )
                    wandb.log({
                        "episode average delay": ep_delay,
                        "episode hit ratio": ep_hit
                    })
                    GLOBAL_EP += 1
                    break
    
    def save(self, save_dir):
        self.AC.save(save_dir)

    def load(self, save_dir):
        self.AC.load(save_dir)


def checkpoint_prefix(config):
    return "./checkpoint/ckpt_%s_%s_%s_%s_%s"%(
        config["m"],
        config["s"],
        config["f"],
        config["u"],
        config["t"]
    )


def thread_worker(worker):
    worker.work()


if __name__ == "__main__":
    parser = build_cmd_parser()
    wandb.init(project='rlbcec', config=parser.parse_args())
    config = wandb.config
    print(config)
    
    MAX_GLOBAL_EP = config['e']
    UPDATE_GLOBAL_ITER = config['i']
    GAMMA = config['g']
    ENTROPY_BETA = config['b']
    LR_A = config['lra']   # learning rate for actor
    LR_C = config['lrc']    # learning rate for critic

    env = EdgeCacheEnvironment.make(config)
    N_S = env.n_s
    N_A = env.n_a

    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        if config["test"]:
            N_WORKERS = 1
            GLOBAL_AC.load(checkpoint_prefix(config))
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC, config))
            
    COORD = tf.train.Coordinator()
    if not config["test"]:
        SESS.run(tf.global_variables_initializer())

    if config["test"]:
            for worker in workers:
                worker.AC.pull_global()
            SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=thread_worker, args=(worker,))
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    delay_per_ep = []
    hit_per_ep = []
    [delay_per_ep.extend(w.ep_delays) for w in workers]
    [hit_per_ep.extend(w.ep_hits) for w in workers]
    wandb.run.summary["average delay"] = np.average(delay_per_ep)
    wandb.run.summary["hit ratio"] = np.average(hit_per_ep)