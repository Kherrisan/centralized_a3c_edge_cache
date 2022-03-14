# GPU based A3C
# haiyinpiao@qq.com

from multiprocessing.queues import Queue as mp_queue
from threading import Thread
import multiprocessing
import numpy as np
import tensorflow as tf
import os

from tqdm import tqdm

import gym
import time
import random

# import threading

import multiprocessing as mp

from keras.models import *
from keras.layers import *
from keras import backend as K

# log and visualization.
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
start = time.time()


# multithreading for brain


def log_reward(R):
    a_time.put(time.time() - start)
    a_reward.put(R)


THREAD_DELAY = 0.001


class SharedCounter(object):
    """ A synchronized shared counter.

    The locking done by multiprocessing.Value ensures that only a single
    process or thread may read or write the in-memory ctypes object. However,
    in order to do n += 1, Python performs a read followed by a write, so a
    second process may read the old value before the new one is written by the
    first process. The solution is to use a multiprocessing.Lock to guarantee
    the atomicity of the modifications to Value.

    This class comes almost entirely from Eli Bendersky's blog:
    http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/

    """

    def __init__(self, n=0):
        self.count = multiprocessing.Value('i', n)

    def increment(self, n=1):
        """ Increment the counter by n (default = 1) """
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        """ Return the value of the counter """
        return self.count.value


class Queue(mp_queue):
    """ A portable implementation of multiprocessing.Queue.

    Because of multithreading / multiprocessing semantics, Queue.qsize() may
    raise the NotImplementedError exception on Unix platforms like Mac OS X
    where sem_getvalue() is not implemented. This subclass addresses this
    problem by using a synchronized shared counter (initialized to zero) and
    increasing / decreasing its value every time the put() and get() methods
    are called, respectively. This not only prevents NotImplementedError from
    being raised, but also allows us to implement a reliable version of both
    qsize() and empty().

    """

    def __init__(self, *args, **kwargs):
        super(Queue, self).__init__(
            ctx=multiprocessing.get_context(), *args, **kwargs)
        self.size = SharedCounter(0)

    def put(self, *args, **kwargs):
        self.size.increment(1)
        super(Queue, self).put(*args, **kwargs)

    def get(self, *args, **kwargs):
        self.size.increment(-1)
        return super(Queue, self).get(*args, **kwargs)

    def qsize(self):
        """ Reliable implementation of multiprocessing.Queue.qsize() """
        return self.size.value

    def empty(self):
        """ Reliable implementation of multiprocessing.Queue.empty() """
        return not self.qsize()


# ---------
a_time = Queue()
a_reward = Queue()


class Brain:
    def __init__(self, state_dim, action_dim, loss_v, loss_entropy, lr):
        self.loss_v = loss_v
        self.loss_entropy = loss_entropy
        self.lr = lr

        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model(state_dim, action_dim)
        self.graph = self._build_graph(self.model, state_dim, action_dim)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()  # avoid modifications

        # multiprocess global sample queue for batch traning.
        self._train_queue = Queue()
        self._train_lock = mp.Lock()

        # multiprocess global state queue for action predict
        self._predict_queue = Queue()
        self._predict_lock = mp.Lock()

        self._predictors = []
        self._trainers = []

    def _build_model(self, state_dim, action_dim):

        l_input = Input(batch_shape=(None, ) + state_dim)
        I_flattern = Flatten()(l_input)
        l_dense = Dense(128, activation='relu')(I_flattern)
        l_dense = Dense(256, activation='relu')(I_flattern)

        out_actions = Dense(action_dim, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()  # have to initialize before threading

        return model

    def _build_graph(self, model, state_dim, action_dim):
        s_t = tf.placeholder(tf.float32, shape=(None, ) + state_dim)
        a_t = tf.placeholder(tf.float32, shape=(None, action_dim))
        # not immediate, but discounted n step reward
        r_t = tf.placeholder(tf.float32, shape=(None, 1))

        p, v = model(s_t)

        log_prob = tf.log(tf.reduce_sum(
            p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * \
            tf.stop_gradient(advantage)									# maximize policy
        # minimize value error
        loss_value = self.loss_v * tf.square(advantage)
        # maximize entropy (regularization)
        entropy = self.loss_entropy * \
            tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(self.lr, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v

    def add_predictor(self, min_batch):
        self._predictors.append(ThreadPredictor(
            self, len(self._predictors), min_batch))
        self._predictors[-1].start()

    def add_trainer(self, min_batch, gamma_n):
        self._trainers.append(ThreadTrainer(
            self, len(self._trainers), min_batch, gamma_n))
        self._trainers[-1].start()

    def save_weights(self, path):
        self.model.save_weights(path + '_model.h5')

    def load_weights(self, path):
        self.model.load_weights(path + '_model.h5')


class ThreadPredictor(Thread):
    def __init__(self, brain, id, min_batch):
        super(ThreadPredictor, self).__init__()
        self.setDaemon(True)

        self._id = id
        self._brain = brain
        self.stop_signal = False
        self.min_batch = min_batch

    def batch_predict(self):
        global envs

        if self._brain._predict_queue.qsize() < self.min_batch:  # more thread could have passed without lock
            time.sleep(0)
            return
            # we can't yield inside lock
        # if self._brain._predict_queue.empty():
        # 	return

        i = 0
        id = []
        s = []
        while not self._brain._predict_queue.empty():
            id_, s_ = self._brain._predict_queue.get()
            if i == 0:
                s = s_
            else:
                s = np.row_stack((s, s_))
            id.append(id_)
            i += 1

        if s == []:
            return

        p = self._brain.predict_p(np.array(s))

        for j in range(i):
            if id[j] < len(envs):
                envs[id[j]].agent.wait_q.put(p[j])

    def run(self):
        while not self.stop_signal:
            self.batch_predict()

    def stop(self):
        self.stop_signal = True


class ThreadTrainer(Thread):
    def __init__(self, brain, id, min_batch, gamma_n):
        super(ThreadTrainer, self).__init__()
        self.setDaemon(True)

        self._id = id
        self._brain = brain
        self.stop_signal = False
        self.min_batch = min_batch
        self.gamma_n = gamma_n

    def batch_train(self):
        if self._brain._train_queue.qsize() < self.min_batch:  # more thread could have passed without lock
            time.sleep(0)
            return 									# we can't yield inside lock

        if self._brain._train_queue.empty():
            return

        i = 0
        s = []
        while not self._brain._train_queue.empty():
            s_, a_, r_, s_next_, s_mask_ = self._brain._train_queue.get()
            if i == 0:
                s, a, r, s_next, s_mask = s_, a_, r_, s_next_, s_mask_
            else:
                s = np.row_stack((s, s_))
                a = np.row_stack((a, a_))
                r = np.row_stack((r, r_))
                s_next = np.row_stack((s_next, s_next_))
                s_mask = np.row_stack((s_mask, s_mask_))
            i += 1
        if s == []:
            return

        if len(s) > 100*self.min_batch:
            print("Optimizer alert! Minimizing train batch of %d" % len(s))

        v = self._brain.predict_v(s_next)
        r = r + self.gamma_n * v * s_mask  # set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize = self._brain.graph
        self._brain.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

    def run(self):
        while not self.stop_signal:
            self.batch_train()

    def stop(self):
        self.stop_signal = True


# ---------
frames = 0


class Agent:
    def __init__(self, state_dim, action_dim, id, eps_start, eps_end, eps_steps, predict_queue, predict_lock, train_queue, train_lock, gamma, n_step_return):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.id = id
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.gamma = gamma
        self.n_step_return = n_step_return

        self.memory = []  # used for n_step return
        self.R = 0.

        # for predicted nn output dispatching
        self.wait_q = Queue(maxsize=1)
        self._predict_queue = predict_queue
        self._predict_lock = predict_lock

        # for training
        self._train_queue = train_queue
        self._train_lock = train_lock

    def getEpsilon(self):
        if(frames >= self.eps_steps):
            return self.eps_end
        else:
            # linearly interpolate
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps

    def act(self, s):
        eps = self.getEpsilon()
        global frames
        frames = frames + 1

        s = np.array([s])
        # put the state in the prediction q
        self._predict_queue.put((self.id, s))
        # wait for the prediction to come back
        p = self.wait_q.get()
        a = np.random.choice(self.action_dim, p=p)
        if random.random() < eps:
            a = random.randint(0, self.action_dim-1)
        return a

    def train(self, s, a, r, s_):

        def train_push(s, a, r, s_):
            if s_ is None:
                s_next = np.zeros(self.state_dim)
                s_mask = 0.
            else:
                s_next = s_
                s_mask = 1.
            s = np.array([s])
            a = np.array([a])
            r = np.array([r])
            s_next = np.array([s_next])
            s_mask = np.array([s_mask])
            self._train_queue.put((s, a, r, s_next, s_mask))

        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n-1]

            return s, a, self.R, s_

        # turn action into one-hot representation
        a_cats = np.zeros(self.action_dim)
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))
        gamma_n = self.gamma ** self.n_step_return
        self.R = (self.R + r * gamma_n) / self.gamma

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / self.gamma
                self.memory.pop(0)
            self.R = 0

        if len(self.memory) >= self.n_step_return:
            s, a, r, s_ = get_sample(self.memory, self.n_step_return)
            train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)


# ---------
episode = 0


class Environment(mp.Process):
    stop_signal = False

    def __init__(self, state_dim, action_dim, id, predict_queue, predict_lock, train_queue, train_lock, env, eps_start, eps_end, eps_steps, gamma, n_step_return, episode_cnt, e_tqdm, e_lock, render=False, train=True):
        mp.Process.__init__(self)

        self.id = id
        self.render = render
        self.env = env
        self.agent = Agent(state_dim, action_dim, id, eps_start, eps_end, eps_steps,
                           predict_queue, predict_lock, train_queue, train_lock, gamma, n_step_return)
        self.episode_cnt = episode_cnt
        self._train = train
        self.e_tqdm = e_tqdm
        self.e_lock = e_lock

    def run_episode(self):
        s = self.env.reset()
        infos = []
        R = 0
        while True:
            time.sleep(THREAD_DELAY)
            if self.render:
                self.env.render()
            a = self.agent.act(s)
            s_, r, done, info = self.env.step(a)
            infos.append(info)

            if done:  # terminal state
                s_ = None

            if self._train:
                self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done:  # or self.stop_signal:
                break

        log_reward(R)
        return infos

    def run(self):
        global episode
        while episode < self.episode_cnt:
            infos = self.run_episode()
            with self.e_lock:
                if self.env.summary_episode:
                    self.e_tqdm.set_description(
                        self.env.summary_episode(infos))
                self.e_tqdm.update(1)
                if(episode < self.episode_cnt):
                    episode += 1
