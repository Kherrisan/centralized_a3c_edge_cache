from utils import rng
import numpy as np
from keras import layers, models, losses
from keras.optimizers import Adam
import time


class DeepQNetwork:
    def __init__(self, learning_rate, reward_decay, epsilon, epsilon_min, epsilon_decay, memory_size, batch_size, state_dim, action_dim, target_net_period):
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.memory_cnt = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.learning_step_cnt = 0
        self.target_net_period = target_net_period
        self.memory = np.empty(shape=(memory_size, 4, ), dtype=object)
        self.loss = np.array([])

        print("state shape: %s\naction shape: %s" % (state_dim, action_dim))

        self._build_net()

    def _build_net(self):
        self.eval_net = models.Sequential()
        self.eval_net.name = "eval_net"
        # self.eval_net.add(layers.Dense(
        #     100, activation='relu', input_shape=self.state_dim))
        self.eval_net.add(layers.Flatten(input_shape=self.state_dim))
        self.eval_net.add(layers.Dense(200, activation='relu'))
        self.eval_net.add(layers.Dense(self.action_dim, activation='linear'))
        self.eval_net.compile(
            optimizer=Adam(lr=self.learning_rate),
            loss=losses.MeanSquaredError()
        )
        # print()
        # self.eval_net.summary()
        self.target_net = models.Sequential()
        self.target_net.name = "target_net"
        # self.target_net.add(layers.Dense(
        #     100, activation='relu', input_shape=self.state_dim))
        self.target_net.add(layers.Flatten(input_shape=self.state_dim))
        self.target_net.add(layers.Dense(200, activation='relu'))
        self.target_net.add(layers.Dense(self.action_dim, activation='linear'))
        self.target_net.compile(
            optimizer=Adam(lr=self.learning_rate),
            loss=losses.MeanSquaredError()
        )
        self._update_target_net()
        # print()
        # self.target_net.summary()

    def _update_target_net(self):
        self.target_net.set_weights(self.eval_net.get_weights())

    def remember(self, s, a, r, s_):
        ptr = self.memory_cnt % self.memory_size
        self.memory[ptr, :] = np.array((s, a, r, s_), dtype=object)
        self.memory_cnt += 1

    def choose_action(self, state):
        if rng.uniform() > self.epsilon:
            shape = list(state.shape)
            shape.insert(0, -1)
            state = np.reshape(state, shape)
            action_values = self.eval_net.predict(state)
            action = np.argmax(action_values)
        else:
            action = rng.integers(0, self.action_dim)
        return action

    def learn(self):
        # experience replay
        start = time.time()
        if self.memory_cnt > self.memory_size:
            sample_index = rng.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = rng.choice(
                self.memory_cnt, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        b_states, b_actions,  b_rewards,  b_states_ = batch_memory[:,
                                                                   0], batch_memory[:, 1], batch_memory[:, 2], batch_memory[:, 3]
        bb_states_ = np.empty(
            (self.batch_size, self.state_dim[0], self.state_dim[1], self.state_dim[2]))
        bb_states = np.empty(
            (self.batch_size, self.state_dim[0], self.state_dim[1], self.state_dim[2]))
        print("slice time: %sms" % ((time.time()-start)*1000))
        start = time.time()
        for i in range(self.batch_size):
            bb_states_[i, :, :] = b_states_[i]
        for i in range(self.batch_size):
            bb_states[i, :, :] = b_states[i]
        print("transition time: %sms" % ((time.time()-start)*1000))
        start = time.time()
        b_qs = self.eval_net.predict(bb_states_)  # b_qs: batch x action_cnt
        # b_max_q: batch x 1 (max q value for each batch)
        b_max_q = np.max(b_qs, axis=1)
        # b_q: batch x 1 (max q indices for each batch)
        # b_q = np.argmax(b_qs, axis=1)
        target = b_rewards + self.reward_decay * b_max_q  # target: batch x 1
        actions = b_actions.copy().astype(np.int64)  # actions: batch x 1
        actions_index = np.asarray(actions).flatten()
        eval_q = self.eval_net.predict(bb_states)  # eval_q: batch x action_cnt
        # eval_q 中每行的 actions 位置要 fit 到 target 的值
        eval_q[np.arange(self.batch_size), actions_index] = target
        history = self.eval_net.fit(
            bb_states, eval_q, validation_split=0.2, verbose=0, batch_size=self.batch_size)
        print("learning time: %sms" % ((time.time()-start)*1000))

        self.learning_step_cnt += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history["val_loss"][0]
