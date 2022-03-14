from utils import rng
import numpy as np


class QLearning:
    def __init__(self, learning_rate, reward_decay, epsilon, state_dim, action_dim):
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.epsilon = epsilon
        self.state_dim = state_dim
        self.action_dim = action_dim
        table_dim = list(state_dim)
        table_dim.insert(3, action_dim)
        self.q_table = rng.uniform(0, 1, size=table_dim)
        self.states = []
        for m in range(state_dim[0]):
            for c in range(state_dim[1]):
                for f in range(state_dim[2]):
                    self.states.append((m, c, f))

    def learn(self, state, action, reward, state_):
        q = self.q_table[state[0], state[1], state[2], action]
        new_q = reward + self.reward_decay * \
            np.max(self.q_table[state_[0], state_[1], state_[2], :])
        self.q_table[state[0], state[1], state[2],
                     action] += self.learning_rate * (new_q - q)

    def choose_action(self, state):
        if rng.uniform() < self.epsilon:
            action = rng.integers(self.action_dim)
        else:

            action = np.argmax(self.q_table[state[0], state[1], state[2], :])
        return action
