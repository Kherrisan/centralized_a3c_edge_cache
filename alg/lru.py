import numpy as np
import wandb


class LRUAllocator:
    def __init__(self, state_dim, user_bs_mat):
        self.bs_cnt = state_dim[0]
        self.bs_size = state_dim[1]
        self.file_cnt = state_dim[2]
        self.timer = 1
        self.access_state = np.zeros((self.bs_cnt, self.bs_size))
        self.user_bs_mat = user_bs_mat

    def choose_action(self, state, request):
        if request is None:
            return None
        # request: user, file
        # random choose a bs connected to the user
        user, file = request[0], request[1]
        bss = np.nonzero(self.user_bs_mat[user, :])[0]
        bs = np.random.choice(bss, 1)[0]
        if np.nonzero(state[bs, :, file])[0].size != 0:
            # hit the cache
            cache_index = np.nonzero(state[bs, :, file])[0][0]
            self.access_state[bs, cache_index] = self.timer
            action = None
        else:
            # miss the cache
            if np.sum(state[bs, :, :]) >= self.bs_size:
                # replace the least recently used cache entry with the request file
                lru_index = np.argmin(self.access_state[bs, :])
                self.access_state[bs, lru_index] = self.timer
                action = (bs, lru_index, file)
            else:
                # enough cache size, found an empty cache entry
                cache_state = np.sum(state[bs, :, :], axis=1)
                empty_index = np.nonzero(1-cache_state)[0][0]
                self.access_state[bs, empty_index] = self.timer
                action = (bs, empty_index, file)
        self.timer += 1
        return action


def test_lru(env, h_parameter):
    episodes = h_parameter["e"]
    agent = LRUAllocator(env.state.shape, env.user_bs_mat)
    delay_per_ep = []
    hit_r_per_ep = []
    for e in range(episodes):
        s = env.reset()
        req = None
        delays = []
        hits = []
        while True:
            s = np.reshape(s, env.state_dim)
            a = agent.choose_action(s, req)
            s_, _, done, info = env.step(a)
            req = info["req"]
            hits.append(info["hit"])
            delays.append(info["delay"])
            if done:
                a_d = np.average(delays)
                h_r = np.average(hits)
                delay_per_ep.append(a_d)
                hit_r_per_ep.append(h_r)
                print("episode: ", e, " | average_delay: ",
                      a_d, " | hit_ratio: ", h_r)
                wandb.log({
                    "episode average delay": a_d,
                    "episode hit ratio": h_r
                })
                break
            s = s_
    return delay_per_ep, hit_r_per_ep
