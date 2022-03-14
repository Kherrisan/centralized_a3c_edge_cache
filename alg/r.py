import numpy as np
import wandb


class RandomAllocator:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def choose_action(self):
        return np.random.randint(0, self.action_dim)


def test_random(env, h_parameter):
    episodes = h_parameter["e"]
    agent = RandomAllocator(len(env.actions))
    delay_per_ep = []
    hit_r_per_ep = []
    for e in range(episodes):
        s = env.reset()
        delays = []
        hits = []
        while True:
            a = agent.choose_action()
            s_, _, done, info = env.step(a)
            hits.append(info["hit"])
            delays.append(info["delay"])
            if done:
                a_d = np.average(delays)
                h_r = np.average(hits)
                delay_per_ep.append(a_d)
                hit_r_per_ep.append(h_r)
                print("episode: ", e, " | average_delay: ",
                      a_d, " | hit_ratio: ", h_r)
                if e % 100 == 0 or e == episodes-1:
                    wandb.log({
                        "episode average delay": a_d,
                        "episode hit ratio": h_r
                    })
                else:
                    wandb.log({
                        "episode average delay": a_d,
                        "episode hit ratio": h_r
                    }, commit=False)
                break
            s = s_
    return delay_per_ep, hit_r_per_ep
