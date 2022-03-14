import imp
import numpy as np
from utils import ZipfGenerator, rng
# from _ddqn import DeepQNetwork
from alg.lru import test_lru
from alg.lfu import test_lfu
from alg.r import test_random
import wandb
from utils import build_cmd_parser
import argparse

# wandb.init(project="rlbec", entity="zoudikai")
# def get_session():
#     """ Limit session memory usage
#     """
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     return tf.Session(config=config)


ZIPF_SKEWNESS = 0.8


def build_actions(m, s, f):
    actions = []
    for di in range(m):
        for dci in range(s):
            for fi in range(f):
                actions.append((di, dci, fi))
    print("action space: %s" % (len(actions),))
    return actions


def build_user_bs_mat(m, u):
    user_bs_mat = np.zeros((u, m))
    for mi in range(m):
        cnt = rng.poisson(u / 4)
        selected = rng.choice(np.arange(u), cnt, replace=False)
        user_bs_mat[selected, mi] = 1
    for ui in range(u):
        if np.sum(user_bs_mat[ui, :]) == 0:
            user_bs_mat[ui, rng.randint(0, m)] = 1
    print("U: %s\nM: %s" % (
        user_bs_mat.shape[0],
        user_bs_mat.shape[1]))
    bs_sum = np.sum(user_bs_mat, axis=0)
    print("bs max: %s, bs min: %s" % (np.max(bs_sum), np.min(bs_sum)))
    user_sum = np.sum(user_bs_mat, axis=1)
    print("user max: %s, user min: %s" %
          (np.max(user_sum), np.min(user_sum)))
    return user_bs_mat


def build_user_trace(m, s, f, u, t, user_bs_mat):
    r = 0
    time_user_trace = np.zeros(
        (t, u, f))
    zipf = ZipfGenerator(f, ZIPF_SKEWNESS)
    for ti in range(t):
        for bi in range(m):
            connected = np.nonzero(user_bs_mat[:, bi])[0]
            cnt = rng.poisson(u/2)
            r += cnt
            selected = rng.choice(connected, cnt)
            for si in range(cnt):
                time_user_trace[ti, selected[si], zipf.next()] = 1
    print("request count: %s" % (r,))
    return time_user_trace, r


def build_bs_mat(m):
    bs_connection_probability = 0.7
    bs_mat = np.zeros((m, m))
    # generate bs-bs matrix
    for mi in range(m):
        for mmi in range(mi+1, m):
            if rng.uniform() < bs_connection_probability:
                bs_mat[mi, mmi] = 1
                bs_mat[mmi, mi] = 1
    print("bs matrix:\n%s" % (bs_mat,))
    return bs_mat


def build_env(m, s, f, u, t):
    actions = build_actions(m, s, f)
    user_bs_mat = build_user_bs_mat(m, u)
    user_trace_mat, r = build_user_trace(m, s, f, u, t, user_bs_mat)
    bs_mat = build_bs_mat(m)
    from environment import EdgeCacheEnvironment
    return EdgeCacheEnvironment(m, s, f, u, user_bs_mat, bs_mat, t, user_trace_mat, actions)


def test(config):
    env = build_env(config['m'], config['s'], config['f'], config['u'], config['t'])
    alg = config['a']
    if alg == "lru":
        return test_lru(env, config)
    elif alg == "lfu":
        return test_lfu(env, config)
    elif alg == "random":
        return test_random(env, config)
    else:
        return None


if __name__=="__main__":
    parser = build_cmd_parser()
    wandb.init(project='rlbcec', config=parser.parse_args())
    config = wandb.config
    print(config)
    delay_per_ep, hit_per_ep= test(config)
    wandb.run.summary["average delay"] = np.average(delay_per_ep)
    wandb.run.summary["hit ratio"] = np.average(hit_per_ep)