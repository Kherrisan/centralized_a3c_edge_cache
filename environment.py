import numpy as np
import bisect
from functools import reduce
import threading

USER_BS_DELAY = 5
BS_BS_DELAY = 20
BS_CLOUD_DELAY = 100

ZIPF_SKEWNESS = 0.8


class ZipfGenerator:

    def __init__(self, n, alpha):
        # Calculate Zeta values from 1 to n:
        tmp = [1. / (pow(float(i), alpha)) for i in range(1, n+1)]
        zeta = reduce(lambda sums, x: sums + [sums[-1] + x], tmp, [0])

        # Store the translation map:
        self.distMap = [x / zeta[-1] for x in zeta]

    def next(self):
        # Take a uniform 0-1 pseudo-random value:
        u = np.random.uniform()

        # Translate the Zipf variable:
        return bisect.bisect(self.distMap, u) - 1


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
        cnt = np.random.poisson(u / 4)
        selected = np.random.choice(np.arange(u), cnt, replace=False)
        user_bs_mat[selected, mi] = 1
    for ui in range(u):
        if np.sum(user_bs_mat[ui, :]) == 0:
            user_bs_mat[ui, np.random.randint(0, m)] = 1
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
            cnt = np.random.poisson(u/2)
            r += cnt
            selected = np.random.choice(connected, cnt)
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
            if np.random.uniform() < bs_connection_probability:
                bs_mat[mi, mmi] = 1
                bs_mat[mmi, mi] = 1
    print("bs matrix:\n%s" % (bs_mat,))
    return bs_mat


ENV_CACHE = {}


def build_env(m, s, f, u, t):
    k = ",".join([str(m), str(s), str(f), str(u), str(t)])
    if k in ENV_CACHE.keys():
        env = ENV_CACHE[k].copy()
        return env
    m, s, f, u, t = int(m), int(s), int(f), int(u), int(t)
    actions = build_actions(m, s, f)
    user_bs_mat = build_user_bs_mat(m, u)
    user_trace_mat, r = build_user_trace(m, s, f, u, t, user_bs_mat)
    bs_mat = build_bs_mat(m)
    env = EdgeCacheEnvironment(
        m, s, f, u, user_bs_mat, bs_mat, t, user_trace_mat, actions)
    ENV_CACHE[k] = env
    return env

build_env_lock = threading.Lock()

class EdgeCacheEnvironment:

    @staticmethod
    def make(args):
        with build_env_lock:
            return build_env(
                args["m"],
                args["s"],
                args["f"],
                args["u"],
                args["t"]
            )

    def __init__(self, bs_cnt, bs_size, file_cnt, user_cnt, user_bs_mat, bs_mat, time_slot_cnt, user_trace_mat, actions):
        self.bs_cnt = bs_cnt
        self.user_cnt = user_cnt
        self.file_cnt = file_cnt
        self.bs_size = bs_size
        self.time_slot_cnt = time_slot_cnt
        self.user_trace_mat = user_trace_mat

        self.trace = []
        self.user_bs_mat = user_bs_mat
        self.bs_mat = bs_mat
        self.actions = actions

        self.initial_state = np.zeros(
            (self.bs_cnt, self.bs_size, self.file_cnt), dtype=np.int8)
        self.reset()

        for ti in range(time_slot_cnt):
            reqs = []
            for ui in range(self.user_cnt):
                for fi in range(self.file_cnt):
                    if user_trace_mat[ti, ui, fi] == 1:
                        reqs.append((ui, fi))
            np.random.shuffle(reqs)
            self.trace.extend(reqs)
        self.state_dim = (bs_cnt, bs_size, file_cnt)
        self.n_s = self.bs_cnt*self.bs_size*self.file_cnt
        self.n_a = len(self.actions)

    def done(self):
        return self.step_cnt >= len(self.trace)

    def step(self, a):
        if a is not None:
            if type(a) is tuple:
                action = a
            else:
                action = self.actions[a]
            # perform the action
            # action is tuple(3): (dst, dst_cache_index, file)
            # download file from the cloud, cost backhual traffic
            self.state[action[0], action[1], :] = 0
            self.state[action[0], action[1], action[2]] = 1

        # simulate user request within one time slot to calculate reward
        hit = 0
        req = self.trace[self.step_cnt]
        user, file = req[0], req[1]
        bs = np.nonzero(self.user_bs_mat[user, :])[0]
        if np.nonzero(self.state[bs, :, file])[0].size != 0:
            # hit the cache
            hit = 1
            delay = USER_BS_DELAY
        else:
            if np.sum(self.state[:, :, file]) > 0:
                # res = []
                # for b in bs:
                #     hops = self.__search_path(b, file)
                #     if hops != -1:
                #         res.append(hops)
                # if len(res) != 0:
                # delay = USER_BS_DELAY+BS_BS_DELAY*np.min(res)
                delay = USER_BS_DELAY + BS_BS_DELAY
            else:
                delay = USER_BS_DELAY + BS_CLOUD_DELAY

        self.step_cnt += 1
        info = {"delay": delay, "hit": hit, "req": req}
        r = 1/delay
        return self.state.flatten(), r, self.step_cnt >= len(self.trace), info

    def reset(self):
        self.state = self.initial_state.copy()
        self.step_cnt = 0
        return self.state.flatten()

    def copy(self):
        return EdgeCacheEnvironment(self.bs_cnt, self.bs_size, self.file_cnt, self.user_cnt, self.user_bs_mat, self.bs_mat, self.time_slot_cnt, self.user_trace_mat, self.actions)

    def summary_episode(self, infos):
        delay = 0
        for info in infos:
            delay += info["delay"]
        return "Average delay: {:.2}".format(delay / len(infos))
