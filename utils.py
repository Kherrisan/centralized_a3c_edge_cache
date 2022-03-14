from functools import reduce
import bisect
# from numpy.random import default_rng
from numpy import random
SEED = 2020
rng = random
# rng = default_rng(seed=SEED)
import argparse


class ZipfGenerator:

    def __init__(self, n, alpha):
        # Calculate Zeta values from 1 to n:
        tmp = [1. / (pow(float(i), alpha)) for i in range(1, n+1)]
        zeta = reduce(lambda sums, x: sums + [sums[-1] + x], tmp, [0])

        # Store the translation map:
        self.distMap = [x / zeta[-1] for x in zeta]

    def next(self):
        # Take a uniform 0-1 pseudo-random value:
        u = rng.uniform()

        # Translate the Zipf variable:
        return bisect.bisect(self.distMap, u) - 1


def build_cmd_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--a', type=str, default='a3c')
    parser.add_argument('--m', type=int, default=3)
    parser.add_argument('--s', type=int, default=5)
    parser.add_argument('--f', type=int, default=25)
    parser.add_argument('--u', type=int, default=50)
    parser.add_argument('--t', type=int, default=100)
    parser.add_argument('--i', type=int, default=20)
    parser.add_argument('--g', type=float, default=0.9)
    parser.add_argument('--b', type=float, default=0.001)
    parser.add_argument('--lra', type=float, default=0.0001)
    parser.add_argument('--lrc', type=float, default=0.0001)
    parser.add_argument('--e','--episode', type=int, default=5000)
    return parser