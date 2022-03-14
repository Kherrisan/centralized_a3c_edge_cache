from queue import Queue
import re
import numpy as np
import random
random.seed(15)

M = 3
S = 3
F = 5
mat = np.zeros((M, M))

for i in range(M):
    for j in range(i+1, M):
        if random.uniform(0, 1) < 0.8:
            mat[i, j] = 1
            mat[j, i] = 1

cache = np.zeros((M, S, F))
for i in range(M):
    for j in range(S):
        for k in range(F):
            if random.uniform(0, 1) < 0.25:
                cache[i, j, k] = 1


def search_path(mat, cache, start, target):
    q = Queue()
    visits = []
    q.put(start)
    hops = 0
    levels = [1, 0]
    found = False
    while not q.empty():
        next = q.get()
        levels[0] -= 1
        visits.append(next)
        if np.sum(cache[next, :, target]) > 0:
            found = True
            break
        else:
            for n in np.nonzero(mat[next, :])[0]:
                if n not in visits:
                    q.put(n)
                    levels[1] += 1
        if levels[0] == 0:
            hops += 1
            levels[0] = levels[1]
            levels[1] = 0
    if found:
        return hops
    else:
        return -1


print(mat)
print(cache)
print(search_path(mat, cache, 0, 1))
