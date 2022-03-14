from test import test, build_env
import os
import numpy as np
import wandb

np.random.seed(0)

# generate the user-bs matrix
U = 20
M = 3
T = 100
F = 10
S = 2
E = 1000

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    # test = Test(U, M, S, F, T, E)
    # test.build()
    # test.train_torch_a3c()
    # test.test_random()
    # test.test_lru()
    # test.test_lfu()
    # test.test_q()
    # test.test_ddqn()
    # test.train_a3c()
    # print(test.train_ga3c())
    # print(test.metrics)
    # print(test.metrics["random"])
    # print(test.metrics["lru"])
    # print(test.metrics["lfu"])
    env = build_env(M, S, F, U, T)
    delay_per_ep, hit_per_ep = test(env, "random", {
        "MAX_GLOBAL_EP": E,
        "ENTROPY_BETA": 0.001,
        "UPDATE_GLOBAL_ITER": 500,
        "TRAINING_INTERVAL": 100,
        "GAMMA": 0.95,
        "LR_A": 0.0005,
        "LR_C": 0.0005,
        "LR": 0.0005
    })
    wandb.run.summary["average delay"] = np.average(delay_per_ep)
    wandb.run.summary["hit ratio"] = np.average(hit_per_ep)

# tags = ["random", "lru", "lfu"]
# tags = ["ddqn"]

# plt.plot(test.metrics["random"]["average_delay"], "r", marker="^")
# plt.plot(test.metrics["lru"]["average_delay"], "r", marker="*")
# plt.plot(test.metrics["lfu"]["average_delay"], "g", marker="+")
# plt.plot(test.metrics["ddqn"]["average_delay"], "b", marker="o")
# plt.show()

# with open('average_delay.csv', 'w') as csv:
#     for tag in tags:
#         csv.write(tag+",")
#     csv.write("\n")
#     for i in range(len(test.metrics["random"]["average_delay"])):
#         for tag in tags:
#             csv.write(str(test.metrics[tag]["average_delay"][i])+",")
#         csv.write("\n")

# with open('hit_ratio.csv', 'w') as csv:
#     for tag in tags:
#         csv.write(tag+",")
#     csv.write("\n")
#     for i in range(len(test.metrics["random"]["hit_ratio"])):
#         for tag in tags:
#             csv.write(str(test.metrics[tag]["hit_ratio"][i])+",")
#         csv.write("\n")
