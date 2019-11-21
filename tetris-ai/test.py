import pickle
from run import dqn
from numpy import array
import matplotlib.pyplot as plt
import numpy as np

#dqn(agent=pickle.load(open("Agent2_50000_10000_1.pickle", "rb")), doTrain=False, episodes=100, render_after=1, mem_size=0)

mean_width = 100
score_local_mean = []
scores = np.loadtxt('Agent2_1000050000_scores.txt', delimiter="\n")
plt.plot(scores)
for k in range(len(scores)):
    start = max(k - mean_width, 0)
    end = k + mean_width
    s = scores[start:end]
    score_local_mean.append(sum(s) / len(s))
plt.plot(score_local_mean)
plt.ylim(0, 5000)
plt.show()

#memsiz = [1, 1000, 10000, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000, 225000, 250000]
#times = [210, 198, 1657, 1536, 1236, 1127, 1285, 1292, 1426, 1444, 1343, 1452, 1433]
#means = [19, 19, 325, 214, 406, 194, 328, 342, 281, 330, 261, 324, 288]
#bests = [39, 40, 7189, 44635, 4705, 2695, 2157, 1804, 5281, 3713, 3022, 2098, 1408]