import pickle
from run import dqn
from numpy import array
import matplotlib.pyplot as plt
import numpy as np

#dqn(agent=pickle.load(open("Agent2_75000.pickle", "rb")), doTrain=False, episodes=100, render_after=1, mem_size=0)

mean_width = 100
score_local_mean= []
scores = np.loadtxt('Agent2_1000_scores.txt', delimiter="\n")
plt.plot(scores)
for k in range(len(scores)):
    start = max(k - mean_width, 0)
    end = k + mean_width
    s = scores[start:end]
    score_local_mean.append(sum(s) / len(s))
plt.plot(score_local_mean)
plt.show()