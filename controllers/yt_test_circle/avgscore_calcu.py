import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

score_history_path = 'score/score_history2'
if __name__ == '__main__':
    score_history = np.load(score_history_path + '.npy')
    avgscore = []
    x = []
    for i in range(len(score_history)):
        if i % 100 == 0 and i > 0:
            x.append(i)
            avgscore.append(np.mean(score_history[i-100:i-1]))
            # avgscore = np.mean(score_history[i-100:i-1])
    plt.plot(x,avgscore)
    plt.xlabel('episode', fontsize=14)
    plt.ylabel('avg_score', fontsize=14)
    plt.show()
