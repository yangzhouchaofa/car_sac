import numpy as np
import matplotlib.pyplot as plt
from math import *

path_history_path = 'path/path_history'

if __name__ == '__main__':
    path_history = np.load(path_history_path + '.npy')
    # path_history = list(path_history)
    # x = path_history[:,0]
    # y = path_history[:,1]
    # theta = path_history[:,2]
    eps = path_history[:,3]
    for i in range(int(eps[-1])):
    # for i in range(5,10):
        path = []
        for j in path_history:
            if int(j[3]) == i:
                path.append(list(j))
        path = np.array(path)
        plt.plot(path[:,0], path[:,1])
        plt.arrow(path[0,0],path[0,1],0.1*cos(pi-path[0,2]),0.1*sin(pi-path[0,2]),width=0.1,head_width=0.2,head_length=0.2)
    plt.show()