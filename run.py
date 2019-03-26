from multiprocessing import Pool
import models 
from functools import reduce
import numpy as np
from operator import itemgetter
import heapq
import random
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from copy import deepcopy
import seaborn as sns
#import pygraphviz as pgv
from statistics import stdev, mean
import imageio
import networkx as nx
from scipy.stats import truncnorm
from itertools import repeat
import time
import multiprocessing
import os

#Constants and Variables

s = 100

if __name__ ==  '__main__': 
    num_processors = 8
    start = time.time()
    p=Pool( processes = num_processors)
    mods = p.starmap(models.simulate, zip(range(s), repeat("args")))
    #output = list(map(models.simulate, range(s), repeat(3000)))

    end = time.time()
    print(f'Time to complete: {end - start:.2f}s\n')
    plt.xlabel("timesteps")
    plt.ylabel("fraction of cooperators")
    plt.ylim((0, 1))
    #print(output)
    #for i in range(s):
    #    plt.plot(mods[i].ratio)
    models.avgRadialDist(mods, 6, False)
    plt.show()
    