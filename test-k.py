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
import community
#from networkx.algorithms import community
from scipy.stats import truncnorm
from itertools import repeat
import time
import multiprocessing
import os
from pathlib import Path

#Constants and Variables

s = 100 #10^3 
if __name__ ==  '__main__': 
    num_processors = 8
    start = time.time()
    p=Pool( processes = num_processors)
    args1 = {"continuous": True, "type" : "rand", "d": 5}
    #sim1 = p.starmap(models.simulate, zip(range(s), repeat(args1)))
    args2 = {"continuous": True, "type" : "cl", "d": 5}
    #sim2 = p.starmap(models.simulate, zip(range(s), repeat(args2)))
    args3 = {"continuous": False, "type" : "rand",  "d": 25}
    #sim3 = p.starmap(models.simulate, zip(range(s), repeat(args3)))
    args4 = {"continuous": False, "type" : "cl",  "d": 25}
    #sim4 = p.starmap(models.simulate, zip(range(s), repeat(args4)))
    #output = list(map(models.simulate, range(s), repeat(3000)))
    sim1 = p.starmap(models.simulate, zip(range(1), repeat(args1)))
    #community_gen = community.label_propagation_communities(sim1[0].graph)
    partition = community.best_partition(sim1[0].graph)
    #top_lvl = next(community_gen)
    #next_lvl = next(community_gen)
    print(sorted(map(sorted, partition)))
    end = time.time()
    print(f'Time to complete: {end - start:.2f}s\n')
    #models.drawCrossSection(mods)
    fg= plt.figure()
    fg.subplots(nrows=1, ncols=2 )  
    #models.draw_model(sim1[0])
    plt.draw()
    plt.show()
    #fn = Path('~/Documents/Prosjek/Master/Comp/uniformw-sw0.8-k50-test.svg').expanduser()

    #plt.savefig(fn, bbox_inches='tight')
    """plt.xlabel("timesteps")
    plt.ylabel("fraction of cooperators")
    plt.ylim((0, 1))
    

    for i in range(s):
        plt.plot(mods[i].ratio)
        
    #models.avgRadialDist(mods, 6, False)
    plt.show()"""
    