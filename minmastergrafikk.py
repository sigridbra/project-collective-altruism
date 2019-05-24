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
from pathlib import Path
import community
import dill

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

plt.rcParams["savefig.directory"] = "Master/Comp/Week19_2"

if __name__ ==  '__main__': 
    start = time.time()
    
    fn1 = Path('~/Documents/Prosjek/Master/Comp/balanced/balanced70-d8-check-friends-.svg').expanduser()
    fn2 = Path('~/Documents/Prosjek/Master/Comp/balanced/balanced70-d8-check-friends-clusters.svg').expanduser()
    fn3 = Path('~/Documents/Prosjek/Master/Comp/balanced/balanced70-d8-check-friends-agreeingfriends.svg').expanduser()
    #fg = plt.figure()
    model = models.simulate(0, {"type": "grid", "continuous":True, "k":1, "skew":0})#"initSD": 0.25})
    print("degree: ", nx.info(model.graph))
    
    #fg.clear()
    model.runSim(100, clusters=True, gifname="check-friends")
    

    fg1= plt.figure("states")
    fg1.subplots(nrows=1, ncols=2 )
    models.drawAvgState([model], clusterSD=True)
    models.drawCrossSection([model])
    fg1.savefig(fn1)
    
    
    """fg2 = plt.figure("cross")
    fg2.subplots(nrows=1, ncols=3 )
    models.drawClusterState([model], step=500, subplot=1)
    models.drawClusterState([model], step=1000, subplot=2)
    models.drawClusterState([model])
    fg2.savefig(fn2)"""
    fg3 = plt.figure("agreeingfriends")
    models.drawAvgNumberOfAgreeingFriends([model], pltNr=7)
    plt.draw()
    fg3.savefig(fn3)
    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Time to complete: {mins:5.0f} mins {sec}s\n')
    