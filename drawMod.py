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
    
    fn1 = Path('~/Documents/Prosjek/Master/Comp/balanced/exModel-2.svg').expanduser()
    fn2 = Path('~/Documents/Prosjek/Master/Comp/balanced/exModel-2-clusters.svg').expanduser()
    fn3 = Path('~/Documents/Prosjek/Master/Comp/balanced/exModel-2-agreeingfriends.svg').expanduser()
    #fg = plt.figure()
    fnbefore = Path('~/Documents/Prosjek/Master/Comp/balanced/exModel-before3.svg').expanduser()
    fnafter = Path('~/Documents/Prosjek/Master/Comp/balanced/exModel-after3.svg').expanduser()

    model = models.simulate(0, {"type": "cl", "continuous":True, "d":3, "k":1, "skew":0.0, "influencers":1})#"initSD": 0.25})
    print("degree: ", nx.info(model.graph))
    #bf = plt.figure("before", figsize=(14,7))
    #bf.subplots(nrows=1,ncols=2)

    #models.drawClusteredModel(model)
    #plt.draw()
    #plt.show()
    #bf.savefig(fnbefore)

    #fg.clear()
    model.runSim(500, clusters=True)

    af = plt.figure("after", figsize=(10,10))
    #af.subplots(nrows=1,ncols=2)
    models.draw_model(model)
    plt.show()
    af.savefig(fnafter)

    #fg = plt.figure(figsize=(14,7))
    #models.draw_model(model)
    #plt.show()
    #fn = Path('~/Documents/Prosjek/Master/Figurer/clustered2.svg').expanduser()
    #fg.savefig(fn)
    
    fg1= plt.figure("states", figsize=(14,7))
    fg1.subplots(nrows=1, ncols=2 )
    models.drawAvgState([model], clusterSD=True)
    models.drawCrossSection([model])
    fg1.savefig(fn1)
    
    
    fg2 = plt.figure("cross")
    fg2.subplots(nrows=1, ncols=3 )
    #models.drawClusterState([model], step=500, subplot=1)
    #models.drawClusterState([model], step=1000, subplot=2)
    models.drawClusterState([model])
    fg2.savefig(fn2)
    fg3 = plt.figure("agreeingfriends")
    models.drawAvgNumberOfAgreeingFriends([model], pltNr=7)
    plt.draw()
    fg3.savefig(fn3)
    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Time to complete: {mins:5.0f} mins {sec}s\n')
    