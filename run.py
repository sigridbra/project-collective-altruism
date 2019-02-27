from multiprocessing import Pool
import models 
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
from threading import Thread
#from model import Agent

#Constants and Variables

states = [1, -1] #1 being cooperating, -1 being defecting

defectorUtility = -0.20 
 
politicalClimate=0.2 

selfWeight = 0.8

neighboursWeight = 0.5

s = 100

#Helper
def decision(probability):
    return random.random() < probability

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

if __name__ ==  '__main__': 
    num_processors = 8
    modelList=[]
    start = time.time()
    p=Pool(processes = num_processors)
    #startMake = time.time()
    #for i in range(s):
    #    modelList.append(models.ScaleFreeModel(144, 2))  
    #endMake=time.time()
    output = p.starmap(models.simulate, zip(range(s), repeat(3000)))
    #output = list(map(models.simulate, range(s), repeat(3000)))

    end = time.time()
    
    #print(f'Time to make models: {endMake - startMake:.2f}s\n')
    print(f'Time to complete: {end - start:.2f}s\n')
    plt.xlabel("timesteps")
    plt.ylabel("fraction of cooperators")
    plt.ylim((0, 1))
    for i in range(s):
        plt.plot(output[i])
    