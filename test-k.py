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
fn = Path('~/Documents/Prosjek/Master/clustered-test.svg').expanduser()
s = 1 #10^3 

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
    sim1 = p.starmap(models.simulate, zip(range(1), repeat(args2)))
    #G = nx.barbell_graph(5, 1)
    #community_gen = community.girvan_newman(sim1[0].graph)
    #top_lvl = next(community_gen)
    #next_lvl = next(community_gen)
    #print(sorted(map(sorted, next_lvl)))
    #part = community.label_propagation_communities(sim1[0].graph)
    #for el in part:print(el)
    #print(partition)
    fg = plt.figure()
    #print("Number of Communities: ", len(set(partition.values())))
    """for k, v in partition.items():
        sim1[0].graph.node[k]["louvain-val"] = v
    mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "magenta","cyan"]
    colors = [mypalette[sim1[0].graph.node[node]["louvain-val"] %9 ]  for node in sim1[0].graph.nodes()]
    
    plt.figure(figsize=(10,10))
    plt.axis('off')
    pos = nx.spring_layout(sim1[0].graph, scale=3)
    nx.draw_networkx_nodes(sim1[0].graph, pos, node_color=colors, node_size=40, label=True)
    nx.draw_networkx_edges(sim1[0].graph, pos, alpha=0.4)"""
    models.draw_model(sim1[0])
    
    #top_level_communities = next(communities_generator)
    #next_level_communities = next(communities_generator)
    #print(sorted(map(sorted, next_level_communities)))
    
    end = time.time()
    print(f'Time to complete: {end - start:.2f}s\n')
    ax = fg.gca()
    plt.show()
    

    #fg.savefig(fn, bbox_inches='tight')
    """plt.xlabel("timesteps")
    plt.ylabel("fraction of cooperators")
    plt.ylim((0, 1))
    

    for i in range(s):
        plt.plot(mods[i].ratio)
        
    #models.avgRadialDist(mods, 6, False)
    plt.show()"""
    