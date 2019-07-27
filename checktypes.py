from multiprocessing import Pool
import models 
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
#from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
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
import dill
import community

#matplotlib.use('Agg')

#Constants and Variables
plt.rcParams["svg.fonttype"] = "none"
#plt.rcParams["font.size"] = 16
plt.rcParams["savefig.directory"] = "Master/Comp/New"
#plt.rcParams["savefig.format"] = "svg"

s =100 #10^3 
if __name__ ==  '__main__': 
    num_processors = 8
    start = time.time()
    p=Pool( processes = num_processors)
   
    variables = [ 0.6]#, 0.125, 0.0625 ]
    for v in variables:        
        argList = []
        argList.append({"continuous": True, "type" : "rand", "selfWeight": v, "influencers":0})
        argList.append({"continuous": True, "type" : "cl", "selfWeight": v, "influencers":0})
        argList.append({"continuous": True, "type" : "grid", "selfWeight": v, "influencers":0})
        #argList.append({"continuous": False, "type" : "rand", "selfWeight": v, "influencers":0})
        #argList.append({"continuous": False, "type" : "cl", "selfWeight": v, "influencers":0})
        #argList.append({"continuous": True, "type" : "sf", "selfWeight": v, "influencers":0})
        #argList.append({"continuous": False, "type" : "sf", "selfWeight": v, "influencers":0})
        
        #argList.append({"continuous": False, "type" : "grid", "selfWeight": v, "influencers":0})
        #print("rand")
        titleList = ["Random C", "Clustered C", "Grid C", "Random D", "Clustered D", "Grid D"]        
        filenameList = ["-rand-cont", "-cl-cont", "-grid-cont", "-rand-disc", "-cl-disc", "-grid-disc"]
        for i in range(3):
            sim = p.starmap(models.simulate, zip(range(s), repeat(argList[i])))
            clust = [nx.average_clustering(model.graph) for model in sim]
            modularity = [community.modularity(model.partition, model.graph) for model in sim]
            asso = [nx.degree_assortativity_coefficient(model.graph) for model in sim]
            dist = []
            for model in sim:
                #print(nx.info(model.graph))
                try:
                    d = nx.average_shortest_path_length(model.graph)
                    dist.append(d)
                except:
                    print("Not connected ", titleList[i])
                    continue

            print(" ")
            print(titleList[i])
            print("modularity: ", mean(modularity) )
            print("clustering: ", mean(clust) )
            print("asso: ", mean(asso) )
            print("dist: ", mean(dist) )

            
        simtime= time.time()
        print(f'Time to simulate: {simtime-start}s\n')
        
        
    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Time to complete: {mins:5.0f} mins {sec}s\n')
