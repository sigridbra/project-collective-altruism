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

#matplotlib.use('Agg')
mypalette = ["blue","red","green", "orange", "magenta","cyan",  "violet", "grey","yellow"]


pathfig = '~/Documents/Prosjek/Master/Comp/Week20/'
pathdata = '~/Documents/dev/prosjektoppgave/Week20/'
variables = [ 0.25, 0.35, 0.45, 0.55]
resSlopes = [ [] for i in range(6) ]
for i, v in enumerate(variables):
    filename = f'changepu-pu0.05-skew0.1-rd0.25-newpu{v}-30s-3000steps'
    sims = []
    sims.append(models.loadModels(Path(pathdata +filename + "-rand-cont").expanduser()))
    sims.append(models.loadModels(Path(pathdata +filename + "-cl-cont").expanduser()))      
    sims.append(models.loadModels(Path(pathdata +filename + "-rand-disc").expanduser()))
    sims.append(models.loadModels(Path(pathdata +filename + "-cl-disc").expanduser()))
    sims.append(models.loadModels(Path(pathdata +filename + "-grid-cont").expanduser()))
    sims.append(models.loadModels(Path(pathdata +filename + "-grid-disc").expanduser()))
    
    for j, sim in enumerate(sims):
        slopes = []
        for model in sim:

            try:
                lowIndex = next(ind for ind, val in enumerate(model.states) if val > 0)
            except: 
                print("Never above 0 for ", v)
                lowIndex = 500
                continue

            try:
                highIndex = next(ind for ind, val in enumerate(model.states) if val > 0.4)
            except: 
                print("never above 0.4 for ", v)
                highIndex = 2999

            slope = (model.states[highIndex] - model.states[lowIndex])/(highIndex - lowIndex)
            slopes.append(slope)
        slope = mean(slopes)
        
        resSlopes[j].append(slope)

results = []
#print(resSlopes)
#for i in range(6):
#    results.append([resSlopes[x][i] for x in range(len(variables))])
#print(results)
plt.figure()
#plt.ylim((0, 0.002))
plt.xlabel("Political Climate after 500")
plt.ylabel("Slope")
plt.title("Slope vs political climate after 500 for different models")
for i in range(6):
    #print(results[i])
    plt.scatter(variables, resSlopes[i], color=mypalette[i])    
#print(results)
plt.draw()
plt.show()


