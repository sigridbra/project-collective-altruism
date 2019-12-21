
#
# Some notes on implementation for better or worse
# 
# Original program written by Sigrid Bratsberg, credit where credit is due! 
# 
# Switching from printing to a single file to making multiple out outs for a given
# set of input values is enabled by decommenting the newvar lines
# 
# TODO
# Someone should move all the parameters to the main program
#

import pandas
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

#random.seed(1574705741)
#np.random.seed(1574705741)

if __name__ ==  '__main__': 
    plt.rcParams["svg.fonttype"] = "none"

    #Constants and Variables
    numberOfSimulations =1 #10^3
    numberOfProcessors = 1 #Prosesser kan endres

    start = time.time()
    pool=Pool( processes = numberOfProcessors)
    
    # ----------PATH TO SAVE FIGURES AND DATA-----------
    #pathFig = '~/Documents/Prosjek/Master/Paper/SecondRound/'
    #pathData = '~/Documents/dev/prosjektoppgave/paper/'
    pathFig = '~/phd-stuff/research/project-collective-altruism/figs/'
    pathData = '~/phd-stuff/research/project-collective-altruism/data/'
    
    modelargs=models.getargs()  # requires models.py to be imported
    runs = 100   ## has to be even for multiple runs also n is actually n-1 because I'm bad

    ## comment out all below for single run
    var = 'skew'
    
    ## log grid, only valid on range [-1,1]

    steps = int(runs/2)
    start = modelargs[var]
    endup = -0.05
    enddw = -0.45
    logendup = np.log(endup+(1.0-start))
    logenddw = np.log(enddw+(1.0-start))
    stepup = logendup / steps
    stepdw = logenddw / steps

    gridup = np.array([])
    griddw = np.array([])

    for k in range (steps):
        pt = np.exp(stepup*k)
        gridup = np.append(gridup,pt)
    
    for k in range (steps):
        pt = np.exp(stepdw*k)
        griddw = np.append(griddw,pt)

    gridup = gridup - (1.0-start)
    griddw = griddw - (1.0-start)

    griddw = griddw[1:]
    griddw = np.flip(griddw)

    grid = np.append(griddw,gridup)

    print (grid)

