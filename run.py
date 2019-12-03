
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
    numberOfSimulations =100 #10^3
    numberOfProcessors = 4 #Prosesser kan endres

    start = time.time()
    pool=Pool( processes = numberOfProcessors)
    
    # ----------PATH TO SAVE FIGURES AND DATA-----------
    #pathFig = '~/Documents/Prosjek/Master/Paper/SecondRound/'
    #pathData = '~/Documents/dev/prosjektoppgave/paper/'
    pathFig = '~/phd-stuff/research/project-collective-altruism/figs/'
    pathData = '~/phd-stuff/research/project-collective-altruism/data/'
    
    modelargs=models.getargs()  # requires models.py to be imported
    runs = 2   ## has to be even for multiple runs also n is actually n-1 because I'm bad
   
    ## comment out all below for single run
    #var = 'skew'
    #
    ### log grid, only valid on range [-1,1]

    #steps = int(runs/2)
    #start = modelargs[var]
    #endup = 1.0
    #enddw = -1.0
    #logendup = np.log(endup+(1.0-start))
    #logenddw = np.log(enddw+(1.0-start))
    #stepup = logendup / steps
    #stepdw = logenddw / steps

    #gridup = np.array([])
    #griddw = np.array([])

    #for k in range (steps):
    #    pt = np.exp(stepup*k)
    #    gridup = np.append(gridup,pt)
    #
    #for k in range (steps):
    #    pt = np.exp(stepdw*k)
    #    griddw = np.append(griddw,pt)

    #gridup = gridup - (1.0-start)
    #griddw = griddw - (1.0-start)

    #griddw = griddw[1:]
    #griddw = np.flip(griddw)

    #grid = np.append(griddw,gridup)

    #print (grid)

    for run in range(runs-1):
        print("Started iteration: ", run)
        #newvar = grid[run]
        
        #filename = f'sim2c-50s-4000-sw0_6-2opposing-zealots'
        filename="pol{}_skew{}_sd{}_random{}_tsteps{}_{}".format(modelargs["newPoliticalClimate"],modelargs["skew"],modelargs["initSD"],modelargs["randomness"],modelargs["timesteps"],modelargs["type"])
        fg1= plt.figure("states", frameon=False)
        #fg1= plt.figure("states", frameon=False, figsize=(14,7))
        #fg2 = plt.figure("cross", frameon=False)
        #fg22 = plt.figure("cross2", frameon=False)
        #fg3 = plt.figure("agreeingfriends", frameon=False)
        #fg22 = plt.figure("cross2")
        fn1 = Path(pathFig + filename + '.png').expanduser()
        #fn2 = Path(pathFig + filename +'-crossection.svg').expanduser()  
        #fn22 = Path(pathFig + filename +'-crossection2.svg').expanduser()  
        #fn3 = Path(pathFig + filename +'-avgagreeingfriends.svg').expanduser()        
        #fn = Path(f'~/Documents/Prosjek/Master/Comp/New/144-k10-wi{v}-50s-skew0.05-comparewi.svg').expanduser()
        """ I moved network type specification to models.py for consistancy """
        argList = []
        clstate = True
        #argList.append({"continuous": True, "type" : "rand", "influencers":0})
        #argList.append({"continuous": True, "type" : "cl", "influencers":0})
        #argList.append({"continuous": False, "type" : "rand", "influencers":0})
        #argList.append({"continuous": False, "type" : "cl", "influencers":0})
        #argList.append({"continuous": True, "type" : "sf", "influencers":0})
        #argList.append({"continuous": False, "type" : "sf", "influencers":0})
        #argList.append({"continuous": True, "influencers": 0, "skew": newvar})
        argList.append({"continuous": True, "influencers": 0})
        #argList.append({"continuous": False, "type" : "grid", "influencers":0})
        #print("rand")
        titleList = ["clustered"]        
        filenameList = ["-cl"]
        for i in range(len(argList)):
            sim = pool.starmap(models.simulate, zip(range(numberOfSimulations), repeat(argList[i])))
            plt.figure("states")
            #fg1.subplots(nrows=1, ncols=2 )
            #fg1.subplots()
            models.drawAvgState(sim, avg=True, pltNr=i, title=titleList[i], clusterSD=True)
            #models.drawCrossSection(sim, pltNr=i+1)
            #plt.draw()
            #plt.figure("cross")
            #fg2.subplots(nrows=1, ncols=3 )
            #models.drawClusterState(sim, pltNr=i+1, step=800, subplot=3)
            #models.drawClusterState(sim, pltNr=i+1, step=400, subplot=1)
            #models.drawClusterState(sim, pltNr=i+1, step=600, subplot=2)
            #plt.draw()
            #plt.figure("cross2")
            #fg22.subplots(nrows=1, ncols=3 )
            #models.drawClusterState(sim, pltNr=i+1)
            #models.drawClusterState(sim, pltNr=i+1, step=1000, subplot=1)
            #models.drawClusterState(sim, pltNr=i+1, step=1500, subplot=2)
            #plt.draw()
            #plt.figure("agreeingfriends")
            #models.drawAvgNumberOfAgreeingFriends(sim, pltNr=i+1)
            #plt.draw()
            print("Finished with ", titleList[i])
            #models.saveModels(sim, Path(pathData + filename + filenameList[i]).expanduser())
            #fname = './data/multiskew{}.csv'.format(newvar)
            #fname = './data/states.csv'
            fname = './data/runs.csv'
            #models.saveavgdata(sim, fname,clstate)
            models.savesubdata(sim, fname)
        simtime= time.time()
        print(f'Time to simulate: {simtime-start}s\n')

        plt.figure("states")
        #plt.show()
        """current_handles, current_labels = fg1.gca().get_legend_handles_labels()
        print(current_labels)
        avgs_handles = [current_handles[i*3] for i in range(2)]
        avgs_labels = [ current_labels[i*3] for i in range(2)]
        sd_handles = [current_handles[i*3 + 1] for i in range(2)]
        sd_labels = [ current_labels[i*3+1] for i in range(2)]
        cl_handles = [current_handles[i*3+2] for i in range(2)]
        cl_labels = [ current_labels[i*3+2] for i in range(2)]
        labels = [" " for i in range(4)]+ titleList
        handles = avgs_handles + sd_handles + cl_handles
        labels = avgs_labels + sd_labels + cl_labels
        
        plt.legend(handles, labels, ncol = 3,  fontsize='medium', columnspacing=0.25, handlelength=1, title="Avg    SD    Com.")
        """
        
        plt.legend()
        fg1.savefig(fn1)
        plt.clf()
        plt.close()
        #print("feridg med 1, ", v)
        
        #plt.figure("cross")
        #fg2.savefig(fn2)
        #plt.clf()
        #plt.close()
        #print("feridg med 2, ", v)

        #plt.figure("cross2")
        #fg22.savefig(fn22)
        #plt.clf()
        #plt.close()
        #print("feridg med 22, ", v)
        #
        #fg3 = plt.figure("agreeingfriends", frameon=False)
        #fg3.savefig(fn3)
        #print("feridg med 3, ", v)
        #plt.clf()
        #plt.close()

    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Time to complete: {mins:5.0f} mins {sec}s\n')
