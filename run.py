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
 
if __name__ ==  '__main__': 
    plt.rcParams["svg.fonttype"] = "none"

    #Constants and Variables
    numberOfSimulations =5 #10^3
    numberOfProcessors = 4 #Prosesser kan endres

    start = time.time()
    pool=Pool( processes = numberOfProcessors)
    
    # ----------PATH TO SAVE FIGURES AND DATA-----------
    #pathFig = '~/Documents/Prosjek/Master/Paper/SecondRound/'
    #pathData = '~/Documents/dev/prosjektoppgave/paper/'
    pathFig = '~/phd-stuff/research/altruism/project-collective-altruism/figs/'
    pathData = '~/phd-stuff/research/altruism/project-collective-altruism/data/'


    variables = [ 0]#, 0.125, 0.0625 ]
    for v in variables:
        print("Started variable: ", v)
        filename = f'sim2c-50s-4000-sw0_6-2opposing-zealots'
        fg1= plt.figure("states", frameon=False, figsize=(14,7))
        fg2 = plt.figure("cross", frameon=False)
        fg22 = plt.figure("cross2", frameon=False)
        fg3 = plt.figure("agreeingfriends", frameon=False)
        #fg22 = plt.figure("cross2")
        fn1 = Path(pathFig + filename + '.svg').expanduser()
        fn2 = Path(pathFig + filename +'-crossection.svg').expanduser()  
        fn22 = Path(pathFig + filename +'-crossection2.svg').expanduser()  
        fn3 = Path(pathFig + filename +'-avgagreeingfriends.svg').expanduser()        
        #fn = Path(f'~/Documents/Prosjek/Master/Comp/New/144-k10-wi{v}-50s-skew0.05-comparewi.svg').expanduser()        
        argList = []
        #argList.append({"continuous": True, "type" : "rand", "influencers":0})
        #argList.append({"continuous": True, "type" : "cl", "influencers":0})
        #argList.append({"continuous": False, "type" : "rand", "influencers":0})
        #argList.append({"continuous": False, "type" : "cl", "influencers":0})
        argList.append({"continuous": True, "type" : "sf", "influencers":2})
        #argList.append({"continuous": False, "type" : "sf", "influencers":0})
        argList.append({"continuous": True, "type" : "grid", "influencers":2})
        #argList.append({"continuous": False, "type" : "grid", "influencers":0})
        #print("rand")
        titleList = ["Scale Free", "Grid"]        
        filenameList = ["-sf", "-grid"]
        for i in range(len(argList)):
            sim = pool.starmap(models.simulate, zip(range(numberOfSimulations), repeat(argList[i])))
            plt.figure("states")
            fg1.subplots(nrows=1, ncols=2 )
            models.drawAvgState(sim, avg=True, pltNr=i+1, title=titleList[i], clusterSD=True)
            models.drawCrossSection(sim, pltNr=i+1)
            plt.draw()
            plt.figure("cross")
            fg2.subplots(nrows=1, ncols=3 )
            models.drawClusterState(sim, pltNr=i+1, step=800, subplot=3)
            models.drawClusterState(sim, pltNr=i+1, step=400, subplot=1)
            models.drawClusterState(sim, pltNr=i+1, step=600, subplot=2)
            plt.draw()
            plt.figure("cross2")
            fg22.subplots(nrows=1, ncols=3 )
            models.drawClusterState(sim, pltNr=i+1)
            models.drawClusterState(sim, pltNr=i+1, step=1000, subplot=1)
            models.drawClusterState(sim, pltNr=i+1, step=1500, subplot=2)
            plt.draw()
            plt.figure("agreeingfriends")
            models.drawAvgNumberOfAgreeingFriends(sim, pltNr=i+1)
            plt.draw()
            print("Finished with ", titleList[i])
            #models.saveModels(sim, Path(pathData +filename + filenameList[i]).expanduser())
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
        print("feridg med 1, ", v)
        
        plt.figure("cross")
        fg2.savefig(fn2)
        plt.clf()
        plt.close()
        print("feridg med 2, ", v)

        plt.figure("cross2")
        fg22.savefig(fn22)
        plt.clf()
        plt.close()
        print("feridg med 22, ", v)
        
        fg3 = plt.figure("agreeingfriends", frameon=False)
        fg3.savefig(fn3)
        print("feridg med 3, ", v)
        plt.clf()
        plt.close()
        
    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Time to complete: {mins:5.0f} mins {sec}s\n')
