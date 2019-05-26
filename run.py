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

matplotlib.use('Agg')

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
    pathfig = '~/Documents/Prosjek/Master/Comp/balanced/'
    pathdata = '~/Documents/dev/prosjektoppgave/balanced/'
    variables = [ 0.40, 0.6, 0.8]#, 0.125, 0.0625 ]
    for v in variables:
        filename = f'100s-balanced-4000-testsw{v}'
        fg1= plt.figure("states", frameon=False, figsize=(14,7))
        fg2 = plt.figure("cross", frameon=False)
        #fg22 = plt.figure("cross2")
        fn1 = Path(pathfig + filename + '.svg').expanduser()
        fn2 = Path(pathfig + filename +'-crossection.svg').expanduser()  
        fn22 = Path(pathfig + filename +'-crossection2.svg').expanduser()  
        fn3 = Path(pathfig + filename +'-avgagreeingfriends.svg').expanduser()        
        #fn = Path(f'~/Documents/Prosjek/Master/Comp/New/144-k10-wi{v}-50s-skew0.05-comparewi.svg').expanduser()        
        args1 = {"continuous": True, "type" : "rand", "selfWeight": v}
        args2 = {"continuous": True, "type" : "cl", "selfWeight": v}
        args3 = {"continuous": False, "type" : "rand", "selfWeight": v}
        args4 = {"continuous": False, "type" : "cl", "selfWeight": v}
        args5 = {"continuous": True, "type" : "sf", "selfWeight": v}
        args6 = {"continuous": False, "type" : "sf", "selfWeight": v}
        args7 = {"continuous": True, "type" : "grid", "selfWeight": v}
        args8 = {"continuous": False, "type" : "grid", "selfWeight": v}
        #print("rand")        
        sim1 = p.starmap(models.simulate, zip(range(s), repeat(args1)))
        #models.saveModels(sim1,  Path(pathdata +filename + "-rand-cont").expanduser())
        ##print("cl")  
        sim2 = p.starmap(models.simulate, zip(range(s), repeat(args2)))
        ###print("rand")  
        sim3 = p.starmap(models.simulate, zip(range(s), repeat(args3)))
        ##print("cl")  
        sim4 = p.starmap(models.simulate, zip(range(s), repeat(args4)))
        #print("sf")  
        #sim5 = p.starmap(models.simulate, zip(range(s), repeat(args5)))
        #print("sf")  
        #sim6 = p.starmap(models.simulate, zip(range(s), repeat(args6)))
        #print("sf")  
        sim7 = p.starmap(models.simulate, zip(range(s), repeat(args7)))
        #print("sf")  
        sim8 = p.starmap(models.simulate, zip(range(s), repeat(args8)))
        
        simtime= time.time()
        print(f'Time to simulate: {simtime-start}s\n')
        plt.figure("states")
        fg1.subplots(nrows=1, ncols=2 )
        #print(nx.info(sim1[1].graph))  
        lines = []
        lines.append(models.drawAvgState(sim1, avg=True, pltNr=1, title="Random C", clusterSD=True))
        lines.append(models.drawAvgState(sim2, avg=True, pltNr=2, title="Clustered C",clusterSD=True ))
        lines.append(models.drawAvgState(sim7, avg=True, pltNr=7, title="Grid C", clusterSD=True))
        lines.append(models.drawAvgState(sim3, avg=True, pltNr=3, title="Random D",clusterSD=True))
        lines.append(models.drawAvgState(sim4, avg=True, pltNr=4, title="Clustered D",clusterSD=True ))
        
        #lines.append(models.drawAvgState(sim5, avg=True, pltNr=5, title="sf cont",clusterSD=True))
        #lines.append(models.drawAvgState(sim6, avg=True, pltNr=6, title="sf Dc",clusterSD=True ))
        
        lines.append(models.drawAvgState(sim8, avg=True, pltNr=8, title="Grid D", clusterSD=True))
        
        current_handles, current_labels = plt.gca().get_legend_handles_labels()
        #print(current_labels)
        avgs_handles = [current_handles[i*3] for i in range(6)]
        avgs_labels = [ current_labels[i*3] for i in range(6)]
        sd_handles = [current_handles[i*3 + 1] for i in range(6)]
        sd_labels = [ current_labels[i*3+1] for i in range(6)]
        cl_handles = [current_handles[i*3+2] for i in range(6)]
        cl_labels = [ current_labels[i*3+2] for i in range(6)]
        handles = avgs_handles + sd_handles + cl_handles
        labels = avgs_labels + sd_labels + cl_labels
        
        plt.legend(handles, labels, ncol = 3,  fontsize='medium', columnspacing=0.25, handlelength=1, title="Avg    SD    Com.")
        models.drawCrossSection(sim1)
        models.drawCrossSection(sim2, pltNr=2)
        models.drawCrossSection(sim3, pltNr=3)
        models.drawCrossSection(sim4, pltNr=4)
        #models.drawCrossSection(sim5, pltNr=5)
        models.drawCrossSection(sim7, pltNr=7)
        models.drawCrossSection(sim8, pltNr=8)
        plt.draw()
        #plt.show()
        fg1.savefig(fn1)
        plt.clf()
        plt.close()
        print("feridg med 1, ", v)
        plt.figure("cross")
        fg2.subplots(nrows=1, ncols=3 )
        #models.drawCrossSection(sim5, pltNr=5)
        #models.drawCrossSection(sim6, pltNr=6)
        models.drawClusterState(sim1, pltNr=1)
        models.drawClusterState(sim2, pltNr=2)
        models.drawClusterState(sim3, pltNr=3)
        models.drawClusterState(sim4, pltNr=4)
        models.drawClusterState(sim7, pltNr=7)
        models.drawClusterState(sim8, pltNr=8)
        models.drawClusterState(sim1, pltNr=1, step=500, subplot=1)
        models.drawClusterState(sim2, pltNr=2, step=500, subplot=1)
        models.drawClusterState(sim3, pltNr=3, step=500, subplot=1)
        models.drawClusterState(sim4, pltNr=4, step=500, subplot=1)
        models.drawClusterState(sim7, pltNr=7, step=500, subplot=1)
        models.drawClusterState(sim8, pltNr=8, step=500, subplot=1)
        models.drawClusterState(sim1, pltNr=1, step=1000, subplot=2)
        models.drawClusterState(sim2, pltNr=2, step=1000, subplot=2)
        models.drawClusterState(sim3, pltNr=3, step=1000, subplot=2)
        models.drawClusterState(sim4, pltNr=4, step=1000, subplot=2)
        models.drawClusterState(sim7, pltNr=7, step=1000, subplot=2)
        models.drawClusterState(sim8, pltNr=8, step=1000, subplot=2)
        #models.drawClustersizes(sim1)
        #models.drawClustersizes(sim2, pltNr=2)
        #models.drawClustersizes(sim3, pltNr=3)
        #models.drawClustersizes(sim4, pltNr=4)
        

        plt.draw()
        fg2.savefig(fn2)
        plt.clf()
        plt.close()
        print("feridg med 2, ", v)
        
        fg3 = plt.figure("agreeingfriends", frameon=False)
        models.drawAvgNumberOfAgreeingFriends(sim1, pltNr=1)
        models.drawAvgNumberOfAgreeingFriends(sim2, pltNr=2)
        models.drawAvgNumberOfAgreeingFriends(sim3, pltNr=3)
        models.drawAvgNumberOfAgreeingFriends(sim4, pltNr=4)
        models.drawAvgNumberOfAgreeingFriends(sim7, pltNr=7)
        models.drawAvgNumberOfAgreeingFriends(sim8, pltNr=8)
        plt.draw()
        
        fg3.savefig(fn3)
        print("feridg med 3, ", v)
        plt.clf()
        plt.close('all')
        models.saveModels(sim1,  Path(pathdata +filename + "-rand-cont").expanduser())
        models.saveModels(sim2,  Path(pathdata +filename + "-cl-cont").expanduser())
        models.saveModels(sim3,  Path(pathdata +filename + "-rand-disc").expanduser())
        models.saveModels(sim4,  Path(pathdata +filename + "-cl-disc").expanduser())
        models.saveModels(sim7,  Path(pathdata +filename + "-grid-cont").expanduser())
        models.saveModels(sim8,  Path(pathdata +filename + "-grid-disc").expanduser())
        
    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Time to complete: {mins:5.0f} mins {sec}s\n')
