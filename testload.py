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


pathfig = '~/Documents/Prosjek/Master/Comp/Week19_2/'
pathdata = '~/Documents/dev/prosjektoppgave/Week19_2/'
filename = f'pu0.05-skew0.1-rd0.25-initsd0.25-50s-3000steps-saved'
start = time.time()
sim1 = models.loadModels(Path(pathdata +filename + "-rand-cont1").expanduser())
sim2 = models.loadModels(Path(pathdata +filename + "-cl-cont").expanduser())
sim3 = models.loadModels(Path(pathdata +filename + "-rand-disc").expanduser())
sim4 = models.loadModels(Path(pathdata +filename + "-cl-cont").expanduser())
sim7 = models.loadModels(Path(pathdata +filename + "-grid-cont").expanduser())
sim8 = models.loadModels(Path(pathdata +filename + "-grid-disc").expanduser())

end = time.time()

mins = (end - start) / 60
sec = (end - start) % 60
print(f'Time to complete: {mins:5.0f} mins {sec}s\n')
fg2 = plt.figure("cross")

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
plt.draw()
#fg2.savefig(fn2)

#plt.legend()
#models.drawAvgState(mods, avg=False, pltNr=1, title="grid cont", clusterSD=True)
#models.drawCrossSection(mods)
#models.drawClusterState(mods, pltNr=1)
plt.draw()
plt.show()