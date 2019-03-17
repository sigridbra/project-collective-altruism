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

#Constants and Variables



model = models.NewmanModel(144)
#res = model.runSim(3000, gifname="scaleFreecontinu3")
models.draw_model(model, save=True, filenumber = "Newmann")
print(nx.average_clustering(model.graph))
#plt.plot(res)
#plt.show()
    