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

if __name__ ==  '__main__': 
    model = models.ClusteredPowerlawModel(20, 3)
    plt.figure(0)
    models.draw_model(model)
    plt.show()
    model.runSim(200, gifname="master-detailed")