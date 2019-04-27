import networkx as nx
import random
import pylab as py
from math import floor
from copy import deepcopy
import community
import matplotlib.pyplot as plt

G = nx.complete_graph(20)
edges = deepcopy(G.edges())
for edge in edges:
    try:
        if floor(edge[0]/5.)!=floor(edge[1]/5.):
            if random.random()<0.95:
                G.remove_edge(edge[0],edge[1])
    except:
        break
partition = community.best_partition(G)
for k, v in partition.items():
    G.node[k]["louvain-val"] = v
    mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "magenta","cyan"]
colors = [mypalette[G.node[node]["louvain-val"] %9 ]  for node in G.nodes()]
edge_col = [mypalette[G.node[node]["louvain-val"]+1 %9 ]  for node in G.nodes()]

nx.draw_spring(G, node_color=colors)
py.show()


fixedpos = {1:(0,0), 6:(1,1), 11:(1,0), 16:(0,1)}
pos = nx.spring_layout(G, fixed = fixedpos.keys(), pos = fixedpos)
#print(edge_col)
nx.draw_networkx(G, pos=pos, node_color=colors, linewidths=2)
ax = plt.gca() # to get the current axis
ax.collections[0].set_edgecolor(edge_col) 
plt.show()