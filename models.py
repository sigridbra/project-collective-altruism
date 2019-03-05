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
import os

#Constants and Variables

states = [1, -1] #1 being cooperating, -1 being defecting

defectorUtility = -0.20 
 
politicalClimate=0.2 

selfWeight = 0.8

neighboursWeight = 0.5

#Helper
def decision(probability):
    return random.random() < probability

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def simulate(i, k):
    model = ScaleFreeModel(144, 2)
    res = model.runSim(k)
    #q.put(res)
    return res

class Agent:
    def __init__(self, state):
        self.state = state
        self.interactionsReceived = 0
        self.interactionsGiven = 0
    
    def consider(self, neighbour, neighboursWeight):
        self.interactionsReceived +=1
        neighbour.addInteractionGiven()
        weight = self.state*selfWeight + politicalClimate + defectorUtility + neighboursWeight*neighbour.state + random.uniform(-0.25, 0.25)
        
        #self.state = weight
        if(weight > 0):
            self.state = states[0]
        else:
            self.state = states[1]  
    
    def addInteractionGiven(self):
        self.interactionsGiven +=1
        
    def groupConsider(self, neighbourList):
        return
        
        
    def groupConsiderA(self, neighbour, neighboursWeight, neighbourList):
        nbNeighbours = len(neighbourList)
        nbCoop = 0
        for n in  neighbourList:
            if(n['agent'].state > 0): nbCoop += 1
        p = nbCoop/nbNeighbours
        self.interactionsReceived +=1
        neighbour.addInteractionGiven()
        if(neighbour.state <= 0):
            p=1-p
        
        
        weight = self.state*selfWeight + politicalClimate + defectorUtility + p*neighboursWeight*neighbour.state + random.uniform(-0.25, 0.25)
        
        #self.state = weight
        if(weight > 0):
            self.state = states[0]
        else:
            self.state = states[1]  
    
    
    def setState(self, newState):
        if(newState >= states[1] and newState <= states[0]):
            self.state = newState
        else:
            print("Error state outside state range: ", newState)
        

class Model:
    def __init__(self):
        self.graph = nx.Graph()
        self.ratio = []
        self.defectorDefectingNeighsList = []
        self.cooperatorDefectingNeighsList = []
        self.defectorDefectingNeighsSTDList = []
        self.cooperatorDefectingNeighsSTDList =[]
        self.pos = []
    
    def interact(self):
        nodeIndex = random.randint(0, len(self.graph) - 1)
        node = self.graph.nodes[nodeIndex]['agent']
        
        neighbours =  list(self.graph.adj[nodeIndex].keys())
        if(len(neighbours) == 0):
            return
        
        chosenNeighbourIndex = neighbours[random.randint(0, len(neighbours)-1)]
        chosenNeighbour = self.graph.nodes[chosenNeighbourIndex]['agent']
        
        weight = self.graph[nodeIndex][chosenNeighbourIndex]['weight']
        
        node.consider(chosenNeighbour, weight)
        
    def groupInteract(self):
        nodeIndex = random.randint(0, len(self.graph) - 1)
        node = self.graph.nodes[nodeIndex]['agent']
        
        neighbours =  list(self.graph.adj[nodeIndex].keys())
        if(len(neighbours) == 0):
            return
        
        chosenNeighbourIndex = neighbours[random.randint(0, len(neighbours)-1)]
        chosenNeighbour = self.graph.nodes[chosenNeighbourIndex]['agent']
        
        weight = self.graph[nodeIndex][chosenNeighbourIndex]['weight']
        
        neighbourList = [self.graph.nodes[i] for i in neighbours]
        node.groupConsiderA(chosenNeighbour, weight, neighbourList)
        
    def getAvgNumberOfDefectorNeigh(self):
        defectorFriendsList = []
        defectorNeighboursList = []
        for node in self.graph:
            agreeingNeighbours = 0
            neighbours = list(self.graph.adj[node])
            for neighbourIndex in neighbours:
                if self.graph.nodes[neighbourIndex]['agent'].state == self.graph.nodes[node]['agent'].state:
                    agreeingNeighbours += 1
            if self.graph.nodes[node]['agent'].state== 1:
                defectorNeighboursList.append(agreeingNeighbours) #defectorNeighboursList.append(agreeingNeighbours/len(neighbours))
            else:
                defectorFriendsList.append(agreeingNeighbours)
        
        defectoravg = mean(defectorFriendsList)
        cooperatoravg =mean(defectorNeighboursList)
        defectorSTD = stdev(defectorFriendsList)
        cooperatorSTD =stdev(defectorNeighboursList)
        return(defectoravg, cooperatoravg, defectorSTD, cooperatorSTD)
                
    
    def countCooperatorRatio(self):
        count = 0
        for node in self.graph:
            if self.graph.nodes[node]['agent'].state > 0:
                count+=1
        return count/len(self.graph)
 
    def runSim(self, k, groupInteract=False, drawModel = False, countNeighbours = False, gifname=None):
        
        if(drawModel):
            draw_model(self)
            filenames = []
        
        if(countNeighbours):
            (defectorDefectingNeighs,
             cooperatorDefectingFriends,
             defectorDefectingNeighsSTD,
             cooperatorDefectingFriendsSTD) = self.getAvgNumberOfDefectorNeigh()
            print("Defectors: avg: ", defectorDefectingNeighs, " std: ", defectorDefectingNeighsSTD)
            print("Cooperators: avg: ", cooperatorDefectingFriends, " std: ", cooperatorDefectingFriendsSTD)
    
        for i in range(k):
            if(groupInteract): self.groupInteract()
            else:
                self.interact()
            ratio = self.countCooperatorRatio()
            self.ratio.append(ratio)
            #self.politicalClimate += (ratio-0.5)*0.001 #change the political climate depending on the ratio of cooperators
            
            if(countNeighbours):
                (defectorDefectingNeighs,
                 cooperatorDefectingNeighs,
                 defectorDefectingNeighsSTD,
                 cooperatorDefectingNeighsSTD) = self.getAvgNumberOfDefectorNeigh()
                self.defectorDefectingNeighsList.append(defectorDefectingNeighs)
                self.cooperatorDefectingNeighsList.append(cooperatorDefectingNeighs)
                self.defectorDefectingNeighsSTDList.append(defectorDefectingNeighsSTD)
                self.cooperatorDefectingNeighsSTDList.append(cooperatorDefectingNeighsSTD)
            if(drawModel and (gifname != None) and (i % 10 == 0)):
                draw_model(self, True, i)
                filenames.append("plot" + str(i) +".png")
                
            #if(i % 10 == 0):
                #a = random.randint(0,n)
                #b = random.randint(0,n)
                #while(a==b):
                    #b = random.randint(0,n)
                    #weight = random.uniform(0.1, 0.9)
                    #model.graph.add_edge(a, b, weight = weight)
        if(drawModel and (gifname != None)):
            images = []
            for filename in filenames:
                images.append(imageio.imread(filename))
            imageio.mimsave("network" +gifname+ ".gif", images, duration=0.04167)
       
    
        if(countNeighbours):
            drawDefectingNeighbours(self.defectorDefectingNeighsList,
                                    self.cooperatorDefectingNeighsList,
                                    self.defectorDefectingNeighsSTDList,
                                    self.cooperatorDefectingNeighsSTDList, 
                                    gifname)
        
        return self.ratio

class GridModel(Model):
    def __init__(self, n):
        super().__init__()
        for i in range(n):
            for j in range (n):
                weight=random.uniform(0.1, 0.9)
                agent1 = Agent(states[random.randint(0,1)])
                self.graph.add_node(i*n+j, agent=agent1, pos=(i, j))
                self.pos.append((i, j))
                if(i!=0):
                    model.graph.add_edge(i*n+j, (i-1)*n+j, weight = weight)
                if(j!=0):
                    model.graph.add_edge(i*n+j, i*n+j-1, weight = weight)
    

class ScaleFreeModel(Model):
    def __init__(self, n, m):
        super().__init__()
        X = get_truncated_normal(0.5, 0.15, 0, 1)
        self.graph = nx.barabasi_albert_graph(n, m)
        for n in range (n):
                agent1 = Agent(states[random.randint(0,1)])
                self.graph.nodes[n]['agent'] = agent1
        edges = self.graph.edges() 
        for e in edges: 
            #weight=random.uniform(0.1, 0.9)
            weight=X.rvs(1)
            self.graph[e[0]][e[1]]['weight'] = weight 
        self.pos = nx.kamada_kawai_layout(self.graph)
        
class RandomModel(Model):
    def __init__(self, n, m):
        super().__init__()
        X = get_truncated_normal(0.5, 0.15, 0, 1)
        self.graph =nx.erdos_renyi_graph(n, 0.027972)
        for n in range (n):
                agent1 = Agent(states[random.randint(0,1)])
                self.graph.nodes[n]['agent'] = agent1
        edges = self.graph.edges() 
        for e in edges: 
            #weight=random.uniform(0.1, 0.9)
            weight=X.rvs(1)
            self.graph[e[0]][e[1]]['weight'] = weight 
        self.pos = nx.kamada_kawai_layout(self.graph)
    # bipartite_layout(G, nodes[, align, scale, …])	Position nodes in two straight lines.
    #circular_layout(G[, scale, center, dim])	Position nodes on a circle.
    #kamada_kawai_layout(G[, dist, pos, weight, …])	Position nodes using Kamada-Kawai path-length cost-function.
    #random_layout(G[, center, dim, seed])	Position nodes uniformly at random in the unit square.
    #rescale_layout(pos[, scale])	Return scaled position array to (-scale, scale) in all axes.
    ##shell_layout(G[, nlist, scale, center, dim])	Position nodes in concentric circles.
    #spring_layout(G[, k, pos, fixed, …])	Position nodes using Fruchterman-Reingold force-directed algorithm.
    #spectral_layout(G[, weight, scale, center, dim])	Position nodes using the eigenvectors of the graph Laplacian. 

def makeRandomModel(n, m, algorithm="ba"):
    model = Model()
    X = get_truncated_normal(0.5, 0.15, 0, 1)
    if(algorithm == "er"):
        model.graph =nx.erdos_renyi_graph(n, 0.027972)
    else:
        model.graph = nx.barabasi_albert_graph(n, m)
    for n in range (n):
            agent1 = Agent(states[random.randint(0,1)])
            model.graph.nodes[n]['agent'] = agent1
    edges = model.graph.edges() 
    for e in edges: 
        #weight=random.uniform(0.1, 0.9)
        weight=X.rvs(1)
        model.graph[e[0]][e[1]]['weight'] = weight 
    #pos = nx.nx_agraph.graphviz_layout(model.graph)
    #pos = graphviz_layout(model.graph)
    pos = nx.kamada_kawai_layout(model.graph)
# bipartite_layout(G, nodes[, align, scale, …])	Position nodes in two straight lines.
#circular_layout(G[, scale, center, dim])	Position nodes on a circle.
#kamada_kawai_layout(G[, dist, pos, weight, …])	Position nodes using Kamada-Kawai path-length cost-function.
#random_layout(G[, center, dim, seed])	Position nodes uniformly at random in the unit square.
#rescale_layout(pos[, scale])	Return scaled position array to (-scale, scale) in all axes.
##shell_layout(G[, nlist, scale, center, dim])	Position nodes in concentric circles.
#spring_layout(G[, k, pos, fixed, …])	Position nodes using Fruchterman-Reingold force-directed algorithm.
#spectral_layout(G[, weight, scale, center, dim])	Position nodes using the eigenvectors of the graph Laplacian. 

    model.pos = pos
    
    return model