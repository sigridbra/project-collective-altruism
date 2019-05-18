import numpy as np
#from operator import itemgetter
#import heapq
import random
import matplotlib.pyplot as plt
import matplotlib.colors as col
#from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from copy import deepcopy
import seaborn as sns
from statistics import stdev, mean
import imageio
import networkx as nx
from scipy.stats import truncnorm
import os
#from functools import reduce
import time
import community

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

#Constants and Variables

states = [1, -1] #1 being cooperating, -1 being defecting
defectorUtility = -0.20 
politicalClimate= 0.25
newPoliticalClimate = 0.25 
selfWeight = 0.6
d = 5 #degree
s = 10
k=3000 #10^6
continuous = True
skew = -0.40
initSD = 0.25
mypalette = ["blue","red","green", "orange", "violet", "grey", "magenta","cyan", "yellow"]
randomness = 0.25

args = {"defectorUtility" : defectorUtility, 
        "politicalClimate" : politicalClimate, 
        "selfWeight": selfWeight, "d":d, 
        "s": s, "k" : k, "continuous" : continuous, "type" : "cl", "skew": skew, "initSD": initSD, "newPoliticalClimate": newPoliticalClimate}

def simulate(i, newArgs):
    setArgs(newArgs)
    global args
    start = time.time()
    X = get_truncated_normal(0.5, 0.15, 0, 1)
    
    S = get_truncated_normal(args["skew"], args["initSD"], -1, 1)
    
    if(args["type"] == "cl"):
        model =ClusteredPowerlawModel(144, args["d"], skew=args["skew"], X=X, S=S)
    elif(args["type"] == "sf"):
        model = ScaleFreeModel(144, args["d"], skew=args["skew"], X=X, S=S)
    elif(args["type"] == "grid"):
        model = GridModel(12, skew=args["skew"], X=X, S=S)
    elif(args["type"] == "rand"):
        model = RandomModel(144, args["d"], skew=args["skew"], X=X, S=S)
    else:
        model = RandomModel(144, args["d"],  X=X, S=S)
    #simtime= time.time()
    #print(f'Time to make model: {simtime-start}s\n')
    
    res = model.runSim(args["k"], clusters=True)
    #print("assortativity: ", nx.degree_assortativity_coefficient(model.graph))
    #print("modluarity: ", community.modularity(model.partition, model.graph))
    #print(f'Time to make model: {simtime-start}s\n')
    return model

#Helper
def setArgs(newArgs):
    global args
    #args = {"defectorUtility" : defectorUtility, 
    #    "politicalClimate" : politicalClimate, 
    #    "selfWeight": selfWeight, "d":d, 
    #    "s": s, "k" : k, "continuous" : continuous, "type" : "cl", "skew": skew, "initSD": initSD, "newPoliticalClimate": newPoliticalClimate}
    for arg, value in newArgs.items():
        args[arg] = value
    


def decision(probability):
    return random.random() < probability


def getRandomExpo():
    x = np.random.exponential(scale=0.6667)-1
    if(x>1): return 1
    elif (x< -1): return -1
    return x

class Agent:
    def __init__(self, state, selfWeight):
        self.state = state
        self.interactionsReceived = 0
        self.interactionsGiven = 0
        self.selfWeight = selfWeight
    
    def consider(self, neighbour, neighboursWeight, politicalClimate):
        self.interactionsReceived +=1
        neighbour.addInteractionGiven()
        if(self.selfWeight >= 1): return
        global args
        weight = self.state*self.selfWeight + politicalClimate + args["defectorUtility"] + neighboursWeight*neighbour.state #+ random.uniform(-0.25, 0.25)
        #weight =  politicalClimate + defectorUtility + neighboursWeight*neighbour.state #+ random.uniform(-0.25, 0.25)
        #print("neighbours weight: ", neighboursWeight, " neighbours state: ", neighbour.state, " weight: ", weight)
        
        
        if(args["continuous"]):
            #self.state = weight
            #print("x: ", weight)
            p1 = (randomness+weight)*(1/(2*randomness))
            #print(p1)
            if(p1 <0): p1 = 0
            if(p1 > 1): p1=1
            #print("Self.state: ",self.state, ", neighbour: ", neighbour.state,", p1: ", p1)
            delta = (1/2)*(-self.state+1)*(p1) - ((1/2)*(self.state+1))*(1-p1)
            #print("delta: ", delta)
            #print("  ")
            increment = 2*delta*abs(self.state-neighbour.state)

            self.state += increment
            #Truncate values    
            if(self.state > 1):
                  self.state = states[0]
            elif(self.state <-1):
                self.state = states[1]       
        else:
            if(weight + random.uniform(-randomness, randomness)  > 0):
                self.state = states[0]
            else:
                self.state = states[1]  

    def addInteractionGiven(self):
        self.interactionsGiven +=1
        
    def groupConsider(self, neighbourList):
        return
        
    def groupConsiderA(self, neighbour, neighboursWeight, neighbourList, continuous=False):
        nbNeighbours = len(neighbourList)
        nbCoop = 0
        for n in  neighbourList:
            if(n['agent'].state > 0): nbCoop += 1
        p = nbCoop/nbNeighbours
        self.interactionsReceived +=1
        neighbour.addInteractionGiven()
        if(neighbour.state <= 0):
            p=1-p
        
        weight = self.state*selfWeight + politicalClimate + defectorUtility + p*neighboursWeight*neighbour.state #+ random.uniform(-0.25, 0.25)
        
        if(continuous):
            self.state = weight
            if(weight > 1):
                  self.state = states[0]
            elif(weight <-1):
                self.state = states[1] 
        else:
            if(weight > 0):
                self.state = states[0]
            else:
                self.state = states[1]  
     
    def groupConsiderB(self, impact, continuous = False):
        print("impact: ", impact, "state: ", self.state)
        weight = self.state*selfWeight + politicalClimate + defectorUtility + impact #+ random.uniform(-0.25, 0.25)
        if(continuous):
            self.state = weight
            if(weight > 1):
                  self.state = states[0]
            elif(weight <-1):
                self.state = states[1]  
        else:
            if(weight >= 0):
                self.state = states[0]
            else:
                self.state = states[1] 
        print("new state: ", self.state, "\n")

    #def groupConsiderC(self, neighbours):

    
    
    def setState(self, newState):
        if(newState >= states[1] and newState <= states[0]):
            self.state = newState
        else:
            print("Error state outside state range: ", newState)
        

class Model:
    def __init__(self, X = None, S=None):
        global args
        self.graph = nx.Graph()
        self.politicalClimate = args["politicalClimate"]
        self.ratio = []
        self.states = []
        self.statesds = []
        self.defectorDefectingNeighsList = []
        self.cooperatorDefectingNeighsList = []
        self.defectorDefectingNeighsSTDList = []
        self.cooperatorDefectingNeighsSTDList =[]
        self.pos = []
        self.X = X
        self.S = S
        self.clusteravg = []
        self.clusterSD = []
        self.NbAgreeingFriends = []
        self.avgNbAgreeingList = []
    
    def interact(self):
        nodeIndex = random.randint(0, len(self.graph) - 1)
        #print("in interact: ", nodeIndex)
        node = self.graph.nodes[nodeIndex]['agent']
        neighbours =  list(self.graph.adj[nodeIndex].keys())
        if(len(neighbours) == 0):
            return nodeIndex
        
        chosenNeighbourIndex = neighbours[random.randint(0, len(neighbours)-1)]
        chosenNeighbour = self.graph.nodes[chosenNeighbourIndex]['agent']
        weight = self.graph[nodeIndex][chosenNeighbourIndex]['weight']
        
        node.consider(chosenNeighbour, weight, self.politicalClimate)
        return nodeIndex
        
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
        
    def groupInteractB(self):
        nodeIndex = random.randint(0, len(self.graph) - 1)
        node = self.graph.nodes[nodeIndex]['agent']
        print("Node: ", nodeIndex)
        neighbours =  list(self.graph.adj[nodeIndex].keys())
        print(neighbours)
        if(len(neighbours) == 0):
            return
        
        impact = 0
        for n in neighbours:
            neighbour = self.graph.nodes[n]['agent']
            weight = self.graph[nodeIndex][n]['weight']
            impact += neighbour.state * weight
        
        impact = impact/len(neighbours)
        
        node.groupConsiderB(impact)

    def findNbAgreeingFriends(self, nodeIdx = None):
        global args
        nb = []
        
        if(args["continuous"]):
            for nodeIdx in self.graph.nodes:
                state = self.graph.nodes[nodeIdx]['agent'].state
                neighbours = list(self.graph.adj[nodeIdx])
                neighStates = [self.graph.nodes[n]['agent'].state for n in neighbours ]
                if(len(neighbours) == 0):
                    nb.append(0)
                    continue
                x = abs((mean(neighStates)+state))/2
                nb.append(x)
        else:
            for nodeIdx in self.graph.nodes:
                state = self.graph.nodes[nodeIdx]['agent'].state
                neighbours = list(self.graph.adj[nodeIdx])
                neighs = 0
                if(len(neighbours) == 0):
                    nb.append(0)
                    continue
                for neighbourIdx in neighbours:
                    if(state == self.graph.nodes[neighbourIdx]['agent'].state): neighs+=1
                nb.append(neighs/len(neighbours))
        self.NbAgreeingFriends = nb
            
    def updateAvgNbAgreeingFriends(self, nodeIndex):
        #print(nodeIndex)
        neighbours =  list(self.graph.adj[nodeIndex].keys())
        if(len(neighbours) == 0):
            return self.avgNbAgreeingList[-1]
        nodeState = self.graph.nodes[nodeIndex]['agent'].state
        
        
        if(args["continuous"]):
            neighStates = [self.graph.nodes[n]['agent'].state for n in neighbours ]
            x = abs((mean(neighStates)+nodeState))/2
            self.NbAgreeingFriends[nodeIndex] = x
            for node in neighbours:
                nodeState = self.graph.nodes[node]['agent'].state
                neighneigh = list(self.graph.adj[node])
                neighStates = [self.graph.nodes[n]['agent'].state for n in neighneigh ]
                x = abs((mean(neighStates)+nodeState))/2
                self.NbAgreeingFriends[node] = x
        else:
            neighbours.append(nodeIndex)

            for n in neighbours:
                try:
                    neighneighs = list(self.graph.adj[n])
                    neighs = 0
                    nState = self.graph.nodes[n]['agent'].state
                    if(len(neighneighs) == 0):
                        self.NbAgreeingFriends[n] = (0)
                        continue
                    for neighbourIdx in neighneighs:
                        if(nState == self.graph.nodes[neighbourIdx]['agent'].state): neighs+=1
                    self.NbAgreeingFriends[n] = neighs/len(neighneighs)  
                except:
                    print("node: ", n)
                    print("neighs: ", neighneighs)      
            
        return mean(self.NbAgreeingFriends)


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

    def getAvgState(self):
        states = []
        for node in self.graph:
            states.append(self.graph.nodes[node]['agent'].state)
        statearray = np.array(states)
        avg = statearray.mean(axis=0)
        sd = statearray.std()
        return (avg, sd)

    def getFriendshipWeight(self):
        #weigth = random.uniform(0.1, 0.9)
        #global X
        weigth = self.X.rvs(1)
        return weigth[0]

    def getInitialState(self):
        global args
        if(args['continuous'] != True):
            state = states[random.randint(0,1)]
        else:   
        #    #state = random.uniform(-1, 1)s
            #global S
            state = self.S.rvs(1)[0]
        #state= getRandomExpo()
        
        return state
 
    def runSim(self, k, groupInteract=False, drawModel = False, countNeighbours = False, gifname=None, clusters=False):
        self.partition = community.best_partition(self.graph)
        #modularity = community.modularity(self.partition, self.graph)
        #print("modularity of ", args["type"], " is ", modularity)

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
    
        #create list of number of agreeing friends
        self.findNbAgreeingFriends()
        

        for i in range(k):
            #if(groupInteract): self.groupInteractB()
            #else:
            nodeIndex = self.interact()
            ratio = self.countCooperatorRatio()
            self.ratio.append(ratio)
            (state, sd) = self.getAvgState()
            self.states.append(state)
            self.statesds.append(sd)
            avgFriends = self.updateAvgNbAgreeingFriends(nodeIndex)
            self.avgNbAgreeingList.append(avgFriends)
            
            global args
            if(i ==500 and (args["newPoliticalClimate"] != args["politicalClimate"])):
                self.politicalClimate = args["newPoliticalClimate"]
                

            #self.politicalClimate += (ratio-0.5)*0.001 #change the political climate depending on the ratio of cooperators
            if(clusters):
                (s, sds, size) = findAvgStateInClusters(self, self.partition)
                self.clusterSD.append(sds)
                self.clusteravg.append(s)

            if(countNeighbours):
                (defectorDefectingNeighs,
                 cooperatorDefectingNeighs,
                 defectorDefectingNeighsSTD,
                 cooperatorDefectingNeighsSTD) = self.getAvgNumberOfDefectorNeigh()
                self.defectorDefectingNeighsList.append(defectorDefectingNeighs)
                self.cooperatorDefectingNeighsList.append(cooperatorDefectingNeighs)
                self.defectorDefectingNeighsSTDList.append(defectorDefectingNeighsSTD)
                self.cooperatorDefectingNeighsSTDList.append(cooperatorDefectingNeighsSTD)
            if(gifname != None and (i % 2 == 0)):
                draw_model(self, True, i, extraTitle = f'  avg state: {self.states[-1]:1.2f} ')
                filenames.append("plot" + str(i) +".png")
                
            #if(i % 10 == 0):
                #a = random.randint(0,n)
                #b = random.randint(0,n)
                #while(a==b):
                    #b = random.randint(0,n)
                    #weight = random.uniform(0.1, 0.9)
                    #model.graph.add_edge(a, b, weight = weight)
        if(gifname != None):
            images = []
            for filename in filenames:
                images.append(imageio.imread(filename))
            #0.08167
            imageio.mimsave("network" +gifname+ ".gif", images, duration=0.08167)
        
        (avgs, sds, sizes) = findAvgStateInClusters(self, self.partition)
        self.clusteravg.append(avgs)
    
        if(countNeighbours):
            drawDefectingNeighbours(self.defectorDefectingNeighsList,
                                    self.cooperatorDefectingNeighsList,
                                    self.defectorDefectingNeighsSTDList,
                                    self.cooperatorDefectingNeighsSTDList, 
                                    gifname)
        
        return self.ratio
    
    def populateModel(self, n, skew = 0):
        global args
        for n in range (n):
            agent1 = Agent(self.getInitialState(), args["selfWeight"])
            self.graph.nodes[n]['agent'] = agent1
        edges = self.graph.edges() 
        for e in edges: 
            weight=self.getFriendshipWeight()
            self.graph[e[0]][e[1]]['weight'] = weight
        
        if(skew != 0 and not args["continuous"] ): 
            num = round(abs(skew)*len(self.graph.nodes))
            indexes = random.sample(range(len(self.graph.nodes)), num)
            for i in indexes:
                self.graph.nodes[i]['agent'].state = states[1]
            #self.pos = nx.kamada_kawai_layout(self.graph)
        self.pos = nx.spring_layout(self.graph)

class GridModel(Model):
    def __init__(self, n, skew=0, **kwargs):
        super().__init__(**kwargs)
        global args
        for i in range(n):
            for j in range (n):
                weight = self.getFriendshipWeight()
                agent1 = Agent(self.getInitialState(), args["selfWeight"])
                self.graph.add_node(i*n+j, agent=agent1, pos=(i, j))
                self.pos.append((i, j))
                if(i!=0):
                    self.graph.add_edge(i*n+j, (i-1)*n+j, weight = weight)
                if(j!=0):
                    self.graph.add_edge(i*n+j, i*n+j-1, weight = weight)
        
        if(skew != 0 and not args["continuous"] ): 
                num = round(abs(skew)*len(self.graph.nodes))
                indexes = random.sample(range(len(self.graph.nodes)), num)
                for i in indexes:
                    self.graph.nodes[i]['agent'].state = states[1]

class ScaleFreeModel(Model):
    def __init__(self, n, m, skew= 0, **kwargs):
        super().__init__(**kwargs)
        
        self.graph = nx.barabasi_albert_graph(n, m)
        self.populateModel(n, skew)

class ClusteredPowerlawModel(Model):
    def __init__(self, n, m, skew = 0, **kwargs):
        super().__init__(**kwargs)
        
        self.graph = nx.powerlaw_cluster_graph(n, m, 0.5)
        self.populateModel(n, skew)
        
class RandomModel(Model):
    def __init__(self, n, m, skew= 0, **kwargs):
        #m is avg degree/2
        super().__init__(**kwargs)
        p = 2*m/(n-1)
        
        self.graph =nx.erdos_renyi_graph(n, p)
        self.populateModel(n, skew)
    
class KarateModel(Model):
    def __init__(self, **kwargs):
        #m is avg degree/2
        super().__init__(**kwargs)
        
        self.graph =nx.karate_club_graph()
        for n in range (len(self.graph.nodes)):
                agent1 = Agent(self.getInitialState())
                self.graph.nodes[n]['agent'] = agent1
        edges = self.graph.edges() 
        for e in edges: 
            weight=self.getFriendshipWeight()
            
            self.graph[e[0]][e[1]]['weight'] = weight 
        self.pos = nx.spring_layout(self.graph)

import dill

def saveModels(models, filename):
    with open(filename, 'wb') as f:
        dill.dump(models, f)

def loadModels(filename):
    with open(filename, 'rb') as f:
        models = dill.load(f)
    return models

def findClusters(model):
    partition = community.best_partition(model.graph)
    return partition

    
def findAvgStateInClusters(model, part):
    states = [[] for i in range(len(set(part.values())))]
   
    for n, v in part.items():
        states[v].append(model.graph.node[n]['agent'].state)
    clusters = []
    sd = []
    clsize = []
    for c in range(len(states)):
        clusters.append(mean(states[c]))
        clsize.append(len(states[c]))
        if(len(states[c])>1):
            sd.append(stdev(states[c]))
        else:
            sd.append(0) 
    return (clusters, sd, clsize)

def findAvgSDinClusters(model, part):
    states = [[] for i in range(len(set(part.values())))]
    for n, v in part.items():
        states[v].append(model.graph.node[n]['agent'].state)
    
    sd = []
    for c in range(len(states)):
        if(len(states[c])>1):
            sd.append(stdev(states[c]))
        else:
            sd.append(0)
    return sd

def drawClusteredModel(model):
    partition = findClusters(model)
    
    for k, v in partition.items():
        model.graph.node[k]["louvain-val"] = v
    degrees = nx.degree(model.graph)

    #colors = [mypalette[G.node[node]["louvain-val"] %9 ]  for node in G.nodes()]
#    edge_col = [mypalette[model.graph.node[node]["louvain-val"]+1 % 8 ]  for node in model.graph.nodes()]
    edge_col = []
    for node in model.graph.nodes():
        edge_col.append(mypalette[model.graph.node[node]["louvain-val"] % 9 ])
    
    plt.subplot(1, 2, 2, title="Cluster membership")
    nx.draw(model.graph, model.pos, node_size=[d[1] * 30 for d in degrees], node_color =edge_col)
    (clusters, sd, clsize) = findAvgStateInClusters(model, part= partition)
    text = [f'x={clusters[c]:5.2f} sd={sd[c]:5.2f} n={clsize[c]}' for c in range(len(clusters))]
    #print(text)
    ax = plt.gca()
    handles = [mpatches.Patch(color=mypalette[c], label=text[c]) for c in range(len(text))]
    ax.legend(handles=handles)
    #plt.title("Snapshot of network with states and clusters")
    draw_model(model)#, outline=edge_col, partition = partition)
    
    
from IPython.display import Image


#-------- drawing functions ---------
import matplotlib.patches as mpatches

def draw_model(model, save=False, filenumber = None, outline=None, partition=None, extraTitle=""):
    
    #plt.figure(figsize=(16,16))
    plt.subplot(1, 2, 1, title="State of the nodes")
    color_map = []
    intensities = []
    #pos = []
    for node in model.graph:
        #pos.append(model.graph.nodes[node]['pos'])
        if model.graph.nodes[node]['agent'].state > 0:
            color_map.append((3/255,164/255,94/255, model.graph.nodes[node]['agent'].state))
            intensities.append(model.graph.nodes[node]['agent'].state)
            #color_map.append('#03a45e')
            #else: color_map.append('#f7796d')
            
        else: 
            color_map.append((247/255,121/255,109/255, -1*model.graph.nodes[node]['agent'].state ))
            intensities.append(model.graph.nodes[node]['agent'].state)
    degrees = nx.degree(model.graph)
    #plt.subplot(121)
    nx.draw(model.graph, model.pos, node_size=[d[1] * 30 for d in degrees], linewidths=2, node_color =intensities, cmap=plt.cm.RdYlGn,  vmin=-1, vmax=1 )
    #plt.colorbar(mcp)
    #plt.show()
    
    if(outline !=None):
        #mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "magenta","cyan", "cyan", "cyan", "cyan"]
        ax = plt.gca()
        ax.collections[0].set_edgecolor(outline)
        (clusters, sd, clsize) = findAvgStateInClusters(model, part= partition)
        text = [f'x={clusters[c]:5.2f} sd={sd[c]:5.2f} n={clsize[c]}' for c in range(len(clusters))]
        #print(text)
        handles = [mpatches.Patch(color=mypalette[c], label=text[c]) for c in range(len(text))]
        ax.legend(handles=handles)
        plt.title("Snapshot of network with states and clusters")


    if(save):
        plt.title(str(filenumber)+extraTitle)
        plt.savefig("plot" + str(filenumber) +".png", bbox_inches="tight")
        plt.close('all')

def radialDist(model, depth, isBefore):
    DefectorValues = [[0 for i in range(depth)] for j in range(len(model.graph))]
    CooperatorValues = [[0 for i in range(depth)] for j in range(len(model.graph))]
    
    for nodeIdx in model.graph:
        neighbours = list(model.graph.adj[nodeIdx])
        isCooperator = model.graph.nodes[nodeIdx]['agent'].state > 0
        parent = [nodeIdx]
        for d in range(depth):
            nextLevelNeighs = set([])
            for n in neighbours:
                nextLevelNeighs.update(list(model.graph.adj[n]))
                if(model.graph.nodes[n]['agent'].state > 0 and isCooperator):
                    CooperatorValues[nodeIdx][d] += 1
                elif(model.graph.nodes[n]['agent'].state <= 0 and not isCooperator): 
                    DefectorValues[nodeIdx][d] += 1
            CooperatorValues[nodeIdx][d] = CooperatorValues[nodeIdx][d]/len(neighbours)
            DefectorValues[nodeIdx][d] = DefectorValues[nodeIdx][d]/len(neighbours)
            
            #make sure the parent level isn't checked again
            for n in parent:
                nextLevelNeighs.discard(n) 
            parent = neighbours
            neighbours = nextLevelNeighs
     
    cooperatorRatio = model.countCooperatorRatio()
    
    cooperatorRes = []
    defectorRes = []
    for col in range(depth):
        coopSumRatios = 0
        defectSumRatios = 0
        for row in range(len(CooperatorValues)):
            coopSumRatios += CooperatorValues[row][col]
            defectSumRatios += DefectorValues[row][col]
        cooperatorRes.append(np.array(coopSumRatios)/(len(model.graph)*cooperatorRatio*cooperatorRatio))
        defectorRes.append(np.array(defectSumRatios)/(len(model.graph)*(1-cooperatorRatio)*(1-cooperatorRatio)))

    if isBefore:
        intensity = 0.5
    else:
        intensity = 1
    plt.xlabel("Distance from the nodes")
    plt.ylabel("Normalised ratio of agreein neighbours")
    plt.title("Distance distribution function")
    plt.ylim((0, 2.5))
    plt.plot(range(1, len(cooperatorRes)+1), cooperatorRes, color=((23/255, 104/255, 37/255, intensity)))     
    plt.plot(range(1, len(cooperatorRes)+1), defectorRes, color=((109/255, 10/255, 10/255, intensity))) 

def avgRadialDist(models, depth, isBefore):
    DefectorList = []
    CooperatorList = []
    
    for model in models :
        DefectorValues = [[0 for i in range(depth)] for j in range(len(model.graph))]
        CooperatorValues = [[0 for i in range(depth)] for j in range(len(model.graph))]

        for nodeIdx in model.graph:
            neighbours = list(model.graph.adj[nodeIdx])
            isCooperator = model.graph.nodes[nodeIdx]['agent'].state > 0
            parent = [nodeIdx]
            for d in range(depth):
                nextLevelNeighs = set([])
                for n in neighbours:
                    nextLevelNeighs.update(list(model.graph.adj[n]))
                    if(model.graph.nodes[n]['agent'].state > 0 and isCooperator):
                        CooperatorValues[nodeIdx][d] += 1
                    elif(model.graph.nodes[n]['agent'].state <= 0 and not isCooperator): 
                        DefectorValues[nodeIdx][d] += 1
                if(len(neighbours) == 0):
                    break
                CooperatorValues[nodeIdx][d] = CooperatorValues[nodeIdx][d]/len(neighbours)
                DefectorValues[nodeIdx][d] = DefectorValues[nodeIdx][d]/len(neighbours)

                #make sure the parent level isn't checked again
                for n in parent:
                    nextLevelNeighs.discard(n) 
                parent = neighbours
                neighbours = nextLevelNeighs

        cooperatorRatio = model.countCooperatorRatio()

        cooperatorRes = []
        defectorRes = []
        for col in range(depth):
            coopSumRatios = 0
            defectSumRatios = 0
            for row in range(len(CooperatorValues)):
                coopSumRatios += CooperatorValues[row][col]
                defectSumRatios += DefectorValues[row][col]
            if(cooperatorRatio == 0):
                cooperatorRes.append(1)
            else:
                cooperatorRes.append(np.array(coopSumRatios)/(len(model.graph)*cooperatorRatio*cooperatorRatio))
            if(cooperatorRatio == 1):
                defectorRes.append(1)
            else:
                defectorRes.append(np.array(defectSumRatios)/(len(model.graph)*(1-cooperatorRatio)*(1-cooperatorRatio)))
        DefectorList.append( defectorRes)
        CooperatorList.append( cooperatorRes)
    data = np.array(DefectorList)
    avgDefector = np.average(data, axis=0)
    data = np.array(CooperatorList)
    avgCooperator = np.average(data, axis=0)
            
    if isBefore:
        intensity = 0.5
    else:
        intensity = 1
    plt.xlabel("Distance from the nodes")
    plt.ylabel("Normalised ratio of agreein neighbours")
    plt.title("Distance distribution function")
    plt.ylim((0, 2.5))
    plt.xlim((0.5, 5.5))
    plt.plot(range(1, len(avgDefector)+1), avgCooperator, color=((23/255, 104/255, 37/255, intensity)))     
    plt.plot(range(1, len(avgDefector)+1), avgDefector, color=((109/255, 10/255, 10/255, intensity))) 
    plt.show()

def drawAvgState(models, avg =False, pltNr=1, title="", clusterSD = False):
    plt.xlabel("timesteps")
    plt.ylabel("Average state")
    #mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "grey","grey"]
    plt.subplot(1, 2, 1, title="Avg state + SD")
    
    if(not avg):
        plt.ylim((-1, 1))
        for i in range(len(models)):
            plt.plot(models[i].states)
            plt.plot(models[i].statesds, alpha=0.5)
            if(clusterSD):
                sds = np.array(models[i].clusterSD)
                avgsd = sds.mean(axis=1)
                plt.plot(avgsd, linestyle=":")
    else:
        states = []
        sds = []
        plt.ylim((-1, 1))
        for i in range(len(models)):
            states.append(models[i].states)
            sds.append(models[i].statesds)
        array = np.array(states)
        avg = array.mean(axis=0)
        std = np.array(sds).mean(axis=0)
        plt.plot(avg, color=mypalette[pltNr-1], label=title)
        plt.plot(std, color=col.to_rgba(mypalette[pltNr-1], 0.5))
        #plt.plot(avg+std, color=col.to_rgba(mypalette[pltNr-1], 0.5))
        #text = ["rand cont", "cl cont", "rand disc", "cl disc"]
        text =["grid cont", "grid disc"]
        #handles = [mpatches.Patch(color=mypalette[c+6], label=text[c]) for c in range(len(text))]
        #plt.legend(handles=handles)
        if(clusterSD):
            avgSds = []
            for mod in models:
                #print(mod.clusterSD)
                array = np.array(mod.clusterSD)
                avgSd = array.mean(axis=1)
                avgSds.append(avgSd)
            array = np.array(avgSds)
            avgAvgSd = array.mean(axis=0)
            plt.plot(avgAvgSd, color=mypalette[pltNr-1], linestyle=":")

        #plt.subplot(1, 2, 2)
        #plt.ylim((0, 1))
        #plt.plot(std, color=mypalette[pltNr-1])

def drawCrossSection(models, pltNr = 1):
    values = []
    #mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "grey","grey"]
    for model in models:
        values.append(model.states[-1])
    plt.subplot(1, 2, 2, title="Density Plot of state for simulations")
    ax = plt.gca()
    #ax.set_xscale('log')
    plt.xlim((0, 5))
    plt.ylim((-1, 1))
    #plt.title('Density Plot of state for simulations')
    #plt.xlabel('avg state of cooperators after all time steps')
    plt.xlabel('Density')
    try:
        sns.distplot(values, hist=False, kde=True, color = mypalette[pltNr-1], vertical=True)
    except:
        sns.distplot(values, hist=True, kde=False, color = mypalette[pltNr-1], vertical=True)

    #plt.show()

def drawClustersizes(models, pltNr = 1):
    #mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "grey","grey"]
    sizes = []
    for model in models:
        part = findClusters(model)
        (avg, sd, size) = findAvgStateInClusters(model, part)
        for s in size:
            sizes.append(s)
    plt.subplot(1, 3, 3, title="Density Plot of clustersize simulations")
    plt.xlabel("Clustersize")
    sns.distplot(sizes, hist=True, kde=True, color = mypalette[pltNr-1])

def drawConvergence(variables, modelsList, pltNr = 1):
    #mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "grey","grey"]

    endState = []
    for models in modelsList:
        values = []
        for model in models:
            values.append(model.states[-1])
        endState.append(mean(values))
    plt.subplot(1,2,2)
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.scatter(variables, endState, color=mypalette[pltNr-1])

def drawClusterState(models, pltNr = 1, step=-1, subplot=1):
    if(step < 0):
        plt.subplot(1, 3, 3, title="Avg state in clusters")
        states = []
        for i in range(len(models)):
            #print(models[i].clusteravg[0])
            for c in models[i].clusteravg[-1]:
                states.append(c)
    else:
        plt.subplot(1, 3, subplot, title="Avg state at "+ str(step))
        states = []
        for i in range(len(models)):
            #print(models[i].clusteravg[step])
            for c in models[i].clusteravg[step]:
                states.append(c)
    ax = plt.gca()
    #ax.set_xscale('log')
    plt.xlim((0, 5))
    plt.ylim((-1, 1))
    #plt.title('Density Plot of state for simulations')
    #plt.xlabel('avg state of cooperators after all time steps')
    plt.xlabel('Density')
    #print(states)
    try:
        sns.distplot(states, hist=True, kde=True, color = mypalette[pltNr-1], vertical=True)
    except:
        sns.distplot(states, hist=True, kde=False, color = mypalette[pltNr-1], vertical=True)

def drawAvgNumberOfAgreeingFriends(models, pltNr = 1):
    avgNbAgreeingFriends = [model.avgNbAgreeingList for model in models]
    avgAvg = np.array(avgNbAgreeingFriends).mean(axis=0)
    plt.title("avg agreement of friends")
    plt.ylim((0, 1))
    plt.plot(avgAvg, color=mypalette[pltNr-1])

