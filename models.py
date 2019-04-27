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
politicalClimate=0.20 
selfWeight = 0.6
d = 2 #degree
s = 100
k=3000 #10^6
continuous = True

#X = get_truncated_normal(0.5, 0.25, 0, 1)
#S = get_truncated_normal(0, 0.25, -1, 1)


args = {"defectorUtility" : defectorUtility, 
        "politicalClimate" : politicalClimate, 
        "selfWeight": selfWeight, "d":d, 
        "s": s, "k" : k, "continuous" : continuous, "type" : "cl"}

def simulate(i, newArgs):
    setArgs(newArgs)
    global args
    
    X = get_truncated_normal(0.5, 0.2, 0, 1)
    S = get_truncated_normal(0, 0.25, -1, 1)
    if(args["type"] == "cl"):
        model =ClusteredPowerlawModel(144, args["d"], X=X, S=S)
    elif(args["type"] == "sf"):
        model = ScaleFreeModel(144, args["d"],  X=X, S=S)
    elif(args["type"] == "grid"):
        model = GridModel(12, X=X, S=S)
    elif(args["type"] == "rand"):
        model = RandomModel(144, args["d"],  X=X, S=S)
    else:
        model = RandomModel(144, args["d"],  X=X, S=S)
    
    res = model.runSim(k)
    return model


#Helper
def setArgs(newArgs):
    global args
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
    def __init__(self, state):
        self.state = state
        self.interactionsReceived = 0
        self.interactionsGiven = 0
    
    def consider(self, neighbour, neighboursWeight):
        self.interactionsReceived +=1
        neighbour.addInteractionGiven()
        weight = self.state*selfWeight + politicalClimate + defectorUtility + neighboursWeight*neighbour.state #+ random.uniform(-0.25, 0.25)
        #weight =  politicalClimate + defectorUtility + neighboursWeight*neighbour.state #+ random.uniform(-0.25, 0.25)
        #print("neighbours weight: ", neighboursWeight, " neighbours state: ", neighbour.state, " weight: ", weight)
        if(continuous):
            #self.state = weight
            p1 = (0.25+weight)*2
            if(p1 <0): p1 = 0
            if(p1 > 1): p1=1
         #   print("Self.state: ",self.state, " p1: ", p1)
            delta = 1/2*(-self.state+1)*(p1) - (1/2*(self.state+1))*(1-p1)
          #  print("delta: ", delta)
            self.state += 2*delta
            #Truncate values    
            if(self.state > 1):
                  self.state = states[0]
            elif(self.state <-1):
                self.state = states[1]
                 
        else:
            if(weight + random.uniform(-0.25, 0.25)  > 0):
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
    
    
    def setState(self, newState):
        if(newState >= states[1] and newState <= states[0]):
            self.state = newState
        else:
            print("Error state outside state range: ", newState)
        

class Model:
    def __init__(self, X = None, S=None):
        self.graph = nx.Graph()
        self.ratio = []
        self.states = []
        self.defectorDefectingNeighsList = []
        self.cooperatorDefectingNeighsList = []
        self.defectorDefectingNeighsSTDList = []
        self.cooperatorDefectingNeighsSTDList =[]
        self.pos = []
        self.X = X
        self.S = S
    
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
        state = 0
        for node in self.graph:
            state += self.graph.nodes[node]['agent'].state
        return state/len(self.graph)

    def getFriendshipWeight(self):
        #weigth = random.uniform(0.1, 0.9)
        #global X
        weigth = self.X.rvs(1)
        return weigth[0]

    def getInitialState(self):
        #state = states[random.randint(0,1)]
        #state = random.uniform(-1, 1)
        #state= getRandomExpo()
        #global S
        #state = self.S.rvs(1)
        return states[0]
 
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
            if(groupInteract): self.groupInteractB()
            else:
                self.interact()
            ratio = self.countCooperatorRatio()
            self.ratio.append(ratio)
            state = self.getAvgState()
            self.states.append(state)
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
            if(gifname != None and (i % 10 == 0)):
                draw_model(self, True, i)
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
       
    
        if(countNeighbours):
            drawDefectingNeighbours(self.defectorDefectingNeighsList,
                                    self.cooperatorDefectingNeighsList,
                                    self.defectorDefectingNeighsSTDList,
                                    self.cooperatorDefectingNeighsSTDList, 
                                    gifname)
        
        return self.ratio

class GridModel(Model):
    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        for i in range(n):
            for j in range (n):
                
                weight = self.getFriendshipWeight()
                agent1 = Agent(self.getInitialState())
                self.graph.add_node(i*n+j, agent=agent1, pos=(i, j))
                self.pos.append((i, j))
                if(i!=0):
                    self.graph.add_edge(i*n+j, (i-1)*n+j, weight = weight)
                if(j!=0):
                    self.graph.add_edge(i*n+j, i*n+j-1, weight = weight)
    

class ScaleFreeModel(Model):
    def __init__(self, n, m, **kwargs):
        super().__init__(**kwargs)
        
        self.graph = nx.barabasi_albert_graph(n, m)
        for n in range (n):
                agent1 = Agent(self.getInitialState())
                self.graph.nodes[n]['agent'] = agent1
        edges = self.graph.edges() 
        for e in edges: 
            
            weight = self.getFriendshipWeight()
            self.graph[e[0]][e[1]]['weight'] = weight 
        self.pos = nx.kamada_kawai_layout(self.graph)

class ClusteredPowerlawModel(Model):
    def __init__(self, n, m, **kwargs):
        super().__init__(**kwargs)
        
        self.graph = nx.powerlaw_cluster_graph(n, m, 0.5)
        for n in range (n):
                agent1 = Agent(self.getInitialState())
                self.graph.nodes[n]['agent'] = agent1
        edges = self.graph.edges() 
        for e in edges: 
            weight=self.getFriendshipWeight()
            self.graph[e[0]][e[1]]['weight'] = weight 
        #self.pos = nx.kamada_kawai_layout(self.graph)
        self.pos = nx.spring_layout(self.graph)
        
class RandomModel(Model):
    def __init__(self, n, m, **kwargs):
        #m is avg degree/2
        super().__init__(**kwargs)
        p = 2*m/(n-1)
        
        self.graph =nx.erdos_renyi_graph(n, p)
        for n in range (n):
                agent1 = Agent(self.getInitialState())
                self.graph.nodes[n]['agent'] = agent1
        edges = self.graph.edges() 
        for e in edges: 
            weight=self.getFriendshipWeight()
            
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




def findClusters(model):
    partition = community.best_partition(model.graph)
    return partition
    
def findAvgStateInClusters(model, part):
    #part = findClusters(model)
    print("communities: ", len(set(part.values())))
    print(part)
    states = [[] for i in range(len(set(part.values())))]
   
    for n, v in part.items():
        states[v].append(model.graph.node[n]['agent'].state)
    print(states)
    clusters = []
    sd = []
    clsize = []
    for c in range(len(states)):
        clusters.append(mean(states[c]))
        sd.append(stdev(states[c]))
        clsize.append(len(states[c]))
    return (clusters, sd, clsize)

def drawClusteredModel(model):
    partition = findClusters(model)
    mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "magenta","cyan"]
    for k, v in partition.items():
        model.graph.node[k]["louvain-val"] = v
    
    #colors = [mypalette[G.node[node]["louvain-val"] %9 ]  for node in G.nodes()]
#    edge_col = [mypalette[model.graph.node[node]["louvain-val"]+1 % 8 ]  for node in model.graph.nodes()]
    edge_col = []
    for node in model.graph.nodes():
        edge_col.append(mypalette[model.graph.node[node]["louvain-val"] % 9 ])
    
    draw_model(model, outline=edge_col, partition = partition)
    
    
from IPython.display import Image


#-------- drawing functions ---------
import matplotlib.patches as mpatches

def draw_model(model, save=False, filenumber = None, outline=None, partition=None):
    
    #plt.figure(figsize=(16,16))

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
        mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "magenta","cyan", "cyan", "cyan", "cyan"]
        ax = plt.gca()
        ax.collections[0].set_edgecolor(outline)
        (clusters, sd, clsize) = findAvgStateInClusters(model, part= partition)
        text = [f'x={clusters[c]:5.2f} sd={sd[c]:5.2f} n={clsize[c]}' for c in range(len(clusters))]
        #print(text)
        handles = [mpatches.Patch(color=mypalette[c], label=text[c]) for c in range(len(text))]
        ax.legend(handles=handles)
        plt.title("Snapshot of network with states and clusters")


    if(save):
        plt.title(filenumber)
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

def drawAvgState(models, avg =False, pltNr=1, title=""):
    plt.xlabel("timesteps")
    plt.ylabel("fraction of cooperators")
    mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "grey","grey"]
    plt.subplot(1, 2, 1, title="Avg ratio of cooperators + SD")
    if(not avg):
        plt.ylim((0, 1))
        for i in range(len(models)):
            plt.plot(models[i].ratio)
    else:
        states = []
        plt.ylim((-1, 1))
        for i in range(len(models)):
            states.append(models[i].states)
        array = np.array(states)
        avg = array.mean(axis=0)
        std = array.std(axis=0)
        plt.plot(avg, color=mypalette[pltNr-1], label=title)
        plt.plot(avg-std, color=col.to_rgba(mypalette[pltNr-1], 0.5))
        plt.plot(avg+std, color=col.to_rgba(mypalette[pltNr-1], 0.5))
    

def drawCrossSection(models, pltNr = 1):
    values = []
    mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "grey","grey"]
    for model in models:
        values.append(model.states[-1])
    plt.subplot(1, 2, 2, title="Density Plot of state for simulations")
    plt.xlim((0, 2))
    plt.ylim((-1, 1))
    #plt.title('Density Plot of state for simulations')
    #plt.xlabel('avg state of cooperators after all time steps')
    plt.xlabel('Density')
    sns.distplot(values, hist=False, kde=True, 
    color = mypalette[pltNr-1], vertical=True)
    #plt.show()