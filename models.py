import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as col
from copy import deepcopy
import seaborn as sns
from statistics import stdev, mean
import imageio
import networkx as nx
from scipy.stats import truncnorm
import os
from community import community_louvain
#import community
from operator import itemgetter
import heapq
from IPython.display import Image
import matplotlib.patches as mpatches
import dill

#random.seed(1574705741) # linuxtime at some point
#np.random.seed(1574705741)

#Helper functions

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def setArgs(newArgs):
    global args
    for arg, value in newArgs.items():
        args[arg] = value

def getRandomExpo():
    x = np.random.exponential(scale=0.6667)-1
    if(x>1): return 1
    elif (x< -1): return -1
    return x

#Constants and Variables

STATES = [1, -1] #1 being cooperating, -1 being defecting
defectorUtility = 0.0
politicalClimate= 0.255        # for "equilibrium" use 0.2475
newPoliticalClimate =politicalClimate 
stubbornness = 0.6
degree = 4 
timesteps= 5000  #timesteps
continuous = True
skew = -0.15
initSD = 0.20
mypalette = ["blue","red","green", "orange", "magenta","cyan","violet", "grey", "yellow"]
randomness = 0.25
gridtype = 'grid'

args = {"defectorUtility" : defectorUtility, 
        "politicalClimate" : politicalClimate, 
        "stubbornness": stubbornness, "degree":degree, "timesteps" : timesteps, "continuous" : continuous, "type" : gridtype, "skew": skew, "initSD": initSD, "newPoliticalClimate": newPoliticalClimate, "randomness" : randomness}

def getargs():
    return args

def simulate(i, newArgs):
    setArgs(newArgs)
    global args
    friendshipWeightGenerator = get_truncated_normal(0.5, 0.15, 0, 1) 
    initialStateGenerator = get_truncated_normal(args["skew"], args["initSD"], -1, 1)
    ind = None

    if(args["type"] == "cl"):
        model =ClusteredPowerlawModel(144, args["degree"], skew=args["skew"], friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
    elif(args["type"] == "sf"):
        model = ScaleFreeModel(144, args["degree"], skew=args["skew"], friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
    elif(args["type"] == "grid"):
        ind = [10,64, 82]
        if(args["degree"]>2): doubleDegree = True
        else:doubleDegree = False
        model = GridModel(12, skew=args["skew"], doubleDegree =doubleDegree, friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
    elif(args["type"] == "rand"):
        model = RandomModel(144, args["degree"], skew=args["skew"], friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
    else:
        model = RandomModel(144, args["degree"],  friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)

    model.addInfluencers(newArgs["influencers"], index=ind, hub=False, allSame=False)
    res = model.runSim(args["timesteps"], clusters=True, drawModel=True, gifname=None) ## gifname goes here!
    return model



class Agent:
    def __init__(self, state, stubbornness):
        self.state = state
        self.interactionsReceived = 0
        self.interactionsGiven = 0
        self.stubbornness = stubbornness

    def consider(self, neighbour, neighboursWeight, politicalClimate):
        self.interactionsReceived +=1
        neighbour.addInteractionGiven()
        if(self.stubbornness >= 1): return
        global args
        weight = self.state*self.stubbornness + politicalClimate + args["defectorUtility"] + neighboursWeight*neighbour.state #+ random.uniform(-0.25, 0.25)

        #if(args["continuous"]):
        #    p1 = (randomness+weight)*(1/(2*randomness))

        #    if(p1 <0): p1 = 0
        #    if(p1 > 1): p1=1
        #
        #    delta = (1/2)*(-self.state+1)*(p1) - ((1/2)*(self.state+1))*(1-p1)
        #    increment = 2*delta*abs(self.state-neighbour.state)

        #    self.state += increment
        #    if(self.state > 1):
        #          self.state = STATES[0]
        #    elif(self.state <-1):
        #        self.state = STATES[1]       
        #else:
        #    if(weight + random.uniform(-randomness, randomness)  > 0):
        #        self.state = STATES[0]
        #    else:
        #        self.state = STATES[1]  

        if(args["continuous"]):

            p1 = 0
            sample = random.uniform(-randomness,randomness)
            check = (weight + sample)

            if(check < randomness): 
                p1 = 0

            elif(check > randomness): 
                p1 = 1

            else: 
                p1 = 1/(2*randomness)*(randomness + sample)

            p2 = 1 - p1

            delta = abs(self.state - neighbour.state)*(p1*(1-self.state) - p2*(1+self.state))

            self.state += delta

            if(self.state > 1):
                self.state = STATES[0]
            elif(self.state <-1):
                self.state = STATES[1]       
        else:
            if(weight + random.uniform(-randomness, randomness)  > 0):
                self.state = STATES[0]
            else:
                self.state = STATES[1]

    def addInteractionGiven(self):
        self.interactionsGiven +=1

    def setState(self, newState):
        if(newState >= STATES[1] and newState <= STATES[0]):
            self.state = newState
        else:
            print("Error state outside state range: ", newState)


class Model:
    def __init__(self, friendshipWeightGenerator = None, initialStateGenerator=None):
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
        self.friendshipWeightGenerator = friendshipWeightGenerator
        self.initialStateGenerator = initialStateGenerator
        self.clusteravg = []
        self.clusterSD = []
        self.NbAgreeingFriends = []
        self.avgNbAgreeingList = []
        self.partition = None
        #random.seed(1574771191) # this is used to evaluate pol clim change
        #np.random.seed(1574771191)

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


    def findNbAgreeingFriends(self, nodeIdx = None):
        global args
        nbs = []

        if(args["continuous"]):
            for nodeIdx in self.graph.nodes:
                state = self.graph.nodes[nodeIdx]['agent'].state
                neighbours = list(self.graph.adj[nodeIdx])
                neighStates = [self.graph.nodes[n]['agent'].state for n in neighbours ]
                if(len(neighbours) == 0):
                    nbs.append(0)
                    continue
                x = 1-abs((mean(neighStates)-state))/2
                nbs.append(x)
        else:
            for nodeIdx in self.graph.nodes:
                state = self.graph.nodes[nodeIdx]['agent'].state
                neighbours = list(self.graph.adj[nodeIdx])
                neighs = 0
                if(len(neighbours) == 0):
                    nbs.append(0)
                    continue
                for neighbourIdx in neighbours:
                    if(state == self.graph.nodes[neighbourIdx]['agent'].state): neighs+=1
                nbs.append(neighs/len(neighbours))
        self.NbAgreeingFriends= nbs
        return nbs

    def updateAvgNbAgreeingFriends(self, nodeIndex):
        #print(nodeIndex)
        neighbours =  list(self.graph.adj[nodeIndex].keys())
        if(len(neighbours) == 0):
            return self.avgNbAgreeingList[-1]
        nodeState = self.graph.nodes[nodeIndex]['agent'].state


        if(args["continuous"]):
            neighStates = [self.graph.nodes[n]['agent'].state for n in neighbours ]
            x = 1-abs((mean(neighStates)-nodeState))/2
            self.NbAgreeingFriends[nodeIndex] = x
            for node in neighbours:
                nodeState = self.graph.nodes[node]['agent'].state
                neighneigh = list(self.graph.adj[node])
                neighStates = [self.graph.nodes[n]['agent'].state for n in neighneigh ]
                x = 1-abs((mean(neighStates)-nodeState))/2
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

    def addInfluencers(self, number = 0, index = None, hub = True, allSame =False):
        if(number == 0):
            return
        if(index == None):
            degrees = nx.degree(self.graph)
            if(hub):
                largest = heapq.nlargest(number, degrees, key=itemgetter(1))
                index = [t[0] for t in largest]


            else:
                index = [p[0]  for p in degrees if p[1] == degree*2]
                if(len(index) == 0 or len(index) < number ):
                    extra = [p[0]  for p in degrees if p[1] == degree*2-1]
                    index = index + extra
            #print(largest)
        for i in range(number):
            if(allSame):
                self.graph.node[index[i]]['agent'].setState(STATES[0])
            else:
                self.graph.node[index[i]]['agent'].setState(STATES[i % 2])
            self.graph.node[index[i]]['agent'].stubbornness = 1



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
        weigth = self.friendshipWeightGenerator.rvs(1)
        return weigth[0]

    def getInitialState(self):
        global args
        if(args['continuous'] != True): 
            state = STATES[random.randint(0,1)]
        else:   
            #state = random.uniform(-1, 1)
            state = self.initialStateGenerator.rvs(1)[0]
        #state= getRandomExpo()

        return state

    def runSim(self, timesteps, groupInteract=False, drawModel = False, countNeighbours = False, gifname=None, clusters=False):
        if(self.partition ==None):
            self.partition = community_louvain.best_partition(self.graph)
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
        self.avgNbAgreeingList.append(mean(self.NbAgreeingFriends))


        for i in range(timesteps):

            #print("step: ", i)
            nodeIndex = self.interact()
            ratio = self.countCooperatorRatio()
            self.ratio.append(ratio)
            (state, sd) = self.getAvgState()
            self.states.append(state)
            self.statesds.append(sd)
            avgFriends = self.updateAvgNbAgreeingFriends(nodeIndex)
            #avgFriends = self.findNbAgreeingFriends(nodeIndex)

            self.avgNbAgreeingList.append(avgFriends)

            global args
            if(i == 500 and (args["newPoliticalClimate"] != args["politicalClimate"])):
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
            if(gifname != None and (i % 500 == 0)):
                draw_model(self, True, i, extraTitle = f'  avg state: {self.states[-1]:1.2f} agreement: {self.avgNbAgreeingList[-1]:1.2f}')
                filenames.append("plot" + str(i) +".png")

        if(gifname != None):
            images = []
            for filename in filenames:
                images.append(imageio.imread(filename))
            #0.08167
            imageio.mimsave("network" +gifname+ ".gif", images, duration=0.08167)

        (avgs, sds, sizes) = findAvgStateInClusters(self, self.partition)
        self.clusteravg.append(avgs)

        return self.ratio

    def populateModel(self, n, skew = 0):
        global args
        for n in range (n):
            agent1 = Agent(self.getInitialState(), args["stubbornness"])
            self.graph.node[n]['agent'] = agent1
        edges = self.graph.edges() 
        for e in edges: 
            weight=self.getFriendshipWeight()
            self.graph[e[0]][e[1]]['weight'] = weight

        if(skew != 0 and not args["continuous"] ): 
            num = round(abs(skew)*len(self.graph.nodes))
            indexes = random.sample(range(len(self.graph.nodes)), num)
            for i in indexes:
                self.graph.node[i]['agent'].state = STATES[1]
            #self.pos = nx.kamada_kawai_layout(self.graph)
        #self.pos = forceatlas2.forceatlas2_networkx_layout(self.graph)
        #self.pos = force_atlas2_layout(self.graph)
        self.pos = nx.spring_layout(self.graph)

class GridModel(Model):
    def __init__(self, n, skew=0, doubleDegree=False, **kwargs):
        super().__init__(**kwargs)
        global args
        for i in range(n):
            for j in range (n):
                agent1 = Agent(self.getInitialState(), args["stubbornness"])
                self.graph.add_node(i*n+j, agent=agent1, pos=(i, j))
                self.pos.append((i, j))
                if(i!=0):
                    weight = self.getFriendshipWeight()
                    self.graph.add_edge(i*n+j, (i-1)*n+j, weight = weight)
                if(i==n-1):
                    weight = self.getFriendshipWeight()
                    self.graph.add_edge(i*n+j, j, weight = weight)
                if(j!=0):
                    weight = self.getFriendshipWeight()
                    self.graph.add_edge(i*n+j, i*n+j-1, weight = weight)
                if(j==n-1):
                    weight = self.getFriendshipWeight()
                    self.graph.add_edge(i*n+j, i*n, weight = weight)
        if(doubleDegree):
            for i in range(n):
                for j in range(n):
                    if(i!=0 and j!=0 ):
                        weight = self.getFriendshipWeight()
                        self.graph.add_edge(i*n+j, (i-1)*n+j-1, weight = weight)
                    if(i!=0 and j!=(n-1)):
                        weight = self.getFriendshipWeight()
                        self.graph.add_edge(i*n+j, (i-1)*n+j+1, weight = weight)
                    """
                    if( i != n-1 and j!= n-1):
                        weight = self.getFriendshipWeight()
                        self.graph.add_edge(i*n+j, (i+1)*n+j+1, weight = weight)
                    if(j != 0 and i != n-i):
                        weight = self.getFriendshipWeight()
                        self.graph.add_edge(i*n+j, (i+1)*n+j-1, weight = weight)"""

                    if(j == n-1):
                        if(i == n-1):
                            weight = self.getFriendshipWeight()
                            self.graph.add_edge(i*n+j, 0, weight = weight)
                        else:
                            weight = self.getFriendshipWeight()
                            self.graph.add_edge(i*n+j, (i+1)*n, weight = weight)
                        if(i == 0):
                            weight = self.getFriendshipWeight()
                            self.graph.add_edge(i*n+j, (n-1)*n, weight = weight)
                        else:
                            weight = self.getFriendshipWeight()
                            self.graph.add_edge(i*n+j, (i-1)*n, weight = weight)
                    if( i == n-1):
                        if( j != n-1):
                            weight = self.getFriendshipWeight()
                            self.graph.add_edge(i*n+j, j+1, weight = weight)
                        if(j != 0):
                            weight = self.getFriendshipWeight()
                            self.graph.add_edge(i*n+j, j-1, weight = weight)
                        else: 
                            weight = self.getFriendshipWeight()
                            self.graph.add_edge(i*n+j, (n-1), weight = weight)
        if(skew != 0 and not args["continuous"] ): 
            num = round(abs(skew)*len(self.graph.nodes))
            indexes = random.sample(range(len(self.graph.nodes)), num)
            for i in indexes:
                self.graph.nodes[i]['agent'].state = STATES[1]

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


def saveModels(models, filename):
    with open(filename, 'wb') as f:
        dill.dump(models, f)

def loadModels(filename):
    with open(filename, 'rb') as f:
        models = dill.load(f)
    return models

def findClusters(model):
    partition = community_louvain.best_partition(model.graph)
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
    if(model.partition==None):
        partition = findClusters(model)
    else:
        partition = model.partition

    for k, v in partition.items():
        model.graph.node[k]["louvain-val"] = v
    degrees = nx.degree(model.graph)

    #colors = [mypalette[G.node[node]["louvain-val"] %9 ]  for node in G.nodes()]
#    edge_col = [mypalette[model.graph.node[node]["louvain-val"]+1 % 8 ]  for node in model.graph.nodes()]
    edge_col = []
    for node in model.graph.nodes():
        edge_col.append(mypalette[model.graph.node[node]["louvain-val"] % 9 ])
    #plt.figure(figsize=(16,16))
    plt.subplot(1, 2, 2, title="Cluster Membership")
    nx.draw(model.graph, model.pos, node_size=[d[1] * 20 for d in degrees], node_color =edge_col)
    (clusters, sd, clsize) = findAvgStateInClusters(model, part= partition)
    text = [f'avg={clusters[c]:5.2f} sd={sd[c]:5.2f} n={clsize[c]}' for c in range(len(clusters))]
    #print(text)
    ax = plt.gca()
    handles = [mpatches.Patch(color=mypalette[c], label=text[c]) for c in range(len(text))]
    ax.legend(handles=handles)
    #plt.title("Snapshot of network with states and clusters")
    draw_model(model)#, outline=edge_col, partition = partition)
    
    
#-------- save data functions ---------

def saveavgdata(models, filename, clusterSD):
    states = []
    sds = []
    for i in range(len(models)):
        states.append(models[i].states)
        sds.append(models[i].statesds)
    array = np.array(states)
    avg = array.mean(axis=0)
    std = np.array(sds).mean(axis=0)
    outs = np.column_stack((avg,std))
    hstring = 'avg.std'

    if(clusterSD):
        avgSds = []
        for mod in models:
            array = np.array(mod.clusterSD)
            avgSd = array.mean(axis=1)
            avgSds.append(avgSd)
        array = np.array(avgSds)
        avgAvgSd = array.mean(axis=0)
        outs = np.column_stack((outs,avgAvgSd))
        hstring += ',clstd'

    np.savetxt(filename,outs,delimiter=',',header=hstring) 
    
def savesubdata(models,filename):
    
    outs = []

    for i in range(len(models)):
        outs.append(models[i].states)
    
    outs = np.array(outs)
    np.savetxt(filename,outs,delimiter=',')

#-------- drawing functions ---------

def draw_model(model, save=True, filenumber = None, outline=None, partition=None, extraTitle=""):
    
    #plt.figure(figsize=(4, 4))
    #plt.subplot(1, 2, 1, title="State of the Nodes")
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
    #sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=-1, vmax=1))
    #sm.set_array([])
    #cbar = plt.colorbar(sm)
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


def drawAvgState(models, avg =False, pltNr=1, title="", clusterSD = False):
    plt.xlabel("timestep")
    plt.ylabel("AVG // STD")
    #mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "grey","grey"]
    plt.subplot()
    #plt.subplot(1, 2, 1, title="Average State and SD")
    
    if(not avg):
        plt.ylim((-1, 1))
        for i in range(len(models)):
            plt.plot(models[i].states ,color='#ff7f0e')
            plt.plot(models[i].statesds, alpha=0.5 ,color='#ff7f0e')
            if(clusterSD):
                sds = np.array(models[i].clusterSD)
                avgsd = sds.mean(axis=1)
                plt.plot(avgsd, linestyle=":" ,color='#ff7f0e')
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
        p1 = plt.plot(avg, color='#ff7f0e', label="AVG state")
        p2 = plt.plot(std, color='#ff7f0e', alpha=0.5, label="STD states")
        #plt.plot(avg+std, color=col.to_rgba(mypalette[pltNr-1], 0.5))
        #text = ["rand cont", "cl cont", "rand disc", "cl disc"]
        text =["Clustered"]
        handles = [mpatches.Patch(color=mypalette[c], label=text[c]) for c in range(len(text))]
        plt.legend(handles=handles)
        #print(models[0].states)
        if(clusterSD):
            avgSds = []
            for mod in models:
                array = np.array(mod.clusterSD)
                avgSd = array.mean(axis=1)
                avgSds.append(avgSd)
            array = np.array(avgSds)
            avgAvgSd = array.mean(axis=0)
            plt.plot(avgAvgSd, color='#ff7f0e', linestyle=":", label="STD in clusters")

        #plt.subplot(1, 2, 2)
        #plt.ylim((0, 1))
        #plt.plot(std, color=mypalette[pltNr-1])
        return (p1, p2)

def drawCrossSection(models, pltNr = 1):
    values = []
    #mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "grey","grey"]
    for model in models:
        values.append(model.states[-1])
    plt.subplot(1, 2, 2, title="Density Plot of State for Simulations")
    ax = plt.gca()
    #ax.set_xscale('log')
    plt.xlim((0, 5))
    plt.ylim((-1, 1))
    #plt.title('Density Plot of state for simulations')
    #plt.xlabel('avg state of cooperators after all time steps')
    plt.xlabel('Density')
    #plt.scatter(range(len(values)), values.sort(), color = mypalette[pltNr-1])
    try:
        sns.distplot(values, hist=False, kde=True, color = mypalette[pltNr-1], vertical=True)
    except:
        sns.distplot(values, hist=True, kde=False, color = mypalette[pltNr-1], vertical=True)

    #plt.show()

def drawClustersizes(models, pltNr = 1):
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
    plt.title("Density of Avg State in Communities")
    if(step < 0):
        plt.subplot(1, 3, 3, title="Avg State after Simulation")
        states = []
        for i in range(len(models)):
            for c in models[i].clusteravg[-1]:
                states.append(c)
    else:
        plt.subplot(1, 3, subplot, title="Avg State at t="+ str(step))
        states = []
        for i in range(len(models)):
            for c in models[i].clusteravg[step]:
                states.append(c)
    ax = plt.gca()
    #ax.set_xscale('log')
    plt.xlim((0, 5))
    plt.ylim((-1, 1))
    #plt.title('Density Plot of state for simulations')
    #plt.xlabel('avg state of cooperators after all time steps')
    plt.xlabel('Density')
    plt.ylabel('State')
    try:
        sns.distplot(states, hist=True, kde=True, color = mypalette[pltNr-1], vertical=True)
    except:
        sns.distplot(states, hist=True, kde=False, color = mypalette[pltNr-1], vertical=True)

def drawAvgNumberOfAgreeingFriends(models, pltNr = 1):
    avgNbAgreeingFriends = [model.avgNbAgreeingList for model in models]
    avgAvg = np.array(avgNbAgreeingFriends).mean(axis=0)
    plt.title("Average Agreement of Neighbours")
    plt.ylim((0, 1))
    plt.xlabel("Timesteps")
    plt.ylabel("Agreement")
    plt.plot(avgAvg, color=mypalette[pltNr-1])

