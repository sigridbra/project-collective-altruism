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