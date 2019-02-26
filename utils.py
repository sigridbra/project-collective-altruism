states = [1, -1] #1 being cooperating, -1 being defecting

defectorUtility = -0.20 
 
politicalClimate=0.2 

selfWeight = 0.8

neighboursWeight = 0.5


def standardDeviation(fc):
    res = 4*fc*(1-fc)**3+24*(fc*(1-fc))**2+36*(1-fc)*fc**3+16*fc**4-16*fc**2
    return (res)




def correlation(model, depth):
    AgreeingNeighbours = [[0 for i in range(depth)] for j in range(len(model.graph))]
    
    for nodeIdx in model.graph:
        neighbours = list(model.graph.adj[nodeIdx])
        isCooperator = model.graph.nodes[nodeIdx]['agent'].state > 0
        parent = [nodeIdx]
        for d in range(depth):
            nextLevelNeighs = set([])
            for n in neighbours:
                nextLevelNeighs.update(list(model.graph.adj[n]))
                if(model.graph.nodes[n]['agent'].state > 0 and isCooperator):
                    AgreeingNeighbours[nodeIdx][d] += 1
                elif(model.graph.nodes[n]['agent'].state <= 0 and not isCooperator): 
                    AgreeingNeighbours[nodeIdx][d] += 1
            for n in parent:
                nextLevelNeighs.discard(n) 
            parent = neighbours
            neighbours = nextLevelNeighs
            
    mat = np.matrix(AgreeingNeighbours)
    avg = mat.mean(axis=0)
    arrays = [[] for i in range(depth)]
    res = [0 for i in range(depth)]
    for nodeIdx in model.graph:
        neighbours = list(model.graph.adj[nodeIdx])
        for d in range(depth):
            for n in neighbours:
                arrays[d].append((AgreeingNeighbours[nodeIdx][d]-avg[0, d])*(AgreeingNeighbours[n][d]-avg[0, d]))
    for d in range(depth):
        res[d] = mean(arrays[d])
        
    return res

  
def drawDefectingNeighbours(defectorDefectingNeighsList, cooperatorDefectingNeighsList, defectorDefectingNeighsSTDList, cooperatorDefectingNeighsSTDList, filname = None):
    steps = range(0, len(defectorDefectingNeighsList))
    defector = np.array( defectorDefectingNeighsList)
    defectorSTD = np.array( defectorDefectingNeighsSTDList)
    cooperator = np.array(cooperatorDefectingNeighsList)
    cooperatorSTD = np.array( cooperatorDefectingNeighsSTDList)
     
    plt.figure(figsize=(12, 9))  
    
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].set_title('Development of defector neighbours')
  
    # Remove the plot frame lines. They are unnecessary chartjunk.  
    #axarr[0] = plt.subplot(111)  
    #ax.spines["top"].set_visible(False)  
    #ax.spines["right"].set_visible(False)  
   
    #ax.get_xaxis().tick_bottom()  
    #ax.get_yaxis().tick_left()  
   
  
    # Along the same vein, make sure your axis labels are large  
    # enough to be easily read as well. Make them slightly larger  
    # than your axis tick labels so they stand out.  
    plt.ylabel("avg number of defector friends", fontsize=12)  
  
    # Use matplotlib's fill_between() call to create error bars.    
    axarr[0].fill_between(steps, cooperator - cooperatorSTD,  
                 cooperator + cooperatorSTD,  color="#397c39")  
  
    # Plot the means as a white line in between the error bars.   
    # White stands out best against the dark blue.  
    axarr[0].plot(steps, cooperatorDefectingNeighsList, color="white", lw=2) 
   
    axarr[1].fill_between(steps, defector - defectorSTD,  
                 defector + defectorSTD,  color="#7c393a")  
  
    # Plot the means as a white line in between the error bars.   
    # White stands out best against the dark blue.  
    axarr[1].plot(steps, defectorDefectingNeighsList, color="white", lw=2) 
    
    #plt.title("Development of defector neighbours", fontsize=22)  
    
  
    # Finally, save the figure as a PNG.  
    # You can also save it as a PDF, JPEG, etc.  
    # Just change the file extension in this call.  
    # bbox_inches="tight" removes all the extra whitespace on the edges of your plot.  
    plt.savefig("avg" + filname +".png", bbox_inches="tight");
    
def drawSD(ratio, defectorDefectingNeighsSDList, cooperatorDefectingNeighsSDList):
    steps = range(0, len(defectorDefectingNeighsSDList))
    defectorSD = np.array( defectorDefectingNeighsSDList)
    cooperatorSD = np.array( cooperatorDefectingNeighsSDList)
    ratioArr = np.array(ratio)
    oppositeRatioArr = 1 - ratioArr
    #print(ratioArr, oppositeRatioArr)
    sd = standardDeviation(ratioArr)
    defsd = standardDeviation(oppositeRatioArr)
    #print(sd, defsd)
    
    plt.title("Estimated SD for random distribution - SD at all time steps ")
    plt.xlabel("Timesteps")
    plt.plot(steps, sd -cooperatorSD, color="#397c39")
    plt.plot(steps, defsd- defectorSD, color="#7c393a")