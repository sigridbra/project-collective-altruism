# model.py
import random

states = [1, -1] #1 being cooperating, -1 being defecting

defectorUtility = -0.20 
 
politicalClimate=0.2 

selfWeight = 0.8

#neighboursWeight = 0.5

#Helper
def decision(probability):
    return random.random() < probability
        
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
        if(weight > 1):
            self.state = states[0]
        elif (weight < -1) :
            self.state = states[1]  
    
    def addInteractionGiven(self):
        self.interactionsGiven +=1
        
    def groupConsiderA(self, neighbourList):
        nbNeighbours = len(neighbourList)
        nbCoop = 0
        for n in  neighbourList:
            if(n['agent'].state > 0): nbCoop += 1
        p = nbCoop/nbNeighbours
        if(self.state > 0):
            if(decision(p)):
                self.state = 1
            else:
                self.state = -1
        else:
            if(decision(1-p)):
                self.state = -1
            else:
                self.state = 1
    
    def setState(self, newState):
        if(newState >= states[1] and newState <= states[0]):
            self.state = newState
        else:
            print("Error state outside state range: ", newState)
        