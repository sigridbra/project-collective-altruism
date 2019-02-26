


model = MoneyModel(10)
for i in range(10):
    model.step()
    
%matplotlib inline
import matplotlib.pyplot as plt
agent_wealth = [a.wealth for a in model.schedule.agents]
plt.hist(agent_wealth)