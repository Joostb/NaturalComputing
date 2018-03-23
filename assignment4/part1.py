import math
import random
import numpy as np
import matplotlib.pyplot as plt
from pandas import *

def formula(n, p):
    
    total = 0
    for i in range(math.ceil(n/2), n+1):
        permutation = math.factorial(n) / (math.factorial(i) * math.factorial(n-i))
        total += math.pow(p,i) * math.pow((1-p), (n-i)) * permutation 
        
    return total

def simulation(n, p, n_simulations=10000):
    """ 
        We run n_simulations simulations, and check in what fraction we make the correct
        prediction
    """
    correct = 0
    for i in range(n_simulations):
        diagnostic = np.zeros(n)
        for i in range(n):
            if random.random() <= p:
                diagnostic[i] = 1
            
        prob = sum(diagnostic)/n
        if prob > 0.5:
            correct += 1
    
    return correct / n_simulations
        

def plotting(jury_max_size, jury_step_size, step=0.2):
    probabilities = np.arange(0,1,step)
    juries = np.arange(1, jury_max_size, jury_step_size)
    # for p in probabilities:
    #     for j in juries:
    table = [[formula(j,p) for j in juries] for p in probabilities]

    axs = plt.subplot(frame_on=False)
    axs.xaxis.set_visible(False) 
    axs.yaxis.set_visible(False)

    the_table = axs.table(
        cellText=table, 
        loc='center',
        rowLabels=['a','b','c','d','e'],
        colLabels=[1,2,3,4,5],
        # cellColours=table
        )

    plt.show()
n = 21
p = 0.6

plotting(10, 2)

result = formula(n,p)
sim = simulation(n, p)

print(result)
print(sim)

'''size = [2,5,10,20,40]
proba = [0.3, 0.4, 0.6, 0.8]

for p in proba:
    result = []
    for n in size:
        result.append(formula(n, p))
    plt.plot(size, result, label="Probability = {}".format(p))
    
plt.legend() 
plt.show()'''

