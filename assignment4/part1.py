import math
import random
import numpy as np

def formula(n, p):
    
    total = 0
    for i in range(math.ceil(n/2), n+1):
        permutation = math.factorial(n) / (math.factorial(i) * math.factorial(n-i))
        total += math.pow(p,i) * math.pow((1-p), (n-i)) * permutation 
        
    return total

def simulation(n, p): 
    
    diagnostic = np.zeros(n)
    for i in range(n):
        if random.random() <= p:
            diagnostic[i] = 1
        
    prob = sum(diagnostic)/n
    
    return prob
        

n = 21
p = 0.6

result = formula(n,p)
sim = simulation(n, p)

print(result)
print(sim)

