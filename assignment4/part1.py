import math
import random

def formula(n, p):
    
    total = 0
    for i in range(int(n/2)+1, n+1):
        permutation = math.factorial(n) / math.factorial(i) * math.factorial(n-i)
        total += math.pow(p,i) * math.pow((1-p), (n-i)) * permutation 
        
    return total

def simulation(n, p): 
    
    diagnostic = []
    for i in range(n):
        diagnostic.append(random.randint(0, 1))
        
    prob = sum(diagnostic)/n
    
    return prob
        

n = 21
p = 0.6

result = formula(n,p)
sim = simulation(n, p)

print(result)
print(sim)

