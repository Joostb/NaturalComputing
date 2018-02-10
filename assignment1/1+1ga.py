
import numpy as np
import matplotlib.pyplot as plt

def random_sequence(n):
    return np.random.randint(2, size=n)

def count_ones(individual):
    return np.sum(individual)

def mutation(gene, n, p):
    """ generate a random sequence of 0's and 1's, where each bit has
     a probability of p to be 1 """

    mask = (np.random.rand(n) < p).astype(int)

    return (gene + mask) % 2  # mutate the sequence

def ga(fitness_function, n, iterations, p):
    
    best_individual = random_sequence(n)
    best_fitness = fitness_function(best_individual)
    l = []
    for i in range(iterations):
        sequence = mutation(best_individual, n, p)
        fitness = fitness_function(sequence)
        if fitness > best_fitness:
            best_individual = sequence
            best_fitness = fitness
        
        l.append(best_fitness)

    return best_individual, best_fitness, l


i,f,l = ga(count_ones, 1000, 5000, 1/1000)

plt.plot(l)
plt.show()

