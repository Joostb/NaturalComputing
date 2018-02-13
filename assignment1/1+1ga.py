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


if __name__ == "__main__":
    n = 100
    n_iter = 1500
    p = 1/n

    fitnesses = []
    times_found = 0
    for _ in range(10):
        _, best_fitness, fitness_list = ga(count_ones, n, n_iter, p)
        print("Found Fitness: {}".format(best_fitness))
        fitnesses.append(fitness_list)
        if best_fitness == n:
            times_found += 1

    for fitness in fitnesses:
        plt.plot(fitness)
    plt.ylabel("Best Fitness")
    plt.xlabel("Iteration")
    plt.show()

    print("Optimum found {} times".format(times_found))
