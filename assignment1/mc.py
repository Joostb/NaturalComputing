import numpy as np
import matplotlib.pyplot as plt


def random_sequence(n):
    """
    Generate random bitstring of length n
    :param n:
    :return:
    """
    bitlist = np.random.randint(2, size=n)
    return ' '.join(str(bit) for bit in bitlist)


def count_ones(genes):
    """
    Summation of bitstring, implement as number of 1s in string
    :param genes: bitstring
    :return:
    """
    return genes.count("1")


def mc(fitness_function, n, iterations):
    best_fitness_list = []
    best_individual = random_sequence(n)
    best_fitness = fitness_function(best_individual)

    for i in range(iterations):
        best_fitness_list.append(best_fitness)
        sequence = random_sequence(n)
        fitness = fitness_function(sequence)
        if fitness > best_fitness:
            best_individual = sequence
            best_fitness = fitness

    return best_individual, best_fitness, best_fitness_list


if __name__ == "__main__":
    n_iter = 1500
    n = 100
    print("Running the algorithm 10 times with: \n \t n_iters: {} \n \t n: {}".format(n_iter, n))
    times_found = 0
    fitnesses = []
    for _ in range(10):
        best_individual, best_fitness, best_fitness_list = mc(count_ones, n, n_iter)
        times_found += 1 if best_fitness is n else 0
        fitnesses.append(best_fitness_list)

    for fitness in fitnesses:
        plt.plot(fitness)
    plt.xlabel("Iterations")
    plt.ylabel("Best Fitness")
    plt.show()

    print("Best fitness is found {} times".format(times_found))
