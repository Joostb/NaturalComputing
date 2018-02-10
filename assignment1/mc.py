
import numpy as np
import random
from typing import Callable

def random_sequence(length : int) -> int:
    """ An integer is really a string of bits"""
    return random.randint(0, 2**length - 1)


def count_ones(genes : int) -> int:
    """ Convert to binary representation and count the 1's"""
    return bin(genes).count("1")

def mc(fitness_function : Callable[[int], int], n : int, iterations : int) -> (int, int):

    best_individual = random_sequence(n)
    best_fitness = fitness_function(best_individual)

    for i in range(iterations):
        sequence = random_sequence(n)
        fitness = fitness_function(sequence)
        if fitness > best_fitness:
            best_individual = sequence
            best_fitness = fitness
        
    return best_individual, best_fitness


print(mc(count_ones, 10, 1500))
print("running the algorithm 1500 * 10 times")
print("best individual fount: 0 times")

#  the chance to not find the best individual in 1500 runs is:
# 0.999999999999999999999999998816708642168482291882407152775...

# print(fitness(2**1-1))