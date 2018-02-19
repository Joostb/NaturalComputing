import operator
import math
from deap import algorithms, base, creator, tools, gp
import numpy as np
import matplotlib.pyplot as plt

dep_var = np.arange(-1, 1.1, 0.1)
output = np.array(
    [0, -0.1629, -0.2624, -0.3129, -0.3264, -0.3125, -0.2784, -0.2289, -0.1664, -0.0909,
     0, 0.1111, 0.2496, 0.4251, 0.6496, 0.9375, 1.3056, 1.7731, 2.3616, 3.0951, 4])


def evaluate(individual, points, labels):
    func = toolbox.compile(expr=individual)
    res = [func(x) for x in points]
    res = np.array([x if x is not None else math.inf for x in res])
    abs_errors = abs(res - labels)
    return -sum(abs_errors) / len(points),


def protected_div(left, right):
    if right != 0:
        return left / right
    else:
        return 0


def protected_log(x):
    if x > 0:
        return np.log(x)
    else:
        return 0


def protected_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return 9999999999999999


# Define primitives
primitives = gp.PrimitiveSet(name="MAIN", arity=1)
primitives.addPrimitive(operator.add, arity=2)
primitives.addPrimitive(operator.sub, arity=2)
primitives.addPrimitive(operator.mul, arity=2)
primitives.addPrimitive(protected_div, arity=2)
primitives.addPrimitive(protected_log, arity=1)
primitives.addPrimitive(protected_exp, arity=1)
primitives.addPrimitive(math.sin, arity=1)
primitives.addPrimitive(math.cos, arity=1)
primitives.renameArguments(ARG0="x")

# Maximize fitness, weights 1
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=primitives, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=primitives)
toolbox.register("evaluate", evaluate, points=dep_var, labels=output)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitives)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


if __name__ == "__main__":
    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # cxpb prob of mating
    # mutpb prob of mutation, i.e. crossover
    # ngen number of generations
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0, ngen=50, stats=mstats,
                                   halloffame=hof, verbose=True)

    best_indivs_fitness = []
    for gen in log.chapters['fitness']:
        best_indivs_fitness.append(gen['max'])

    best_indivs_size = []
    for gen in log.chapters['size']:
        best_indivs_size.append(gen['min'])

    gen_size = []
    for gen in log:
        gen_size.append(gen['nevals'])

    plt.plot(best_indivs_fitness)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.show()

    plt.plot(best_indivs_size)
    plt.xlabel("Generation")
    plt.ylabel("Smallest Size")
    plt.show()

    plt.plot(gen_size)
    plt.xlabel("Generation")
    plt.ylabel("Generation Size")
    plt.show()

    print("The best individual was: ", str(hof[0]))
