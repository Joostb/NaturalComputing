import math
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from pandas import *
from tqdm import tqdm


def formula(n, p):
    """
    Formula for calculating the probability given n people with competence p
    :param n: size jury
    :param p: competence
    :return: probability
    """
    total = 0
    for i in range(int(math.ceil(n/2)), n+1):
        permutation = math.factorial(n) / (math.factorial(i) * math.factorial(n-i))
        total += math.pow(p, i) * math.pow((1-p), (n-i)) * permutation
        
    return total


def simulation(n, p, n_simulations=10000):
    """ 
        We run n_simulations simulations, and check in what fraction we make the correct
        prediction
    """
    correct = 0
    for _ in range(n_simulations):
        diagnostic = np.zeros(n)
        for i in range(n):
            if random.random() <= p:
                diagnostic[i] = 1
            
        prob = sum(diagnostic)/n
        if prob > 0.5:
            correct += 1
    
    return correct / n_simulations


def plot_surf_probs(jury_max_size=50, jury_step_size=1, step=0.05):
    probs = np.arange(0, 1, step)
    juries = np.arange(1, jury_max_size, jury_step_size)

    X, Y = np.meshgrid(juries, probs)
    Z = np.zeros((len(probs), len(juries)))

    for y, jury in enumerate(tqdm(juries)):
        for x, prob in enumerate(probs):
            Z[x, y] = formula(jury, prob)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)
    ax.set_xlabel("jury_size")
    ax.set_ylabel("competence")
    ax.set_zlabel("probability")

    plt.show()


def ex_1bc():
    n = 21
    p = 0.6

    result = formula(n, p)
    sim = simulation(n, p)

    print("Actual result:", result)
    print("Simulatation result:", sim)
    print("Difference:", abs(result-sim))


def ex_1e():
    groups = {"radiologist": (0.85, 1), "doctors": (0.8, 3), "students": (0.6, 21)}

    res = []

    for name, params in groups.items():
        res.append(formula(params[1], params[0]))
        print(name, res[-1])

    n_students = 1
    probs_students = []
    while True:
        probs_students.append(formula(n_students, groups["students"][0]))
        if probs_students[-1] > res[1]:
            break
        n_students += 1

    print("Students needed for approximation:", n_students)
    print("Approximations:", probs_students[-2], probs_students[-1])


if __name__ == "__main__":
    ex_1bc()

    plot_surf_probs(25, 1, 0.05)

    ex_1e()
