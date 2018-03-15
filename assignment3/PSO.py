import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import datasets
from tqdm import tqdm
import random
import copy

from utils import Particle


def pso(X, y, n_iter=100, n_clusters=3, n_particles=10):
    n_features = X.shape[1]
    n_samples = X.shape[0]
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    particles = [Particle(n_features, n_clusters, X) for _ in range(n_particles)]
    best_fitness = float('inf')
    global_best_position = None

    history_errors = np.zeros((n_iter, n_particles + 1))

    for iter in tqdm(range(n_iter)):
        fitnesses = np.zeros(n_particles)
        for i, particle in enumerate(particles):
            fitnesses[i] = particle.fitness()
        
        best_index = np.argmin(fitnesses)
        best_particle = particles[best_index]
        best_new_fitness = fitnesses[best_index]

        if best_new_fitness < best_fitness:
            global_best_position = copy.deepcopy(best_particle)
            best_fitness = best_new_fitness

        history_errors[iter, n_particles] = global_best_position.quantization()

        for p, particle in enumerate(particles):
            particle.update_velocity(n_features, global_best_position.centroid_positions)
            particle.update_position(n_features, X_min, X_max)

            history_errors[iter, p] = particle.quantization()

    print("error pso", global_best_position.quantization())
    return global_best_position.quantization(), history_errors
    

def kmeans(X, y, n_iter=100, n_clusters=3):
    n_features = X.shape[1]
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    particle = Particle(n_features, n_clusters, X, X_min, X_max)

    errors = []

    for _ in tqdm(range(n_iter)):
        particle.update_position_kmeans()
        errors.append(particle.quantization())

    print("error kmeans", particle.quantization())
    return particle.quantization(), errors
        

def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target   

    n_iter = 50

    n_classes = len(np.unique(y))

    n_clusters = n_classes

    plt.figure()
    plt.title("Comparison PSO and kmeans")
    plt.ylabel("Quantization Error")
    plt.xlabel("Iteration")

    error_kmeans_best, kmeans_error = kmeans(X, y, n_clusters=n_clusters, n_iter=n_iter)
    plt.plot(kmeans_error, label="kmeans")

    for epoch in range(10):
        error_pso_best, pso_error = pso(X, y, n_clusters=n_clusters, n_iter=n_iter)
        plt.plot(pso_error[:, 10], label="PSO_{}".format(epoch))  # 10 is the global best

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
