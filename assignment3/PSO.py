import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import datasets
from tqdm import tqdm
import random

from utils import Particle


def pso(X, y, n_iter=100, n_clusters=3):
    n_features = X.shape[1]
    n_samples = X.shape[0]
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    particles = [Particle(n_features, n_clusters, x_min=X_min, x_max=X_max) for _ in range(n_clusters)]
    best_fitness = 0
    global_best = None

    for iter in tqdm(range(n_iter)):
        particle_distances = np.zeros(shape=(n_samples, n_clusters))
        for c, particle in enumerate(particles):
            for i, x in enumerate(X):
                particle_distances[i, c] = particle.distance(x)
    
        closest_cluster = np.argmin(particle_distances, axis=1)
        
        cluster_fitnesses = []
        for c, particle in enumerate(particles):
            particle.assign(X[closest_cluster == c])
            cluster_fitnesses.append(particle.fitness())

        bests = [{'fitness': particle.best_fitness, 'best_position': particle.best_position} for particle in particles]
        global_best = bests[np.argmin([best['fitness'] for best in bests])]['best_position'].copy()

        for particle in particles:
            particle.best_global = global_best
            particle.update_velocity(n_features)
            particle.update_position(n_features)

    quantization_error = []
    for particle in particles:
        quantization_error.append(particle.fitness())

    print(quantization_error)
    return quantization_error


def kmeans(X, y, n_iter=100, n_clusters=3):
    n_features = X.shape[1]
    n_samples = X.shape[0]
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    particles = [Particle(n_features, n_clusters, x_min=X_min, x_max=X_max) for _ in range(n_clusters)]
    best_fitness = 0
    global_best = None

    for iter in tqdm(range(n_iter)):
        particle_distances = np.zeros(shape=(n_samples, n_clusters))
        for c, particle in enumerate(particles):
            for i, x in enumerate(X):
                particle_distances[i, c] = particle.distance(x)

        closest_cluster = np.argmin(particle_distances, axis=1)
        for c, particle in enumerate(particles):
            particle.assign(X[closest_cluster == c])

        for c, particle in enumerate(particles):
            particle.position = sum(particle.datapoints)/len(particle.datapoints)      

    quantization_error = []
    for particle in particles:
        quantization_error.append(particle.fitness())

    return quantization_error


def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target    
    n_classes = len(np.unique(y))
    error_pso = pso(X, y, n_clusters=n_classes)
    error_kmeans = kmeans(X,y, n_clusters=n_classes)

    print("error pso", sum(error_pso)/3)
    print("error kmeans", sum(error_kmeans)/3)


if __name__ == "__main__":
    main()
