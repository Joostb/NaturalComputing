import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import datasets
from tqdm import tqdm
import random

from utils import Particle


def pso(X, y, n_iter=100, n_clusters=3, n_particles = 10):
    n_features = X.shape[1]
    n_samples = X.shape[0]
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    particles = [Particle(n_features, n_clusters, X) for _ in range(n_particles)]
    best_fitness = float('inf')
    global_best_position = None

    for iter in tqdm(range(n_iter)):
        fitnesses = np.zeros(n_particles)
        for i,particle in enumerate(particles):
            fitnesses[i] =  particle.fitness()
        
        best_index = np.argmin(fitnesses)
        best_particle = particles[best_index]
        best_new_fitness = fitnesses[best_index]

        if best_new_fitness < best_fitness:
            global_best_position = best_particle
            best_fitness = best_new_fitness
        
        for particle in particles:
            particle.update_velocity(n_features, global_best_position.centroid_positions)
            particle.update_position(n_features, X_min, X_max)

    return global_best_position.quantization()
    


def kmeans(X, y, n_iter=100, n_clusters=3, n_particles = 1):
    n_features = X.shape[1]
    n_samples = X.shape[0]
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    particle = Particle(n_features, n_clusters, X, X_min, X_max) 
    best_fitness = float('inf')

    for iter in tqdm(range(n_iter)):
        #particle.fitness()

        particle.update_position_kmeans()
            
    print(particle.fitness())
    return particle.quantization()
        

def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target   

    n_classes = len(np.unique(y))
    error_pso = pso(X, y, n_clusters=n_classes)
    error_kmeans = kmeans(X,y,n_clusters=n_classes)

    print("error pso", error_pso)
    print("error kmeans", error_kmeans)


if __name__ == "__main__":
    main()
