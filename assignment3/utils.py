import numpy as np
import random


class Particle(object):
    """
    Particle corresponds to one centroid
    """
    def __init__(self, n_dim, n_clusters, x_min=0, x_max=1):
        self.n_clusters = n_clusters
        self.datapoints = []
        self.position = [np.random.uniform(x_min[i], x_max[i]) for i in range(n_dim)]
        self.velocity = [np.random.rand() for _ in range(n_dim)]
        self.best = self.position.copy()
        self.best_fitness = -float('inf')
        # Local best
        self.best_position = self.position.copy()

    def fitness(self):
        """
        Fitness of one particle is the negative average inter-distance (since we want to maximize fitness)
        :return:
        """
        if len(self.datapoints) is 0:
            return -float('inf')
        inter_cluster_distance = [self.distance(datapoint) for datapoint in self.datapoints]
        averaged_inter_distance = sum(inter_cluster_distance)/len(self.datapoints)

        cluster_fitness = -averaged_inter_distance

        if cluster_fitness > self.best_fitness:
            self.best_fitness = averaged_inter_distance
            self.best_position = self.position.copy()

        return cluster_fitness

    def assign(self, datapoints):
        self.datapoints = datapoints

    def distance(self, datapoint):
        """
        The distance is defined as the euclidean distance
        :param datapoint:
        :return:
        """
        d = (datapoint - self.position)**2
        d = np.sqrt(sum(d))
        return d


    def update_velocity(self, n_dim):
        """
        Update the velocity with using inertia and acceleration constants
        """
        w = 0.5
        c1 = 2 
        c2 = 2 

        for i in range(0, n_dim):
            r1 = random.random()
            r2 = random.random()
            
            cognitive = c1*r1*(self.best_position[i] - self.position[i])
            social = c2*r2*(self.best[i] - self.position[i])
            self.velocity[i] = w*self.position[i] + cognitive + social

    def update_position(self, n_dim):
        for i in range(0, n_dim):
            self.position[i] = self.position[i] + self.velocity[i]


        


        
        
