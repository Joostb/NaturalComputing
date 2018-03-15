import numpy as np
import random


class Particle(object):
    """
    Particle corresponds to one centroid
    """
    def __init__(self, n_dim, n_clusters, datapoints, x_min=0, x_max=1):
        self.n_clusters = n_clusters
        self.datapoints = datapoints
        # self.centroid_position = [[np.random.uniform(x_min[i], x_max[i]) for i in range(n_dim)] for _ in range(n_clusters)]
        self.centroid_positions = np.random.uniform(x_min, x_max, size=(n_clusters, n_dim))
        self.velocity = np.random.rand( n_clusters, n_dim)

        # Local best
        self.best_fitness = float('inf')
        self.best_position = self.centroid_positions.copy()

    def fitness(self):
        """
        Fitness of one particle is the negative average inter-distance (since we want to maximize fitness)
        :return:
        """

        inter_cluster_distance = [self.distance(datapoint) for datapoint in self.datapoints]
        averaged_inter_distance = sum(inter_cluster_distance)/len(self.datapoints)

        cluster_fitness = averaged_inter_distance

        if cluster_fitness < self.best_fitness:
            self.best_fitness = cluster_fitness
            self.best_position = self.centroid_positions.copy()

        return cluster_fitness

    def quantization(self):

        inter_cluster_distance = [self.distance(datapoint, with_centroid=True) for datapoint in self.datapoints]
        cluster_sizes = np.zeros(self.n_clusters)
        sums = np.zeros(self.n_clusters)
        for d, cluster in inter_cluster_distance:
            
            cluster_sizes[cluster] += 1
            sums[cluster] += d

        return np.sum(d/cluster_sizes)/self.n_clusters

    # def assign(self, datapoints):
    #     self.datapoints = datapoints

    def distance(self, datapoint, with_centroid=False):
        """
        The distance is defined as the euclidean distance
        :param datapoint:
        :return:
        """

        min_d = np.inf
        best_centroid = 0
        for  i, centroid in enumerate(self.centroid_positions):
            
            d = np.sqrt(np.sum((datapoint - centroid)**2))
            if d < min_d:
                min_d = d
                best_centroid = i

        if with_centroid:
            return min_d, best_centroid

        return min_d

    def update_velocity(self, n_dim, global_best_position):
        """
        Update the velocity with using inertia and acceleration constants
        """
        w = 0.5
        c1 = 1 
        c2 = 2 
        
        r1 = random.random()
        r2 = random.random()
 
        self.velocity = w*self.velocity + c1*r1*(self.best_position - self.centroid_positions) + c2*r2*(global_best_position - self.centroid_positions)

    def update_position(self, n_dim, x_min=0, x_max=1):
        self.centroid_positions = self.centroid_positions + self.velocity
    
        self.centroid_positions = np.clip(self.centroid_positions, x_min, x_max)

        


        
        
