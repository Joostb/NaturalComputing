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
        Fitness of one particle is the same as the quantization error
        :return:
        """
        return self.quantization()


    def quantization(self):

        inter_cluster_distance = [self.distance(datapoint, with_centroid=True) for datapoint in self.datapoints]
        cluster_sizes = np.zeros(self.n_clusters)
        sums = np.zeros(self.n_clusters)
        for d, cluster in inter_cluster_distance:
            cluster_sizes[cluster] += 1
            sums[cluster] += d

        return np.sum(sums / (cluster_sizes+1))/self.n_clusters

    # def assign(self, datapoints):
    #     self.datapoints = datapoints

    def distance(self, datapoint, with_centroid=False):
        """
        The distance is defined as the euclidean distance
        :param datapoint:
        :return:
        """

        min_d = np.inf
        best_centroid = -1
        for i, centroid in enumerate(self.centroid_positions):
            
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
        w = 0.72
        c1 = 1.49
        c2 = 1.49
        
        r1 = random.random()
        r2 = random.random()
 
        self.velocity = w*self.velocity + c1*r1*(self.best_position - self.centroid_positions) + c2*r2*(global_best_position - self.centroid_positions)
        return self.velocity
    
    def update_position(self, n_dim, x_min=0, x_max=1):
        
        self.centroid_positions = self.centroid_positions + self.velocity
        self.centroid_positions = np.clip(self.centroid_positions, x_min, x_max)
        
    def update_position_kmeans(self):
        
        centroid_update = np.zeros(self.centroid_positions.shape)
        centroid_nb = np.zeros(self.n_clusters)
        for datapoint in self.datapoints: 
            d, centroid = self.distance(datapoint, with_centroid=True)
            centroid_update[centroid] += datapoint
            centroid_nb[centroid] += 1
        
        for centroid in range(self.n_clusters):
            self.centroid_positions[centroid] = centroid_update[centroid]/centroid_nb[centroid]
