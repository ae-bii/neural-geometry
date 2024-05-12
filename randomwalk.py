import numpy as np
import random


class RandomWalkOptimization:
    def __init__(self, graph, start, criterion):
        if graph.shape[0] != graph.shape[1]:
            raise ValueError("Graph adjacency matrix dimensions must be square.")
        
        self.graph = graph
        self.start = start
        self.criterion = criterion

    def optimize(self, callback, criterion=None):
        if criterion is None:
            criterion = self.criterion

        pass

    def normalize_graph(self):
        rowsums = self.graph.sum(axis=1)
        self.graph = self.graph / rowsums[:, np.newaxis]
