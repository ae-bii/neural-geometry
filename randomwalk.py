import numpy as np
import random


class RandomWalkOptimization:
    def __init__(self, graph, start, criterion):
        self.graph = graph
        self.start = start
        self.criterion = criterion

    def optimize(self, callback, criterion=None):
        if criterion is None:
            criterion = self.criterion

        pass

    def normalize_graph(self):
        pass
