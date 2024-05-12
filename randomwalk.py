import numpy as np
import random


class RandomWalkOptimization:
    def __init__(self, graph, signatures, start, criterion):
        if graph.shape[0] != graph.shape[1]:
            raise ValueError("Graph adjacency matrix dimensions must be square.")

        self.graph = graph
        self.start = start
        self.signatures = signatures
        self.criterion = criterion

    def optimize(self, objective, callback=None):
        visited = set()
        current_node = self.start
        best_node = (None, float("inf"), float("inf"))

        while True:
            if current_node in visited:
                break

            visited.add(current_node)

            # test loss = index 1
            metric, loss = objective(current_node)

            if loss < best_node[2]:
                best_node = (self.signatures[current_node], metric, loss)

            if callback:
                callback(best_node)

            neighbors = np.where(self.graph[current_node] > 0)[0]

            if len(neighbors) == 0:
                break

            current_node = random.choice(neighbors)

        return best_node

    def normalize_graph(self):
        rowsums = self.graph.sum(axis=1)
        self.graph = self.graph / rowsums[:, np.newaxis]
