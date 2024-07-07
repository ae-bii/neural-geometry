import numpy as np
import random


class RandomWalkOptimizer:
    """
    RandomWalkOptimizer class for optimizing a given objective function using random walk algorithm.

    Args:
        graph (numpy.ndarray): The adjacency matrix of the graph.
        signatures (list): List of signatures corresponding to each node in the graph.
        start (int): The starting node for the random walk.
        criterion (str): The criterion for optimization.

    Attributes:
        graph (numpy.ndarray): The adjacency matrix of the graph.
        start (int): The starting node for the random walk.
        signatures (list): List of signatures corresponding to each node in the graph.
        criterion (str): The criterion for optimization.

    Methods:
        optimize(objective, max_iters, callback=None): Optimizes the objective function using random walk algorithm.
        normalize_graph(): Normalizes the graph by dividing each row by its sum.
        optimize_with_backtracking(objective, max_iters, callback=None): Optimizes the objective function using random walk algorithm with backtracking.
    """

    def __init__(self, graph, signatures, start, criterion):
        if graph.shape[0] != graph.shape[1]:
            raise ValueError("Graph adjacency matrix dimensions must be square.")

        self.graph = graph
        self.start = start
        self.signatures = signatures
        self.criterion = criterion

    def optimize(self, objective, max_iters, callback=None):
        """
        Optimizes the objective function using random walk algorithm.

        Args:
            objective (function): The objective function to be optimized.
            max_iters (int): The maximum number of iterations for the optimization.
            callback (function, optional): A callback function to be called after each iteration. Defaults to None.

        Returns:
            tuple: A tuple containing the best node, metric, and loss.
        """
        visited = set()
        current_node = self.start
        best_node = (None, float("inf"), float("inf"))
        it = 0

        while it < max_iters:
            print("--------------------")
            print("Iteration: " + str(it + 1))
            print("--------------------")
            # if current_node in visited:
            #     break

            visited.add(current_node)

            # test loss = index 1
            metric, loss = objective(self.signatures[current_node])

            if loss < best_node[2]:
                best_node = (self.signatures[current_node], metric, loss)

            if callback:
                callback(self.signatures[current_node], (metric, loss))

            neighbors = np.where(self.graph[current_node] > 0)[0]

            if len(neighbors) == 0:
                break

            current_node = random.choice(neighbors)
            it += 1

        return best_node

    def normalize_graph(self):
        """
        Normalizes the graph by dividing each row by its sum.
        """
        rowsums = self.graph.sum(axis=1)
        self.graph = self.graph / rowsums[:, np.newaxis]

    def optimize_with_backtracking(self, objective, max_iters, callback=None):
        """
        Optimizes the objective function using random walk algorithm with backtracking.

        Args:
            objective (function): The objective function to be optimized.
            max_iters (int): The maximum number of iterations for the optimization.
            callback (function, optional): A callback function to be called after each iteration. Defaults to None.

        Returns:
            tuple: A tuple containing the best node, metric, and loss.
        """
        visited = set()
        current_node = self.start
        best_node = (None, float("inf"), float("inf"))
        it = 0
        path = []  # to keep track of the path taken

        while it < max_iters:
            if current_node in visited:
                if path:
                    current_node = path.pop()
                else:
                    break
                continue

            print("--------------------")
            print("Iteration: " + str(it + 1))
            print("--------------------")

            visited.add(current_node)
            path.append(current_node)

            metric, loss = objective(self.signatures[current_node])

            if loss < best_node[2]:
                best_node = (self.signatures[current_node], metric, loss)

            if callback:
                callback(self.signatures[current_node], (metric, loss))

            neighbors = np.where(self.graph[current_node] > 0)[0]

            # filter out visited neighbors
            neighbors = [node for node in neighbors if node not in visited]

            if not neighbors:
                path.pop()

                if path:
                    # backtrack to the previous node if there are no unvisited neighbors
                    current_node = path[-1]
                else:
                    break
            else:
                current_node = random.choice(neighbors)

            it += 1

        return best_node
