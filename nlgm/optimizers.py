import numpy as np
import random


class RandomWalkOptimizer:
    """
    Random-walk optimizer over a graph of candidate signatures.

    Parameters
    ----------
    graph : numpy.ndarray
        Square adjacency matrix.
    signatures : list
        Signature values indexed by graph node.
    start : int
        Starting node index.
    criterion : str
        Optimization criterion label.
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
        Optimize objective with random-neighbor traversal.

        Parameters
        ----------
        objective : callable
            Function taking a signature and returning ``(metric, loss)``.
        max_iters : int
            Maximum number of walk iterations.
        callback : callable, optional
            Callback invoked as ``callback(signature, (metric, loss))``.

        Returns
        -------
        tuple
            Best result as ``(signature, metric, loss)``.
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
        Normalize transition weights row-wise.
        """
        rowsums = self.graph.sum(axis=1)
        self.graph = self.graph / rowsums[:, np.newaxis]

    def optimize_with_backtracking(self, objective, max_iters, callback=None):
        """
        Optimize objective using random walk with path backtracking.

        Parameters
        ----------
        objective : callable
            Function taking a signature and returning ``(metric, loss)``.
        max_iters : int
            Maximum number of walk iterations.
        callback : callable, optional
            Callback invoked as ``callback(signature, (metric, loss))``.

        Returns
        -------
        tuple
            Best result as ``(signature, metric, loss)``.
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
