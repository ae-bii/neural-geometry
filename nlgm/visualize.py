import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def visualize_optimization_trajectory(
    adjacency_matrix, signatures, evaluated_signatures
):
    """
    Visualize the optimization trajectory in the graph search space.

    Args:
        adjacency_matrix (numpy.ndarray): Adjacency matrix representing the graph search space.
        signatures (list): List of signatures corresponding to each node in the graph.
        evaluated_signatures (list): List of signatures evaluated during the optimization process.
    """
    # Create a graph object from the adjacency matrix
    graph = nx.from_numpy_array(adjacency_matrix)

    # Assign labels to the nodes based on the signatures
    labels = {i: str(signature) for i, signature in enumerate(signatures)}
    nx.set_node_attributes(graph, labels, "signature")

    # Visualize the graph search space with the optimization trajectory
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color="lightblue")
    nx.draw_networkx_edges(graph, pos, edge_color="gray", width=1.0)
    nx.draw_networkx_labels(graph, pos, labels, font_size=10)

    # Highlight the optimization trajectory
    evaluated_indices = [
        signatures.index(signature) for signature in evaluated_signatures
    ]
    trajectory_edges = [
        (evaluated_indices[i], evaluated_indices[i + 1])
        for i in range(len(evaluated_indices) - 1)
    ]
    nx.draw_networkx_edges(
        graph, pos, edgelist=trajectory_edges, edge_color="red", width=2.0
    )

    plt.axis("off")
    plt.title("Graph Search Space with Optimization Trajectory")

    # Create the "figures" folder if it doesn't exist
    os.makedirs("figures", exist_ok=True)

    # Save the figure with a descriptive name
    plt.savefig("figures/optimization_trajectory.png")

    plt.show()


def visualize_validation_metrics(evaluated_metrics):
    """
    Visualize the evolution of validation metrics during the optimization process.

    Args:
        evaluated_metrics (list): List of validation metrics obtained during the optimization process.
    """
    # Visualize the metric evolution
    plt.figure()
    plt.plot(evaluated_metrics)
    plt.xlabel("Iteration")
    plt.ylabel("Validation Metric")
    plt.title("Metric Evolution during Optimization")

    # Save the figure with a descriptive name
    plt.savefig("figures/validation_metrics.png")

    plt.show()


def visualize_loss_trajectories(loss_trajectories):
    """
    Visualize the loss trajectories for different geometries explored during the optimization process.

    Args:
        loss_trajectories (list): List of loss trajectories, where each trajectory corresponds to a different geometry.
    """
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0, 1, len(loss_trajectories)))
    for i, trajectory in enumerate(loss_trajectories):
        plt.plot(trajectory, color=colors[i], alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Loss Trajectories for Different Geometries")
    plt.colorbar(plt.cm.ScalarMappable(cmap="viridis"), label="Optimization Iteration")

    # Save the figure with a descriptive name
    plt.savefig("figures/loss_trajectories.png")

    plt.show()
