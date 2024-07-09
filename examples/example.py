import torch
from torchvision import datasets, transforms
import random
from nlgm.optimizers import RandomWalkOptimizer
from nlgm.autoencoder import GeometricAutoencoder
from nlgm.train import train_and_evaluate
from nlgm.searchspace import construct_graph_search_space

# Define the data transforms
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# Load the MNIST dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

subsample_percent = 0.005

# Calculate the number of samples for 5% of the training data
subsample_size = int(subsample_percent * len(train_dataset))

# Create a random subset of indices for the training data
subset_indices = torch.randperm(len(train_dataset))[:subsample_size]

# Use the subset indices to create the subsampled dataset
subsampled_train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)

# Create data loaders for the subsampled training data and test data
train_loader = torch.utils.data.DataLoader(
    subsampled_train_dataset, batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

n_p = 5

# Construct the adjacency matrix and signatures for the graph search space
adjacency_matrix, signatures = construct_graph_search_space(n_p=n_p)

epochs = 10


# Define the objective function
def objective_function(signature):
    latent_dim = len(signature) * 2
    model = GeometricAutoencoder(signature, latent_dim=latent_dim)
    train_losses, test_loss = train_and_evaluate(
        model, train_loader, test_loader, epochs=epochs, device=torch.device("cuda")
    )
    return train_losses, test_loss


# Perform optimization
evaluated_signatures = []
evaluated_metrics = []
loss_trajectories = []


def callback(signature, metric):
    evaluated_signatures.append(signature)
    loss_trajectories.append(metric[0])
    evaluated_metrics.append(metric[1])


# Create an instance of the RandomWalkOptimizer
optimizer = RandomWalkOptimizer(
    adjacency_matrix, signatures, random.randint(0, len(signatures) - 1), None
)

# Optimize the objective function
optimal_signature, optimal_val_metric, optimal_train_metric = (
    optimizer.optimize_with_backtracking(objective_function, 10, callback)
)

# Print the optimal signature and validation metric
print("Optimal signature:", optimal_signature)
print("Optimal validation metric:", optimal_val_metric)
