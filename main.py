import torch
import numpy as np
from torchvision import datasets, transforms

from bayesian_optimization import bayesian_optimization
from autoencoder import GeometricAutoencoder
from train import train_and_evaluate
from graph_search_space import construct_graph_search_space

subsample_percent = .005

# Define the data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Calculate the number of samples for 5% of the training data
subsample_size = int(subsample_percent * len(train_dataset))

# Create a random subset of indices for the training data
subset_indices = torch.randperm(len(train_dataset))[:subsample_size]

# Use the subset indices to create the subsampled dataset
subsampled_train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)

# Create data loaders for the subsampled training data and test data
train_loader = torch.utils.data.DataLoader(subsampled_train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


n_p = 5
latent_dim = n_p*2
adjacency_matrix, signatures = construct_graph_search_space(n_p=n_p)
epochs = 10

# Define the objective function for Bayesian optimization
def objective_function(signature):
    model = GeometricAutoencoder(signature, latent_dim=latent_dim)
    train_losses, test_loss = train_and_evaluate(model, train_loader, test_loader, epochs=epochs)
    return train_losses, test_loss

# Perform Bayesian optimization
evaluated_signatures = []
evaluated_metrics = []
loss_trajectories = []

def callback(signature, metric):
    evaluated_signatures.append(signature)
    loss_trajectories.append(metric[0])  
    evaluated_metrics.append(metric[1])  

optimal_signature, optimal_val_metric, optimal_train_metric = bayesian_optimization(objective_function, signatures, adjacency_matrix, n_iterations=10, callback=callback)

print("Optimal signature:", optimal_signature)
print("Optimal validation metric:", optimal_val_metric)
