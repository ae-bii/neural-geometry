# -*- coding: utf-8 -*-
"""
A basic NN implementation
"""

# from google.colab import drive
# drive.mount('/content/drive')

# Importing PyTorch libraries + Matplotlib + NumPy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

# Checking if a GPU is available
# Otherwise, we just use the CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Loading MNIST data into into training, validation, and testing datasets
mnist_train = datasets.MNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

# We split 5000/60000 of the data points from training to validation
# We end up with 55000 training, 5000, validation, and 10000 testing points
mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [55000, 5000])

mnist_test = datasets.MNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

# General functions for training, testing, and plotting graphs for all models!


def train(model, name, epochs, optimizer, train_dataloader, val_dataloader=None):
    """Training Function
    Trains a provided PyTorch model for classification over given dataset.
    By default uses Cross Entropy Loss.
    """
    # Use cross entropy loss for all models for multiclass classification!
    loss_fn = nn.CrossEntropyLoss()

    # Initializing arrays for tracking training loss, validation loss, validation accuracy
    train_loss, val_loss, val_acc = [], [], []
    best_val_loss = float("inf")

    # Iterate over all epochs!
    for e in range(epochs):
        # set model to training mode, enables dropout and adjusting gradient
        model.train()
        epoch_loss = 0

        # Iterating over all minibatches in data (randomly shuffled each epoch)
        for x, y in train_dataloader:
            # Predict output of minibatch
            pred = model(x.to(device))
            # Calculate softmax loss of output
            loss = loss_fn(pred, y.to(device))
            # Update epoch loss by batched softmax loss
            epoch_loss += loss.item() * x.size()[0]

            # backpropagate and update gradient
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Keep track of training loss
        train_loss.append(epoch_loss / len(train_dataloader.dataset))

        # Calculating validation loss at end of epoch
        if val_dataloader is not None:
            # Disable gradient modification
            with torch.no_grad():
                # Set model in evaluation mode (disables things like dropout)
                model.eval()
                loss, correct = 0, 0
                for x, y in val_dataloader:
                    # Get predictions on validation minibatches and calculate loss + accuracy
                    pred = model(x.to(device))
                    loss += loss_fn(pred, y.to(device)).item() * x.size()[0]
                    correct += (
                        (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()
                    )
                val_loss.append(loss / len(val_dataloader.dataset))
                val_acc.append(correct / len(val_dataloader.dataset))

                # If the validation loss is better than previously, save this model!
                if val_loss[-1] < best_val_loss:
                    best_val_loss = val_loss[-1]
                    torch.save(
                        {
                            "epoch": e + 1,
                            "model_state_dict": model.state_dict(),  # weights of model
                            "optimizer_state_dict": optimizer.state_dict(),  # optimizer params
                            "loss": loss_fn,
                        },
                        "/content/drive/MyDrive/Colab Notebooks/dlforcv/models/" + name,
                    )

        # print out training and validation loss every 1/10th of the epochs!
        if e == 0 or (e + 1) % (epochs / 10) == 0:
            print(
                f"Epoch {e + 1}/{epochs} => Train Loss: {train_loss[-1]}, Val. Loss: {val_loss[-1]}"
            )

    # Save final model after all training
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_fn,
        },
        "/content/drive/MyDrive/Colab Notebooks/dlforcv/models/final_" + name,
    )

    return (train_loss, val_loss, val_acc)


def test(model, dataloader):
    """Test a given model on a training dataset."""
    # Set model to evaluation mode, disabling things like dropout
    model.eval()
    # test_loss = 0
    correct = 0

    # Disable gradient updating
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x.to(device))
            correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()

    correct /= len(dataloader.dataset)
    print(f"Test Accuracy: {correct}")


def show_digits(model, dataloader):
    """Shows 10 random digits from a given dataloader and the model's predictions."""
    # Set model to evaluation mode
    model.eval()

    # disable gradient updating
    with torch.no_grad():
        # get one batch from dataloader and predict
        x, y = next(iter(dataloader))
        pred = model(x.to(device))

        # shuffle indices randomly and display img/predictions for first 10
        idxs = np.arange(x.size()[0])
        np.random.shuffle(idxs)

        for i in range(10):
            # fig = plt.figure(figsize=(2, 2))
            print(f"Label: {y[idxs[i]]}")
            print(f"Pred. Label: {pred[idxs[i]].argmax()}")
            print(f"Preds: {pred[idxs[i]]}")
            plt.imshow(x[idxs[i]].squeeze())
            plt.show()


def plot_losses(train_loss, val_loss, val_acc):
    """Plots the training and validation losses, as well as validation accuracy."""
    fig, axes = plt.subplots(ncols=2, figsize=[15, 6])

    x = np.arange(len(train_loss))
    axes[0].scatter(x, train_loss, label="Train Loss", s=1)
    axes[0].scatter(x, val_loss, label="Validation Loss", s=1)
    axes[0].legend()
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Losses")

    axes[1].scatter(x, val_acc, s=1)
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Validation Accuracy")

    fig.show()


"""# Softmax Regression!"""


# Softmax regression PyTorch implementation
class SoftmaxRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # first we flatten image into a 784-dimensional vector
        self.flatten = nn.Flatten()
        # then we use a fully connected linear layer to simulate w^Tx+b
        # note that nn.Linear by default includes a bias term!
        self.w = nn.Linear(784, 10)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.w(x)
        return logits


# load mnist training and validation datasets into dataloaders that automatically shuffle
# set batch size to 100 for more efficient computation (minibatch SGD)
train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=100, shuffle=True)

# initializing softmax regression class and send to either CPU or GPU
softmax = SoftmaxRegressor().to(device)
# use SGD with learning rate of 0.5
optimizer = torch.optim.SGD(softmax.parameters(), lr=0.5)
# train model and keep track of training loss, validation loss, and val. accuracy
reg_outputs = train(softmax, "softmax_reg.pt", 50, optimizer, train_loader, val_loader)

plot_losses(*reg_outputs)

# setting up testing dataloader with batch size 100 and testing accuracy
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=True)
test(softmax, test_loader)

# displaying digits, actual labels, and predictions
show_digits(softmax, test_loader)

# initializing a new softmax regression model
softmax_loaded = SoftmaxRegressor().to(device)

# load model from checkpoint file with lowest validation accuracy
reg_checkpoint = torch.load(
    "/content/drive/MyDrive/Colab Notebooks/dlforcv/models/softmax_reg.pt"
)
# load the weights into the model
softmax_loaded.load_state_dict(reg_checkpoint["model_state_dict"])
epoch = reg_checkpoint["epoch"]

print(f"Best Epoch: {epoch}")
# test this new model on the testing dataset
test(softmax_loaded, test_loader)

"""# Softmax MLP!"""


# functor to initialize all weights in linear and convolutional layers
# use normal distribution to initialize weights with mean 0 and std dev. of 0.1
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight, std=0.1)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, std=0.1)


class SoftmaxMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # first, flatten image into 28*28=784 dimensional flat input
        self.flatten = nn.Flatten()
        # then, pass through one fully connected hidden layer with ReLU activation
        # then, pass again to fully connected output layer
        self.linear = nn.Sequential(
            nn.Linear(28 * 28, 512), nn.ReLU(), nn.Linear(512, 10)
        )
        # initializing weights normally
        self.linear.apply(init_weights)

    def forward(self, x):
        # This function simply passes the model through the layers and returns
        # a prediction of the output
        x = self.flatten(x)
        logits = self.linear(x)
        return logits


# setting up dataloaders with mnist training/validation data and using size 50 minibatches
train_loader = DataLoader(mnist_train, batch_size=50, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=50, shuffle=True)

# setting up softmax multilayer perceptron and sending to either CPU or GPU
# we use an adam optimizer (essentially extended stochastic gradient descent)
mlp = SoftmaxMLP().to(device)
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

# train model for 50 epochs and save into mlp.pt or final_mlp.pt file names
mlp_outputs = train(mlp, "mlp.pt", 50, optimizer, train_loader, val_loader)

plot_losses(*mlp_outputs)

# then, we test our model on the training dataset
test_loader = DataLoader(mnist_test, batch_size=50, shuffle=True)
test(mlp, test_loader)

# looking at a few randomly chosen digits from the training dataset and predictions...
show_digits(mlp, test_loader)

# testing our best model that was saved during the training (min. validation loss)
mlp_loaded = SoftmaxMLP().to(device)

# load model checkpoint and update weights from the saved state dictionary
mlp_checkpoint = torch.load(
    "/content/drive/MyDrive/Colab Notebooks/dlforcv/models/mlp.pt"
)
mlp_loaded.load_state_dict(mlp_checkpoint["model_state_dict"])
epoch = mlp_checkpoint["epoch"]

# testing our model
print(f"Best Epoch: {epoch}")
test(mlp_loaded, test_loader)

"""# LeNet!"""

# an implementation of LeNet, a basic CNN model


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Defining first convolutional layer
        # 5x5 kernel with 32 output channels
        # we use padding 2 to make sure the image stays the same size and stride 1
        # use ReLU for nonlinear activation and then maxpool with 2x2 size (stride 2)
        # max pooling halves image in both dimensions so images are 14x14
        self.l1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Second convolutional layer also uses a 5x5 kernel, stride 1 and padding 2
        # we use relu again and maxpooling with 2x2 kernel which halves both dimensions of the image again
        # we now have 64 channels with 7x7 images
        self.l2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Use fully connected layer that converts 64 channels of 7x7 images to 1024 neurons
        # use relu for nonlinear activation and implement dropout with probability 0.1
        # to select potentially better neurons to activate
        self.l3 = nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024), nn.ReLU(), nn.Dropout(p=0.1)
        )

        # final fully connected layer connects 1024 neurons to outputs
        self.l4 = nn.Linear(1024, 10)

        # initialize all weights in convolutional and linear layers normally
        # with mean 0 and std deviation of 0.1
        self.l1.apply(init_weights)
        self.l2.apply(init_weights)
        self.l3.apply(init_weights)
        self.l4.apply(init_weights)

    # forward propagation of inputs
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        # take output of convolutional layers and flatten for fully connected layers
        x = x.view(x.shape[0], -1)
        x = self.l3(x)
        logits = self.l4(x)
        return logits


# load training and validation data into dataloaders with minibatch size 50
train_loader = DataLoader(mnist_train, batch_size=50, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=50, shuffle=True)

# initialize lenet model to CPU or GPU
lenet = LeNet().to(device)
# use adam optimizer, built on top of normal stochastic gradient descent
optimizer = torch.optim.Adam(lenet.parameters(), lr=1e-4)

# train lenet model and save to lenet.pt or final_lenet.pt
lenet_outputs = train(lenet, "lenet.pt", 50, optimizer, train_loader, val_loader)

plot_losses(*lenet_outputs)

# testing lenet model
test_loader = DataLoader(mnist_test, batch_size=50, shuffle=True)
test(lenet, test_loader)

show_digits(lenet, test_loader)

# loading best saved lenet model
lenet_loaded = LeNet().to(device)

lenet_checkpoint = torch.load(
    "/content/drive/MyDrive/Colab Notebooks/dlforcv/models/lenet.pt"
)
lenet_loaded.load_state_dict(lenet_checkpoint["model_state_dict"])
epoch = lenet_checkpoint["epoch"]

# testing saved model
print(f"Best Epoch: {epoch}")
test(lenet_loaded, test_loader)
