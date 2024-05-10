import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class ImageAutoEncoder(nn.Module):
    """
    Image auto-encoder architecture as described in Appendix section D.2 Table 4 of
    https://arxiv.org/pdf/2309.04810.

    TODO: fix the dimensions based on the dataset
    """

    def __init__(self):
        super().__init__()
        encoder = [nn.Conv2d(1, 20, 3), nn.LazyBatchNorm2d(), nn.SiLU()]

        for _ in range(10):
            encoder.extend([nn.Conv2d(20, 20, 3), nn.LazyBatchNorm2d(), nn.SiLU()])

        encoder.extend(
            [
                nn.Conv2d(20, 2, 3),
                nn.LazyBatchNorm2d(),
                nn.SiLU(),
                nn.Flatten(),
                nn.LazyLinear(6 * 6), # TODO: figure out the dims here?
                nn.SiLU(),
            ]
        )

        self.encoder = nn.Sequential(*encoder)

        decoder = [
            nn.LazyLinear(),
            nn.SiLU(),
            nn.Unflatten(2, (6, 6)), # TODO: figure out the dims here?
            nn.ConvTranspose2d(2, 20, 3),
            nn.LazyBatchNorm2d(),
            nn.SiLU(),
        ]

        for _ in range(10):
            decoder.extend([nn.Conv2d(20, 20, 3), nn.LazyBatchNorm2d(), nn.SiLU()])

        decoder.extend([nn.ConvTranspose2d(20, 1, 3), nn.LazyBatchNorm2d(), nn.Sigmoid()])

        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        return decoding


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
                # if val_loss[-1] < best_val_loss:
                #     best_val_loss = val_loss[-1]
                #     torch.save(
                #         {
                #             "epoch": e + 1,
                #             "model_state_dict": model.state_dict(),  # weights of model
                #             "optimizer_state_dict": optimizer.state_dict(),  # optimizer params
                #             "loss": loss_fn,
                #         },
                #         "/content/drive/MyDrive/Colab Notebooks/dlforcv/models/" + name,
                #     )

        # print out training and validation loss every 1/10th of the epochs!
        if e == 0 or (e + 1) % (epochs / 10) == 0:
            print(
                f"Epoch {e + 1}/{epochs} => Train Loss: {train_loss[-1]}, Val. Loss: {val_loss[-1]}"
            )

    # Save final model after all training
    # torch.save(
    #     {
    #         "epoch": epochs,
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "loss": loss_fn,
    #     },
    #     "./trained_" + name,
    # )

    return (train_loss, val_loss, val_acc)


def test(model, dataloader):
    """Test a given model on a training dataset."""
    # Set model to evaluation mode, disabling things like dropout
    model.eval()
    test_loss, correct = 0, 0

    # Disable gradient updating
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x.to(device))
            correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()

    correct /= len(dataloader.dataset)
    print(f"Test Accuracy: {correct}")



# ==========================================================================================
# ============================= FOR USE IN `mnist_experiment.ipynb`=========================
# ==========================================================================================


import torch
import torch.nn as nn
from manifolds import ProductManifold

class Encoder(nn.Module):
    def __init__(self, hidden_dim=20, latent_dim=2):  
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, latent_dim, 3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        z = self.encoder(x)
        return z

class Decoder(nn.Module):
    def __init__(self, hidden_dim=20, latent_dim=2):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (hidden_dim, 7, 7)),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 3, stride=2, padding=1, output_padding=1),  # Upsample to 14x14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_dim),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 3, stride=2, padding=1, output_padding=1),  # Upsample to 28x28
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),  # Reduce to 1 channel
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward(self, z):
        x_recon = self.decoder(z)
        return x_recon

class GeometricAutoencoder(nn.Module):
    def __init__(self, signature, hidden_dim=20, latent_dim=2):
        super(GeometricAutoencoder, self).__init__()
        self.geometry = ProductManifold(signature)
        self.encoder = Encoder(hidden_dim, latent_dim)
        self.decoder = Decoder(hidden_dim, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        z = self.geometry.exponential_map(z)
        x_recon = self.decoder(z)
        return x_recon