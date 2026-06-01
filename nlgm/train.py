import torch
from torch import nn
from tqdm.auto import tqdm


def train_and_evaluate(
    model,
    train_loader,
    test_loader,
    epochs=10,
    device=torch.device("cpu"),
    progress_bar=True,
):
    """
    Train a model and evaluate on a test loader each epoch.

    Parameters
    ----------
    model : nn.Module
        Model to train and evaluate.
    train_loader : torch.utils.data.DataLoader
        Training data loader.
    test_loader : torch.utils.data.DataLoader
        Test data loader.
    epochs : int, default=10
        Number of training epochs.
    device : torch.device, default=torch.device("cpu")
        Device used for training and evaluation.
    progress_bar : bool, default=True
        Whether to show epoch-level progress.

    Returns
    -------
    list[float]
        Per-epoch training losses.
    float
        Final epoch test loss.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []

    model.to(device)

    it = tqdm(range(epochs), desc="Epochs") if progress_bar else range(epochs)
    for epoch in it:
        model.train()
        train_loss = 0
        for images, _ in tqdm(train_loader, desc="Training Loop"):
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed_images = model(images)
            loss = criterion(reconstructed_images, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                reconstructed_images = model(images)
                loss = criterion(reconstructed_images, images)
                test_loss += loss.item()
        test_loss /= len(test_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

    return train_losses, test_loss
