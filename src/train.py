import torch
from torch import nn
from tqdm import tqdm


def train_and_evaluate(
    model,
    train_loader,
    test_loader,
    epochs=10,
    device=torch.device("cpu"),
    progress_bar=True,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []

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
