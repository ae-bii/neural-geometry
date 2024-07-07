from torch import nn
from nlgm.manifolds import ProductManifold


# ==========================================================================================
# ============================= FOR USE IN `mnist_experiment.ipynb`=========================
# ==========================================================================================


class Encoder(nn.Module):
    def __init__(self, hidden_dim=20, latent_dim=2):
        """
        Encoder class for the geometric autoencoder.

        Args:
            hidden_dim (int): Number of hidden dimensions.
            latent_dim (int): Number of latent dimensions.
        """
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
            nn.Flatten(),
        )

    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded output tensor.
        """
        z = self.encoder(x)
        return z


class Decoder(nn.Module):
    def __init__(self, hidden_dim=20, latent_dim=2):
        """
        Decoder class for the geometric autoencoder.

        Args:
            hidden_dim (int): Number of hidden dimensions.
            latent_dim (int): Number of latent dimensions.
        """
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (hidden_dim, 7, 7)),
            nn.ConvTranspose2d(
                hidden_dim, hidden_dim, 3, stride=2, padding=1, output_padding=1
            ),  # Upsample to 14x14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_dim),
            nn.ConvTranspose2d(
                hidden_dim, hidden_dim, 3, stride=2, padding=1, output_padding=1
            ),  # Upsample to 28x28
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),  # Reduce to 1 channel
            nn.Sigmoid(),  # Output in range [0, 1]
        )

    def forward(self, z):
        """
        Forward pass of the decoder.

        Args:
            z (torch.Tensor): Encoded input tensor.

        Returns:
            torch.Tensor: Decoded output tensor.
        """
        x_recon = self.decoder(z)
        return x_recon


class GeometricAutoencoder(nn.Module):
    def __init__(self, signature, hidden_dim=20, latent_dim=2):
        """
        Geometric Autoencoder class.

        Args:
            signature (list): List of signature dimensions.
            hidden_dim (int): Number of hidden dimensions.
            latent_dim (int): Number of latent dimensions.
        """
        super(GeometricAutoencoder, self).__init__()
        self.geometry = ProductManifold(signature)
        self.encoder = Encoder(hidden_dim, latent_dim)
        self.decoder = Decoder(hidden_dim, latent_dim)

    def forward(self, x):
        """
        Forward pass of the geometric autoencoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Decoded output tensor.
        """
        z = self.encoder(x)
        z = self.geometry.exponential_map(z)
        x_recon = self.decoder(z)
        return x_recon
