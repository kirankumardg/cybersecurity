
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, data_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)
