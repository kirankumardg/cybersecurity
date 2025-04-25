
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from generator import Generator
from discriminator import Discriminator


column_names = [f"feature_{i}" for i in range(41)] + ["label"]
df = pd.read_csv("KDDTrain+.txt", names=column_names)


df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
X = df.drop(['label'], axis=1)
y = df['label']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
data_dim = X_scaled.shape[1]
batch_size = 64

G = Generator(latent_dim, data_dim).to(device)
D = Discriminator(data_dim).to(device)
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)


for epoch in range(1000):
    idx = np.random.randint(0, X_tensor.size(0), batch_size)
    real_samples = X_tensor[idx]
    real_labels = torch.ones((batch_size, 1)).to(device)
    fake_labels = torch.zeros((batch_size, 1)).to(device)

    z = torch.randn(batch_size, latent_dim).to(device)
    fake_samples = G(z)

    D_loss = criterion(D(real_samples), real_labels) + criterion(D(fake_samples.detach()), fake_labels)
    optimizer_D.zero_grad()
    D_loss.backward()
    optimizer_D.step()

    z = torch.randn(batch_size, latent_dim).to(device)
    G_loss = criterion(D(G(z)), real_labels)
    optimizer_G.zero_grad()
    G_loss.backward()
    optimizer_G.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {D_loss.item():.4f}, G Loss: {G_loss.item():.4f}")

# Generate synthetic data
z = torch.randn(1000, latent_dim).to(device)
synthetic_data = G(z).detach().cpu().numpy()
synthetic_data_original = scaler.inverse_transform(synthetic_data)
pd.DataFrame(synthetic_data_original).to_csv("synthetic_data.csv", index=False)
