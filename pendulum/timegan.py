import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import glob
from math import pi as M_PI

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# === Model components ===

class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.activation(self.linear(out))
        return out

class Recovery(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, h):
        out, _ = self.lstm(h)
        out = self.linear(out)
        return out

class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_layers):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(z_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.LeakyReLU()

        self.init_h = nn.Linear(z_dim, hidden_dim * num_layers)
        self.init_c = nn.Linear(z_dim, hidden_dim * num_layers)

    def forward(self, z):
        batch_size = z.size(0)
        device = z.device

        # Use mean of z across time to initialize hidden states
        z_mean = z.mean(dim=1)  # (batch_size, z_dim)

        h0 = self.init_h(z_mean).view(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = self.init_c(z_mean).view(self.num_layers, batch_size, self.hidden_dim).to(device)

        # Add small noise to initial states here:
        h0 = h0 + 0.01 * torch.randn_like(h0)
        c0 = c0 + 0.01 * torch.randn_like(c0)

        out, _ = self.lstm(z, (h0, c0))
        out = self.activation(self.linear(out))
        return out



class Supervisor(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.LeakyReLU() #nn.Sigmoid()

    def forward(self, h):
        out, _ = self.lstm(h)
        out = self.activation(self.linear(out))
        return out

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, h):
        out, _ = self.lstm(h)
        out = self.linear(out)
        out = self.activation(out)
        return out

# === Losses ===

def mse_loss(x, y):
    return torch.mean((x - y) ** 2)

def bce_loss(y_pred, y_true):
    bce = nn.BCELoss()
    return bce(y_pred, y_true)

# === TimeGAN class ===

class TimeGAN:
    def __init__(self, feature_dim, hidden_dim=24, z_dim=32, num_layers = 2, seq_len=24):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.z_dim = z_dim
        self.seq_len = seq_len

        self.embedder = Embedder(feature_dim, hidden_dim, num_layers).to(device)
        self.recovery = Recovery(hidden_dim, feature_dim, num_layers).to(device)
        self.generator = Generator(z_dim, hidden_dim, num_layers).to(device)
        self.supervisor = Supervisor(hidden_dim, num_layers).to(device)
        self.discriminator = Discriminator(hidden_dim, num_layers).to(device)

        self.optim_embedder = optim.Adam(list(self.embedder.parameters()) + list(self.recovery.parameters()), lr=1e-3)
        self.optim_generator = optim.Adam(list(self.generator.parameters()) + list(self.supervisor.parameters()), lr=1e-4)
        self.optim_discriminator = optim.Adam(self.discriminator.parameters(), lr=1e-3)

    def train_embedder(self, data_loader, epochs=10, log_path='embedder_loss.csv'):
        self.embedder.train()
        self.recovery.train()
        logs = []
        for epoch in range(epochs):
            total_loss = 0
            for x in tqdm(data_loader, desc=f"Embedder Epoch {epoch+1}"):
                x = x[0].to(device)
                self.optim_embedder.zero_grad()
                h = self.embedder(x)
                x_tilde = self.recovery(h)
                loss = mse_loss(x, x_tilde)
                loss.backward()
                self.optim_embedder.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(data_loader)
            logs.append({'epoch': epoch+1, 'embedder_loss': avg_loss})
            print(f"Embedder Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")

        pd.DataFrame(logs).to_csv(log_path, index=False)

    def train_supervisor(self, data_loader, epochs=10, log_path='supervisor_loss.csv'):
        self.supervisor.train()
        logs = []
        for epoch in range(epochs):
            total_loss = 0
            for x in tqdm(data_loader, desc=f"Supervisor Epoch {epoch+1}"):
                x = x[0].to(device)
                self.optim_generator.zero_grad()
                h = self.embedder(x).detach()
                h_supervise = self.supervisor(h)
                loss = mse_loss(h[:,1:,:], h_supervise[:,:-1,:])
                loss.backward()
                self.optim_generator.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(data_loader)
            logs.append({'epoch': epoch+1, 'supervisor_loss': avg_loss})
            print(f"Supervisor Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")

        pd.DataFrame(logs).to_csv(log_path, index=False)

    def train_adversarial(self, data_loader, epochs=100, log_path='adversarial_loss.csv'):
        logs = []
        for epoch in range(epochs):
            total_d_loss = 0
            total_g_loss = 0
            for x in tqdm(data_loader, desc=f"Adversarial Epoch {epoch+1}"):
                x = x[0].to(device)
                batch_size = x.size(0)

                # Discriminator
                self.optim_discriminator.zero_grad()
                h = self.embedder(x).detach()
                y_real = self.discriminator(h)

                z = torch.randn(batch_size, self.seq_len, self.z_dim).to(device)
                e_hat = self.generator(z)
                h_hat = self.supervisor(e_hat)
                y_fake = self.discriminator(h_hat.detach())

                d_loss = -torch.mean(torch.log(y_real + 1e-8) + torch.log(1 - y_fake + 1e-8))
                d_loss.backward()
                self.optim_discriminator.step()

                # Generator
                self.optim_generator.zero_grad()
                e_hat = self.generator(z)
                h_hat = self.supervisor(e_hat)
                y_fake = self.discriminator(h_hat)

                g_loss_u = -torch.mean(torch.log(y_fake + 1e-8))
                g_loss_s = mse_loss(h_hat[:,1:,:], h_hat[:,:-1,:])
                x_tilde = self.recovery(h_hat)
                g_loss_v = mse_loss(x, x_tilde)

                g_loss = g_loss_u + 10 * g_loss_s + 10 * g_loss_v
                g_loss.backward()
                self.optim_generator.step()

                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()

            avg_d_loss = total_d_loss / len(data_loader)
            avg_g_loss = total_g_loss / len(data_loader)
            logs.append({'epoch': epoch+1, 'discriminator_loss': avg_d_loss, 'generator_loss': avg_g_loss})
            print(f"Epoch {epoch+1}/{epochs} D_loss: {avg_d_loss:.4f} G_loss: {avg_g_loss:.4f}")

        pd.DataFrame(logs).to_csv(log_path, index=False)


    def generate(self, n_samples):
        output_dir = "synthetic_data/timegan/unwrapped/"
        os.makedirs(output_dir, exist_ok=True)

        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()

        with torch.no_grad():
            for idx in range(0, n_samples):
                z = torch.rand(1, self.seq_len, self.z_dim).to(device) * 2 * M_PI
                e_hat = self.generator(z)
                h_hat = self.supervisor(e_hat)
                x_hat = self.recovery(h_hat)

                synthetic_data =  x_hat.cpu().numpy()

                num_samples, seq_len, num_features = synthetic_data.shape

                sample = synthetic_data[0]  # shape (seq_len, num_features)
                df = pd.DataFrame(sample, columns=[f"theta{j+1}" for j in range(num_features)])
                filepath = os.path.join(output_dir, f"synthetic_sample_{idx:03d}.csv")
                df.to_csv(filepath, index=False)




# === Function to load your data ===
def load_your_data():
    data_dir = "real_data/polar/"
    all_sequences = []

    for filepath in sorted(glob.glob(f"{data_dir}/polar_*.csv")):
        df = pd.read_csv(filepath)
        theta_1 = df['theta1'].values.astype(np.float32)
        theta_2 = df['theta2'].values.astype(np.float32)
        seq = np.stack([theta_1, theta_2], axis=-1)  # shape (seq_len, 2)
        #seq = np.mod(seq, 2 * np.pi)
        all_sequences.append(seq)

    N = 1000  # number of sequences you want to keep
    all_sequences = all_sequences[:N]

    data = np.array(all_sequences)

    return data


def load_model():
    feature_dim = data.shape[2]
    seq_len = data.shape[1]
    hidden_dim = 32
    z_dim = 32
    num_layers = 4

    model = TimeGAN(feature_dim=feature_dim, hidden_dim = hidden_dim, z_dim = z_dim, num_layers = num_layers, seq_len=seq_len)

    # Load weights
    model.load_state_dict(torch.load("saved_models/timegan_model_unwrapped.pth"))
    model.eval()

    with torch.no_grad():
        synthetic_data = model.generate(num_samples=100)


if __name__ == "__main__":
    # Load your data
    data = load_your_data()
    dataset = TensorDataset(torch.tensor(data))
    batch_size = 16
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    feature_dim = data.shape[2]
    seq_len = data.shape[1]

    hidden_dim = 32
    z_dim = 32
    num_layers = 4

    model = TimeGAN(feature_dim=feature_dim, hidden_dim = hidden_dim, z_dim = z_dim, num_layers = num_layers, seq_len=seq_len)

    print("Training embedder...")
    model.train_embedder(loader, epochs=50, log_path='timegan_loss/embedder_loss_unwrapped.csv')

    print("Training supervisor...")
    model.train_supervisor(loader, epochs=50, log_path='timegan_loss/supervisor_loss_unwrapped.csv')

    print("Training adversarial...")
    model.train_adversarial(loader, epochs=50, log_path='timegan_loss/adversarial_loss_unwrapped.csv')

    z1 = torch.rand(1, model.seq_len, model.z_dim, device=device) * 2 * M_PI
    z2 = torch.rand(1, model.seq_len, model.z_dim, device=device) * 2 * M_PI

    x1 = model.recovery(model.supervisor(model.generator(z1)))
    x2 = model.recovery(model.supervisor(model.generator(z2)))

    print("First step difference:", torch.sum(torch.abs(x1[:,0,:] - x2[:,0,:])).item())


    print("Generating synthetic data...")
    synthetic_data = model.generate(1000) 



