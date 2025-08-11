import torch
import torch.nn as nn
import pandas as pd
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== PARAMETERS ====
T_in = 200    # input length (initial condition steps)
T_out = 500   # output length (predicted steps)
num_samples = 1000  # number of CSV files
data_dir = "real_data/polar"

# ==== LOAD DATA ====
all_sequences = []
for s in range(num_samples):
    filepath = f"{data_dir}/polar_{s:03d}.csv"
    df = pd.read_csv(filepath)

    theta_1 = df['theta1'].values
    theta_2 = df['theta2'].values

    seq = np.stack([theta_1, theta_2], axis=-1)  # shape: (T_total, 2)
    all_sequences.append(seq)

all_sequences = np.array(all_sequences)  # shape: (num_sequences, T_total, 2)

# ==== NORMALIZE ====
scaler = StandardScaler()
num_seq, T_total, feats = all_sequences.shape
all_sequences = scaler.fit_transform(all_sequences.reshape(-1, feats)).reshape(num_seq, T_total, feats)

# ==== CREATE DATASET ====
class PendulumDataset(Dataset):
    def __init__(self, sequences, T_in, T_out):
        self.X = []
        self.Y = []
        for seq in sequences:
            if len(seq) >= T_in + T_out:
                self.X.append(seq[:T_in])   # first part as input
                self.Y.append(seq[T_in:T_in+T_out])  # next part as target
        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = PendulumDataset(all_sequences, T_in, T_out)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



# ========== Encoder ==========
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        output, (h, c) = self.lstm(x)
        return h, c  # hidden and cell state for decoder


# ========== Decoder ==========
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h, c):
        output, (h, c) = self.lstm(x, (h, c))
        pred = self.fc(output)
        return pred, h, c


# ========== Seq2Seq Wrapper ==========
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, input_dim, output_dim, T_out):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.T_out = T_out
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, src):
        batch_size = src.size(0)
        h, c = self.encoder(src)

        # Initial input to decoder (e.g., last step of encoder or zero)
        decoder_input = src[:, -1:, :]  # shape: [B, 1, input_dim]

        outputs = []
        for _ in range(self.T_out):
            out, h, c = self.decoder(decoder_input, h, c)
            outputs.append(out)
            decoder_input = out  # feeding back the output

        return torch.cat(outputs, dim=1)  # [B, T_out, output_dim]


# ==== TRAIN / TEST SPLIT ====
dataset = PendulumDataset(all_sequences, T_in, T_out)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==== MODEL ====
input_dim = 2
hidden_dim = 128
output_dim = 2
num_layers = 3

encoder = Encoder(input_dim, hidden_dim, num_layers)
decoder = Decoder(input_dim, hidden_dim, output_dim, num_layers)
model = Seq2Seq(encoder, decoder, input_dim, output_dim, T_out).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==== CSV LOGGING ====
log_file = "training_log_deep.csv"
with open(log_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "TrainLoss", "TestLoss"])  # header

# ==== TRAIN LOOP ====
n_epochs = 50
for epoch in range(1, n_epochs+1):
    # Training
    model.train()
    train_loss = 0.0
    for X_batch, Y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs} [Train]", leave=False):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, Y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)

    # Testing
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in tqdm(test_loader, desc=f"Epoch {epoch}/{n_epochs} [Test]", leave=False):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            preds = model(X_batch)
            loss = criterion(preds, Y_batch)
            test_loss += loss.item() * X_batch.size(0)
    test_loss /= len(test_loader.dataset)

    # Print and log
    print(f"Epoch {epoch}/{n_epochs} - Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, test_loss])


torch.save(model.state_dict(), "seq2seq_pendulum_deep.pth")
print("Model saved to seq2seq_pendulum_deep.pth")


'''
model.eval()
with torch.no_grad():
    init_seq = dataset[0][0].unsqueeze(0)  # shape: [1, T_in, 2]
    pred_future = model(init_seq)  # [1, T_out, 2]
'''