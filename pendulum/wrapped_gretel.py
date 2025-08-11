import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as md
import glob
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType


# ===== 1. Load your CSV time series data =====
data_dir = "real_data/polar/"
all_sequences = []

for filepath in sorted(glob.glob(f"{data_dir}/polar_*.csv")):
    df = pd.read_csv(filepath)
    theta_1 = df['theta1'][:6000].values  #skip last timestep to make it more easily divisible
    theta_2 = df['theta2'][:6000].values
    seq = np.stack([theta_1, theta_2], axis=-1)  # shape (seq_len, 2)
    all_sequences.append(seq)

N = 1000  # number of sequences you want to keep

all_sequences = all_sequences[:N]
data = np.array(all_sequences)  # shape (num_sequences, seq_len, num_features)

wrapped_data = np.mod(data, 2 * np.pi)  #Make the data within range [0, 2pi]

num_seq, seq_len, num_features = wrapped_data.shape


config = DGANConfig(
    max_sequence_len=seq_len,
    sample_len=30,
    batch_size=100,
    apply_feature_scaling=True,
    apply_example_scaling = True,
    generator_learning_rate=1e-4,
    discriminator_learning_rate=1e-4,
    epochs=30,
)

model = DGAN(config)

model.train_numpy(features = wrapped_data)

model.save('saved_models/dgan_wrapped.pt')

print('Model has been trained and saved')

num_samples = 10

# Generate synthetic data
_, synthetic_scaled = model.generate_numpy(num_samples)


for idx, sample in enumerate(synthetic_scaled): 
    df = pd.DataFrame(sample, columns=['theta1','theta2'])
    filepath = f'synthetic_data/dgan/wrapped/wrapped_synthetic_sample_{idx:03d}.csv'
    df.to_csv(filepath, index=False)



for idx, sample in enumerate(synthetic_scaled): 
    unwrapped = np.unwrap(sample)
    df = pd.DataFrame(unwrapped, columns=['theta1','theta2'])
    filepath = f'synthetic_data/dgan/wrapped/unwrapped/unwrapped_synthetic_sample_{idx:03d}.csv'
    df.to_csv(filepath, index=False)



