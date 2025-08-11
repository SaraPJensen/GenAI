from os import path
import os
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
# from sklearn.preprocessing import MinMaxScaler
# from ydata_synthetic.synthesizers.timeseries import TimeGAN
# from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '3' for more silence
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
exit()

# ===== 1. Load your CSV time series data =====
data_dir = "real_data/polar/"
all_sequences = []

for filepath in sorted(glob.glob(f"{data_dir}/polar_*.csv")):
    df = pd.read_csv(filepath)
    theta_1 = df['theta1'].values
    theta_2 = df['theta2'].values
    seq = np.stack([theta_1, theta_2], axis=-1)  # shape (seq_len, 2)
    all_sequences.append(seq)

data = np.array(all_sequences)  # shape (num_sequences, seq_len, 2)
num_seq, seq_len, num_features = data.shape


# ===== 2. Normalize =====
data_reshaped = data.reshape(-1, num_features)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_reshaped).reshape(num_seq, seq_len, num_features)

# ===== 3. Setup TimeGAN parameters =====
model_params = ModelParameters(
    batch_size=32,
    lr=5e-4,
    noise_dim=32,
    layers_dim=128,
    latent_dim=24,
)

train_params = TrainParameters(
    epochs=5
)

# ===== 4. Train with tqdm progress bar =====
# class TqdmCallback:
#     def __init__(self, epochs):
#         self.epochs = epochs
#         self.pbar = tqdm(total=epochs, desc="Training TimeGAN")

#     def __call__(self, epoch, logs=None):
#         self.pbar.update(1)
#         if epoch == self.epochs - 1:
#             self.pbar.close()

synthesizer = TimeGAN(
    hidden_dim=24,
    seq_len=seq_len,
    n_seq=num_features,
    gamma=1.0,
    model_parameters = model_params)

#tqdm_cb = TqdmCallback(train_params.epochs)
synthesizer.train(data_scaled, train_params.epochs)


# ===== 5. Save the trained model =====
model_path = "timegan_pendulum_model"
synthesizer.save(model_path)
print(f"Model saved to directory '{model_path}'")

# ===== 6. Generate synthetic samples and save as CSV =====
n_samples = 100
synthetic_data = synthesizer.sample(n_samples)

synthetic_original = scaler.inverse_transform(
    synthetic_data.reshape(-1, num_features)
).reshape(n_samples, seq_len, num_features)

output_dir = "synthetic_data"
os.makedirs(output_dir, exist_ok=True)

for i in range(n_samples):
    df_synth = pd.DataFrame(synthetic_original[i], columns=["theta1", "theta2"])
    outpath = os.path.join(output_dir, f"synthetic_pendulum_{i:03d}.csv")
    df_synth.to_csv(outpath, index=False)

print(f"{n_samples} synthetic sequences saved to '{output_dir}/'")