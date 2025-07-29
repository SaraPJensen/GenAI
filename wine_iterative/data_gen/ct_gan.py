from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
import pandas as pd
import os


def make_ctgan_data(percentage):

    real_data = pd.read_csv('../datasets/real_wine_data.csv')
    metadata = Metadata.load_from_json('../datasets/wine_metadata.json')

    os.makedirs(f'../datasets/{percentage}_train', exist_ok = True)
    os.makedirs(f'loss_progress/{percentage}_train', exist_ok = True)
    os.makedirs(f'saved_models/{percentage}_train', exist_ok = True)

    num_elements = int((percentage/100)*len(real_data)) 
    training_subset = real_data.head(num_elements)

    # Step 1: Create the synthesizer
    synthesizer = CTGANSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=False,
        epochs = 1000, 
        verbose = True
        )

    # Train the synthesizer
    synthesizer.fit(training_subset)

    # Get and save progress
    progress = synthesizer.get_loss_values()
    progress.to_csv(f'loss_progress/{percentage}_train/ctgan_{percentage}_wine_loss.csv', index = False)

    # Save model
    synthesizer.save(f'saved_models/{percentage}_train/ctgan_{percentage}_wine.pkl')

    # Generate synthetic data
    synthetic_data = synthesizer.sample(num_rows=1000)

    synthetic_data.to_csv(f'../datasets/{percentage}_train/ctgan_{percentage}_wine_data.csv', index = False)

