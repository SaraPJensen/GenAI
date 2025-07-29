from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
import pandas as pd
import os


def make_gaussian_data(percentage):

    real_data = pd.read_csv('../datasets/real_wine_data.csv')
    metadata = Metadata.load_from_json('../datasets/wine_metadata.json')

    os.makedirs(f'../datasets/{percentage}_train', exist_ok = True)
    os.makedirs(f'saved_models/{percentage}_train', exist_ok = True)

    num_elements = int((percentage/100)*len(real_data)) 
    training_subset = real_data.head(num_elements)

    # Step 1: Create the synthesizer
    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=False
        )

    # Train the synthesizer
    synthesizer.fit(training_subset)

    # Save model
    synthesizer.save(f'saved_models/{percentage}_train/gaussian_{percentage}_wine.pkl')

    # Generate synthetic data
    synthetic_data = synthesizer.sample(num_rows=1000)
    synthetic_data.to_csv(f'../datasets/{percentage}_train/gaussian_{percentage}_wine_data.csv', index = False)

