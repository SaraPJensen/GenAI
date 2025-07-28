from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import Metadata
import pandas as pd
import os


def make_copula_data(percentage):

    real_data = pd.read_csv('../datasets/real_cancer_data.csv')
    metadata = Metadata.load_from_json('../datasets/cancer_metadata.json')

    os.makedirs(f'../datasets/{percentage}_train', exist_ok = True)
    os.makedirs(f'loss_progress/{percentage}_train', exist_ok = True)
    os.makedirs(f'saved_models/{percentage}_train', exist_ok = True)

    num_elements = int((percentage/100)*len(real_data)) 
    training_subset = real_data.head(num_elements)

    print('Size of training set:', len(training_subset))


    # Step 1: Create the synthesizer
    synthesizer = CopulaGANSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=False,
        default_distribution='norm',
        epochs = 1000, 
        verbose = True
        )

    # Train the synthesizer
    synthesizer.fit(training_subset)

    # Get and save progress
    progress = synthesizer.get_loss_values()
    progress.to_csv(f'loss_progress/{percentage}_train/copula_{percentage}_cancer_loss.csv', index = False)

    # Save model
    synthesizer.save(f'saved_models/{percentage}_train/copula_{percentage}_cancer.pkl')

    # Generate synthetic data
    synthetic_data = synthesizer.sample(num_rows=1000)

    synthetic_data.to_csv(f'../datasets/{percentage}_train/copula_{percentage}_cancer_data.csv', index = False)

