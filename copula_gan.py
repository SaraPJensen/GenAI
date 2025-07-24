from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import Metadata
import pandas as pd

real_data = pd.read_csv('datasets/real_cancer_data.csv')
metadata = Metadata.load_from_json('datasets/cancer_metadata.json')

# Step 1: Create the synthesizer
synthesizer = CopulaGANSynthesizer(
    metadata,
    enforce_min_max_values=True,
    enforce_rounding=False,
    epochs = 1000, 
    verbose = True
    )

# Train the synthesizer
synthesizer.fit(real_data)

# Get and save progress
progress = synthesizer.get_loss_values()
progress.to_csv('loss_progress/copula_loss.csv')

# Save model
synthesizer.save('saved_models/copula.pkl')

# Generate synthetic data
synthetic_data = synthesizer.sample(num_rows=1000)

synthetic_data.to_csv('datasets/copula_data.csv', index = False)

'''
loss_fig = synthesizer.get_loss_values_plot()
loss_fig.write_image('loss_progress/copula_loss_plot.png')
'''