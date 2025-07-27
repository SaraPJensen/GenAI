from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
import pandas as pd

real_data = pd.read_csv('../datasets/real_wine_data.csv')
metadata = Metadata.load_from_json('../datasets/wine_metadata.json')

# Step 1: Create the synthesizer
synthesizer = CTGANSynthesizer(
    metadata,
    enforce_min_max_values=True,
    enforce_rounding=False,
    epochs = 1000, 
    verbose = True
    )

# Train the synthesizer
synthesizer.fit(real_data)

# Get and save loss progress
progress = synthesizer.get_loss_values()
progress.to_csv('loss_progress/ctgan_loss.csv', index = False)

#Save model
synthesizer.save('saved_models/ctgan.pkl')

# Generate synthetic data
synthetic_data = synthesizer.sample(num_rows=1000)
synthetic_data.to_csv('../datasets/ctgan_data.csv', index = False)

'''
# Generate loss figure
loss_fig = synthesizer.get_loss_values_plot()
loss_fig.write_image('loss_progress/ctgan_loss_plot.png')
'''