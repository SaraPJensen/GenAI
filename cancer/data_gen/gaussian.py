from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
import pandas as pd

real_data = pd.read_csv('../datasets/real_cancer_data.csv')
metadata = Metadata.load_from_json('../datasets/cancer_metadata.json')

# Step 1: Create the synthesizer
synthesizer = GaussianCopulaSynthesizer(metadata)

# Step 2: Train the synthesizer
synthesizer.fit(real_data)
synthesizer.save('saved_models/gaussian.pkl')

# Step 3: Generate synthetic data
synthetic_data = synthesizer.sample(num_rows=1000)

synthetic_data.to_csv('../datasets/gaussian_cancer_data.csv', index = False)

