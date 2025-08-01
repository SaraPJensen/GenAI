from sklearn.datasets import load_breast_cancer
import pandas as pd 
import os
from sdv.metadata import Metadata


def make_cancer_data():

    data = load_breast_cancer(as_frame = True)
    df = data.frame  #Include targets as last column: 0 = malignant, 1 = benign

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    #Save as csv
    os.makedirs('../datasets', exist_ok = True)

    df_shuffled.to_csv('../datasets/real_cancer_data.csv', index = False)


    clean_df = pd.read_csv('../datasets/real_cancer_data.csv') #Make metadata on the dataframe without indices

    # Delete the file if it already exists
    if os.path.exists('../datasets/cancer_metadata.json'):
        os.remove('../datasets/cancer_metadata.json')

    metadata = Metadata.detect_from_dataframe(clean_df)
    metadata.save_to_json('../datasets/cancer_metadata.json')


