from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
import plotly as plt
import pandas as pd
import json



def quality(model):
    quality_file = f'quality_reports/{model}.csv'

    with open(quality_file, 'w') as file:
        file.write('percentage,all_data_col_shapes,all_data_col_trends,train_data_col_shapes,train_data_col_trends,test_data_col_shapes,test_data_col_trends\n')

    for p in range (10, 100, 10):
        report_full = QualityReport()
        report_train = QualityReport()
        report_test = QualityReport()

        real_data = pd.read_csv('../datasets/real_cancer_data.csv')

        first_subset = int((p/100)*len(real_data))
        last_subset = int(len(real_data) - first_subset) #Test with the subset of the real data which wasn't used to train the model
        
        real_data_train = real_data.head(first_subset)
        real_data_test = real_data.tail(last_subset)

        synthetic_data = pd.read_csv(f'../datasets/{p}_train/{model}_{p}_cancer_data.csv')

        with open('../datasets/cancer_metadata.json', 'r') as f:
            full_metadata = json.load(f)

        # Extract only the metadata for the single table
        metadata = full_metadata['tables']['table']

        report_full.generate(real_data, synthetic_data, metadata, verbose = False)
        report_train.generate(real_data_train, synthetic_data, metadata, verbose = False)
        report_test.generate(real_data_test, synthetic_data, metadata, verbose = False)

        props_full = report_full.get_properties()
        col_shap = props_full['Score'][0]
        col_trend = props_full['Score'][1]

        props_train = report_train.get_properties()
        col_shap_train = props_train['Score'][0]
        col_trend_train = props_train['Score'][1]

        props_test = report_test.get_properties()
        col_shap_test = props_test['Score'][0]
        col_trend_test = props_test['Score'][1]

        with open(quality_file, 'a') as file:
            file.write(f'{p},{col_shap},{col_trend},{col_shap_train},{col_trend_train},{col_shap_test},{col_trend_test}\n')



def quality_sanity():

    quality_file = f'quality_reports/real.csv'

    with open(quality_file, 'w') as file:
        file.write('percentage,all_data_col_shapes,all_data_col_trends,test_data_col_shapes,test_data_col_trends\n')

    for p in range (10, 100, 10):
        split_report = QualityReport()
        complete_report = QualityReport()

        real_data = pd.read_csv('../datasets/real_cancer_data.csv')

        first_subset = int((p/100)*len(real_data))
        last_subset = int(len(real_data) - first_subset) #Test with the subset of the real data which wasn't used to train the model
        
        real_data_train = real_data.head(first_subset)
        real_data_test = real_data.tail(last_subset)

        # print(first_subset)
        # print(last_subset)
        # print(first_subset+last_subset)
        # print()

        with open('../datasets/cancer_metadata.json', 'r') as f:
            full_metadata = json.load(f)

        # Extract only the metadata for the single table
        metadata = full_metadata['tables']['table']

        #First argument is real data, second is synthetic data
        split_report.generate(real_data_train, real_data_test, metadata, verbose = False)
        complete_report.generate(real_data_train, real_data, metadata, verbose = False)

        complete_props = complete_report.get_properties()
        complete_col_shap = complete_props['Score'][0]
        complete_col_trend = complete_props['Score'][1]

        split_props = split_report.get_properties()
        split_col_shap = split_props['Score'][0]
        split_col_trend = split_props['Score'][1]

 
        with open(quality_file, 'a') as file:
            file.write(f'{p},{complete_col_shap},{complete_col_trend},{split_col_shap},{split_col_trend}\n')



quality_sanity()
exit()


#quality_sanity()

quality('ctgan')
quality('tvae')
quality('gaussian')
quality('copula')


