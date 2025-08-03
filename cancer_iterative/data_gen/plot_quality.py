import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_complete_col_shape():
    modelnames = ['real', 'gaussian', 'ctgan', 'tvae', 'copula']
    dataset_names = ['Real', 'GaussianCopula', 'CTGAN', 'TVAE', 'CopulaGAN']

    fig = go.Figure()
    colours = ["firebrick", "goldenrod", "forestgreen", "slateblue", "cornflowerblue", "darksalmon", "yellowgreen", "teal", "cornflowerblue", "slateblue"]

    for (model, c, data_name) in zip(modelnames, colours, dataset_names):
        quality_file = f'quality_reports/{model}.csv'
        df = pd.read_csv(quality_file)

        col_shape_scores = df['all_data_col_shapes']
        percentages = df['percentage']

        fig.add_trace(go.Scatter(x=percentages, 
                            y = col_shape_scores, 
                            mode='lines+markers', 
                            name=f'{data_name}',
                            line=dict(color=c)))

    fig.update_layout(
        title=dict(text=f'Column shapes score,<br>w.r.t. complete real cancer dataset', font=dict(size=28)),
        xaxis=dict(title=dict(text = '% of real data to train synthesizer',
                            font = dict(size=20)), 
                    tickfont=dict(size=20)),
        yaxis=dict(title=dict(text = 'Column shapes score',
                            font = dict(size=20)), 
                    tickfont=dict(size=20)),
        legend=dict(title=dict(text= "Dataset synthesizer",
                                font = dict(size=22)),
                            font=dict(size=20)))
    

    img_bytes = fig.to_image(format="png", scale=3)
    with open('quality_reports/complete_col_shapes.png', "wb") as f:
        f.write(img_bytes)



def plot_complete_col_trends():
    modelnames = ['real', 'gaussian', 'ctgan', 'tvae', 'copula']
    dataset_names = ['Real', 'GaussianCopula', 'CTGAN', 'TVAE', 'CopulaGAN']

    fig = go.Figure()
    colours = ["firebrick", "goldenrod", "forestgreen", "slateblue", "cornflowerblue", "darksalmon", "yellowgreen", "teal", "cornflowerblue", "slateblue"]

    for (model, c, data_name) in zip(modelnames, colours, dataset_names):
        quality_file = f'quality_reports/{model}.csv'
        df = pd.read_csv(quality_file)

        col_shape_scores = df['all_data_col_trends']
        percentages = df['percentage']

        fig.add_trace(go.Scatter(x=percentages, 
                            y = col_shape_scores, 
                            mode='lines+markers', 
                            name=f'{data_name}',
                            line=dict(color=c)))

    fig.update_layout(
        title=dict(text=f'Column pair trends score,<br>w.r.t. complete real cancer dataset', font=dict(size=28)),
        xaxis=dict(title=dict(text = '% of real data to train synthesizer',
                            font = dict(size=20)), 
                    tickfont=dict(size=20)),
        yaxis=dict(title=dict(text = 'Column trends score',
                            font = dict(size=20)), 
                    tickfont=dict(size=20)),
        legend=dict(title=dict(text= "Dataset synthesizer",
                                font = dict(size=22)),
                            font=dict(size=20)))
    

    img_bytes = fig.to_image(format="png", scale=3)
    with open('quality_reports/complete_col_trends.png', "wb") as f:
        f.write(img_bytes)



def plot_test_col_shape():
    modelnames = ['real', 'gaussian', 'ctgan', 'tvae', 'copula']
    dataset_names = ['Real', 'GaussianCopula', 'CTGAN', 'TVAE', 'CopulaGAN']

    fig = go.Figure()
    colours = ["firebrick", "goldenrod", "forestgreen", "slateblue", "cornflowerblue", "darksalmon", "yellowgreen", "teal", "cornflowerblue", "slateblue"]

    for (model, c, data_name) in zip(modelnames, colours, dataset_names):
        quality_file = f'quality_reports/{model}.csv'
        df = pd.read_csv(quality_file)

        col_shape_scores = df['test_data_col_shapes']
        percentages = df['percentage']

        fig.add_trace(go.Scatter(x=percentages, 
                            y = col_shape_scores, 
                            mode='lines+markers', 
                            name=f'{data_name}',
                            line=dict(color=c)))

    fig.update_layout(
        title=dict(text=f'Column shapes score,<br>w.r.t. real test cancer dataset', font=dict(size=28)),
        xaxis=dict(title=dict(text = '% of real data to train synthesizer',
                            font = dict(size=20)), 
                    tickfont=dict(size=20)),
        yaxis=dict(title=dict(text = 'Column shapes score',
                            font = dict(size=20)), 
                    tickfont=dict(size=20)),
        legend=dict(title=dict(text= "Dataset synthesizer",
                                font = dict(size=22)),
                            font=dict(size=20)))
    

    img_bytes = fig.to_image(format="png", scale=3)
    with open('quality_reports/test_col_shapes.png', "wb") as f:
        f.write(img_bytes)



def plot_test_col_trends():
    modelnames = ['real', 'gaussian', 'ctgan', 'tvae', 'copula']
    dataset_names = ['Real', 'GaussianCopula', 'CTGAN', 'TVAE', 'CopulaGAN']

    fig = go.Figure()
    colours = ["firebrick", "goldenrod", "forestgreen", "slateblue", "cornflowerblue", "darksalmon", "yellowgreen", "teal", "cornflowerblue", "slateblue"]

    for (model, c, data_name) in zip(modelnames, colours, dataset_names):
        quality_file = f'quality_reports/{model}.csv'
        df = pd.read_csv(quality_file)

        col_shape_scores = df['test_data_col_trends']
        percentages = df['percentage']

        fig.add_trace(go.Scatter(x=percentages, 
                            y = col_shape_scores, 
                            mode='lines+markers', 
                            name=f'{data_name}',
                            line=dict(color=c)))

    fig.update_layout(
        title=dict(text=f'Column pair trends score,<br>w.r.t. real test cancer dataset', font=dict(size=28)),
        xaxis=dict(title=dict(text = '% of real data to train synthesizer',
                            font = dict(size=20)), 
                    tickfont=dict(size=20)),
        yaxis=dict(title=dict(text = 'Column trends score',
                            font = dict(size=20)), 
                    tickfont=dict(size=20)),
        legend=dict(title=dict(text= "Dataset synthesizer",
                                font = dict(size=22)),
                            font=dict(size=20)))
    

    img_bytes = fig.to_image(format="png", scale=3)
    with open('quality_reports/test_col_trends.png', "wb") as f:
        f.write(img_bytes)



plot_complete_col_shape()
plot_complete_col_trends()

plot_test_col_shape()
plot_test_col_trends()