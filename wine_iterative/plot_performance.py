import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_final_real_test_acc():
    modelnames = ['real', 'gaussian', 'ctgan', 'tvae', 'copula']
    dataset_names = ['Real', 'GaussianCopula', 'CTGAN', 'TVAE', 'CopulaGAN']

    fig = go.Figure()
    colours = ["firebrick", "goldenrod", "forestgreen", "slateblue", "cornflowerblue", "darksalmon", "yellowgreen", "teal", "cornflowerblue", "slateblue"]

    for (model, c, data_name) in zip(modelnames, colours, dataset_names):
        final_real_test_acc = []

        for p in range(10, 100, 10):
            filepath = f'training_progress/{p}_train/{model}_progress.csv'
            df = pd.read_csv(filepath)

            real_test_acc = df['real_test_accuracy']
            final_real_test_acc.append(real_test_acc.iloc[-1])
        
        fig.add_trace(go.Scatter(x=list(range(10, 100, 10)), 
                            y = final_real_test_acc, 
                            mode='lines+markers', 
                            name=f'{data_name}',
                            line=dict(color=c)))

    fig.update_layout(
        title=dict(text=f'Final prediction accuracy on test-set,<br>real wine quality data', font=dict(size=28)),
        xaxis=dict(title=dict(text = '% of real data to train synthesizer',
                            font = dict(size=20)), 
                    tickfont=dict(size=20)),
        yaxis=dict(title=dict(text = 'Accuracy',
                            font = dict(size=20)), 
                    tickfont=dict(size=20)),
        legend=dict(title=dict(text= "Dataset synthesizer",
                                font = dict(size=22)),
                            font=dict(size=20)))
    

    fig.write_image(f'training_progress/real_test_final_acc.pdf')


def plot_final_real_complete_acc():
    modelnames = ['real', 'gaussian', 'ctgan', 'tvae', 'copula']
    dataset_names = ['Real', 'GaussianCopula', 'CTGAN', 'TVAE', 'CopulaGAN']

    fig = go.Figure()
    colours = ["firebrick", "goldenrod", "forestgreen", "slateblue", "cornflowerblue", "darksalmon", "yellowgreen", "teal", "cornflowerblue", "slateblue"]

    for (model, c, data_name) in zip(modelnames, colours, dataset_names):
        final_real_complete_acc = []

        for p in range(10, 100, 10):
            filepath = f'training_progress/{p}_train/{model}_progress.csv'
            df = pd.read_csv(filepath)

            real_complete_acc = df['real_complete_accuracy']
            final_real_complete_acc.append(real_complete_acc.iloc[-1])
        
        fig.add_trace(go.Scatter(x=list(range(10, 100, 10)), 
                            y = final_real_complete_acc, 
                            mode='lines+markers', 
                            name=f'{data_name}',
                            line=dict(color=c)))

    fig.update_layout(
        title=dict(text=f'Final prediction accuracy on complete set,<br>real wine quality data', font=dict(size=28)),
        xaxis=dict(title=dict(text = '% of real data to train synthesizer',
                            font = dict(size=20)), 
                    tickfont=dict(size=20)),
        yaxis=dict(title=dict(text = 'Accuracy',
                            font = dict(size=20)), 
                    tickfont=dict(size=20)),
        legend=dict(title=dict(text= "Dataset synthesizer",
                                font = dict(size=22)),
                            font=dict(size=20)))
    

    fig.write_image(f'training_progress/real_complete_final_acc.pdf')



def plot_final_test_acc():
    modelnames = ['real', 'gaussian', 'ctgan', 'tvae', 'copula']
    dataset_names = ['Real', 'GaussianCopula', 'CTGAN', 'TVAE', 'CopulaGAN']

    fig = go.Figure()
    colours = ["firebrick", "goldenrod", "forestgreen", "slateblue", "cornflowerblue", "darksalmon", "yellowgreen", "teal", "cornflowerblue", "slateblue"]

    for (model, c, data_name) in zip(modelnames, colours, dataset_names):
        final_test_acc = []

        for p in range(10, 100, 10):
            filepath = f'training_progress/{p}_train/{model}_progress.csv'
            df = pd.read_csv(filepath)

            real_test_acc = df['test_accuracy']
            final_test_acc.append(real_test_acc.iloc[-1])
        
        fig.add_trace(go.Scatter(x=list(range(10, 100, 10)), 
                            y = final_test_acc, 
                            mode='lines+markers', 
                            name=f'{data_name}',
                            line=dict(color=c)))

    fig.update_layout(
        title=dict(text=f'Final prediction accuracy on test-set,<br>wine quality data', font=dict(size=28)),
        xaxis=dict(title=dict(text = '% of real data to train synthesizer',
                            font = dict(size=20)), 
                    tickfont=dict(size=20)),
        yaxis=dict(title=dict(text = 'Accuracy',
                            font = dict(size=20)), 
                    tickfont=dict(size=20)),
        legend=dict(title=dict(text= "Dataset synthesizer",
                                font = dict(size=22)),
                            font=dict(size=20)))
    

    fig.write_image(f'training_progress/test_final_acc.pdf')


def plot_final_training_acc():
    modelnames = ['real', 'gaussian', 'ctgan', 'tvae', 'copula']
    dataset_names = ['Real', 'GaussianCopula', 'CTGAN', 'TVAE', 'CopulaGAN']

    fig = go.Figure()
    colours = ["firebrick", "goldenrod", "forestgreen", "slateblue", "cornflowerblue", "darksalmon", "yellowgreen", "teal", "cornflowerblue", "slateblue"]

    for (model, c, data_name) in zip(modelnames, colours, dataset_names):
        final_training_acc = []

        for p in range(10, 100, 10):
            filepath = f'training_progress/{p}_train/{model}_progress.csv'
            df = pd.read_csv(filepath)

            real_training_acc = df['train_accuracy']
            final_training_acc.append(real_training_acc.iloc[-1])
        
        fig.add_trace(go.Scatter(x=list(range(10, 100, 10)), 
                            y = final_training_acc, 
                            mode='lines+markers', 
                            name=f'{data_name}',
                            line=dict(color=c)))

    fig.update_layout(
        title=dict(text=f'Final prediction accuracy on training-set,<br>wine quality data', font=dict(size=28)),
        xaxis=dict(title=dict(text = '% of real data to train synthesizer',
                            font = dict(size=20)), 
                    tickfont=dict(size=20)),
        yaxis=dict(title=dict(text = 'Accuracy',
                            font = dict(size=20)), 
                    tickfont=dict(size=20)),
        legend=dict(title=dict(text= "Dataset synthesizer",
                                font = dict(size=22)),
                            font=dict(size=20)))
    

    fig.write_image(f'training_progress/training_final_acc.pdf')



#plot_final_real_test_acc()
# plot_final_test_acc()
# plot_final_training_acc()
plot_final_real_complete_acc()


