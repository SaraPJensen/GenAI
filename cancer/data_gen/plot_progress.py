import plotly.graph_objects as go
import pandas as pd


def plotting_loss(modeltype):
    if modeltype == 'ctgan':
        filepath = 'loss_progress/ctgan_loss.csv'
        model_name = 'CTGAN'

    elif modeltype == 'copula':
        filepath = 'loss_progress/copula_loss.csv'
        model_name = 'CopulaGAN'

    elif modeltype == 'tvae':
        filepath = 'loss_progress/tvae_loss.csv'
        model_name = 'TVAE'  

    else:
        print('Wrong modeltype')
        exit()

    if modeltype == 'tvae':
        df = pd.read_csv(filepath)

        all_loss = df['Loss']
        tot_epochs = df['Epoch'].iloc[-1]
        epochs = list(range(0, tot_epochs+1))


        loss = [(all_loss.iloc[i] + all_loss.iloc[i+1])/2 for i in range(0, tot_epochs*2+1, 2)] 

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=epochs, 
                                y = loss, 
                                mode='lines', 
                                #name='Generator<br> Loss',
                                line=dict(color='firebrick')))


        fig.update_layout(
            title=dict(text=f'Loss progress for {model_name}', font=dict(size=30)),
            xaxis=dict(title=dict(text = 'Epoch',
                                font = dict(size=18)), 
                        tickfont=dict(size=18)),
            yaxis=dict(title=dict(text = 'Loss',
                                font = dict(size=18)), 
                        tickfont=dict(size=18)),
            legend=dict(font=dict(size=20)))


        fig.write_image(f'loss_progress/{modeltype}_progress_plot.pdf')

        
    
    else: 
        df = pd.read_csv(filepath)

        epochs = df['Epoch']
        generator_loss = df['Generator Loss']
        discriminator_loss = df['Discriminator Loss']

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=epochs, 
                                y = generator_loss, 
                                mode='lines', 
                                name='Generator<br> Loss',
                                line=dict(color='firebrick')))

        fig.add_trace(go.Scatter(x=epochs, 
                                y = discriminator_loss, 
                                mode='lines', 
                                name='Discriminator<br> Loss',
                                line=dict(color='darkturquoise')))


        fig.update_layout(
            title=dict(text=f'Loss progress for {model_name}', font=dict(size=30)),
            xaxis=dict(title=dict(text = 'Epoch',
                                font = dict(size=18)), 
                        tickfont=dict(size=18)),
            yaxis=dict(title=dict(text = 'Loss',
                                font = dict(size=18)), 
                        tickfont=dict(size=18)),
            legend=dict(font=dict(size=20)))

        fig.write_image(f'loss_progress/{modeltype}_progress_plot.pdf')


modeltype = 'tvae'

plotting_loss(modeltype)
