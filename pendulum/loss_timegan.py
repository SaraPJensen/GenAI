import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_timegan_progress(wrapping = None):
    model_parts = ['adversarial', 'embedder', 'supervisor']

    for part in model_parts:

        if wrapping == 'unwrapped':
            filepath = f'timegan_loss/{part}_loss_unwrapped.csv'
            add_on = ',<br>unwrapped data'
            filefix = '_unwrapped'

        else:
            filepath = f'timegan_loss/{part}_loss.csv'
            add_on = ''
            filefix = ''


        df = pd.read_csv(filepath)

        epoch = df['epoch']

        if part == 'adversarial':
            disc_loss = df['discriminator_loss']
            gen_loss = df['generator_loss']

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=epoch, 
                            y = disc_loss, 
                            mode='lines', 
                            line=dict(color='firebrick')))

            fig.update_layout(
            title=dict(text=f'Loss progress for TimeGAN discriminator{add_on}', font=dict(size=28)),
            xaxis=dict(title=dict(text = 'Epoch',
                                font = dict(size=18)), 
                        tickfont=dict(size=18)),
            yaxis=dict(title=dict(text = 'Loss',
                                font = dict(size=18)), 
                        tickfont=dict(size=18)),
            legend=dict(font=dict(size=20)))

            img_bytes = fig.to_image(format="png", scale=3)
            with open(f'timegan_loss/discriminator{filefix}.png', "wb") as f:
                f.write(img_bytes)
            

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=epoch, 
                            y = gen_loss, 
                            mode='lines', 
                            line=dict(color='darkturquoise')))


            fig.update_layout(
            title=dict(text=f'Loss progress for TimeGAN generator{add_on}', font=dict(size=28)),
            xaxis=dict(title=dict(text = 'Epoch',
                                font = dict(size=18)), 
                        tickfont=dict(size=18)),
            yaxis=dict(title=dict(text = 'Loss',
                                font = dict(size=18)), 
                        tickfont=dict(size=18)),
            legend=dict(font=dict(size=20)))


            img_bytes = fig.to_image(format="png", scale=3)
            with open(f'timegan_loss/generator{filefix}.png', "wb") as f:
                f.write(img_bytes)



        else:
            loss = df[f'{part}_loss']

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=epoch, 
                            y = loss, 
                            mode='lines', 
                            line=dict(color='firebrick')))

            fig.update_layout(
            title=dict(text=f'Loss progress for TimeGAN {part}{add_on}', font=dict(size=28)),
            xaxis=dict(title=dict(text = 'Epoch',
                                font = dict(size=18)), 
                        tickfont=dict(size=18)),
            yaxis=dict(title=dict(text = 'Loss',
                                font = dict(size=18)), 
                        tickfont=dict(size=18)),
            legend=dict(font=dict(size=20)))

            img_bytes = fig.to_image(format="png", scale=3)
            with open(f'timegan_loss/{part}{filefix}.png', "wb") as f:
                f.write(img_bytes)


plot_timegan_progress('unwrapped')