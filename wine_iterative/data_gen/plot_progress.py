import plotly.graph_objects as go
import pandas as pd



def plot_all_gan_loss(modeltype, gan_part):

    if modeltype == 'ctgan':
        model_name = 'CTGAN'

    elif modeltype == 'copula':
        model_name = 'CopulaGAN'

    fig = go.Figure()
    colours = ["firebrick", "goldenrod", "forestgreen", "lightseagreen", "steelblue", "darksalmon", "yellowgreen", "teal", "cornflowerblue", "slateblue"]

    for p, c in zip(range(10, 109, 10), colours):
        filepath = f'loss_progress/{p}_train/{modeltype}_{p}_wine_loss.csv'
        df = pd.read_csv(filepath)


        epochs = df['Epoch']
        loss = df[f'{gan_part} Loss']


        fig.add_trace(go.Scatter(x=epochs, 
                            y = loss, 
                            mode='lines', 
                            name=p,
                            line=dict(color=c)))
            
    fig.update_layout(
            title=dict(text=f'{gan_part} loss progress for {model_name},<br>wine data', font=dict(size=28)),
            xaxis=dict(title=dict(text = 'Epoch',
                                font = dict(size=20)), 
                        tickfont=dict(size=20)),
            yaxis=dict(title=dict(text = 'Loss',
                                font = dict(size=20)), 
                        tickfont=dict(size=20)),
            legend=dict(title=dict(text= "Percentage<br>training data",
                                   font = dict(size=20)),
                                font=dict(size=20)))


    img_bytes = fig.to_image(format="png", scale=3)
    with open(f'loss_progress/{modeltype}_{gan_part}_progress_plot.png', "wb") as f:
        f.write(img_bytes)






def plot_all_tvae_loss():
    modeltype = 'tvae'
    model_name = 'TVAE'  

    fig = go.Figure()

    colours = ["firebrick", "goldenrod", "forestgreen", "lightseagreen", "steelblue", "darksalmon", "yellowgreen", "teal", "cornflowerblue", "slateblue"]

    for p, c in zip(range(10, 109, 10), colours):
        filepath = f'loss_progress/{p}_train/{modeltype}_{p}_wine_loss.csv'
        df = pd.read_csv(filepath)


        if df['Batch'][1] == 1:
            all_loss = df['Loss']
            tot_epochs = df['Epoch'].iloc[-1]
            epochs = list(range(0, tot_epochs+1))
            loss = [(all_loss.iloc[i] + all_loss.iloc[i+1])/2 for i in range(0, tot_epochs*2+1, 2)] 

        else:
            epochs = df['Epoch']
            loss = df['Loss']

        fig.add_trace(go.Scatter(x=epochs, 
                            y = loss, 
                            mode='lines', 
                            name=p,
                            line=dict(color=c)))
            
    fig.update_layout(
            title=dict(text=f'Loss progress for {model_name},<br>wine data', font=dict(size=28)),
            xaxis=dict(title=dict(text = 'Epoch',
                                font = dict(size=20)), 
                        tickfont=dict(size=20)),
            yaxis=dict(title=dict(text = 'Loss',
                                font = dict(size=20)), 
                        tickfont=dict(size=20)),
            legend=dict(title=dict(text= "Percentage<br>training data",
                                   font = dict(size=20)),
                                font=dict(size=20)))

    img_bytes = fig.to_image(format="png", scale=3)
    with open(f'loss_progress/{modeltype}_progress_plot.png', "wb") as f:
        f.write(img_bytes)




def plotting_gan_loss(modeltype, percentage):

    filepath = f'loss_progress/{percentage}_train/{modeltype}_{percentage}_wine_loss.csv'

    if modeltype == 'ctgan':
        model_name = 'CTGAN'

    elif modeltype == 'copula':
        model_name = 'CopulaGAN'

    else:
        print('Wrong modeltype')
        exit()

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
        title=dict(text=f'Loss progress for {model_name}, {percentage} % of wine data', font=dict(size=28)),
        xaxis=dict(title=dict(text = 'Epoch',
                            font = dict(size=18)), 
                    tickfont=dict(size=18)),
        yaxis=dict(title=dict(text = 'Loss',
                            font = dict(size=18)), 
                    tickfont=dict(size=18)),
        legend=dict(font=dict(size=20)))

    img_bytes = fig.to_image(format="png", scale=3)
    with open(f'loss_progress/{modeltype}_{percentage}_progress_plot.png', "wb") as f:
        f.write(img_bytes)




plot_all_tvae_loss()

plot_all_gan_loss('copula', 'Discriminator')
plot_all_gan_loss('copula', 'Generator')

plot_all_gan_loss('ctgan', 'Discriminator')
plot_all_gan_loss('ctgan', 'Generator')

for p in range(10, 109, 10):
    plotting_gan_loss('copula', p)
    plotting_gan_loss('ctgan', p)


