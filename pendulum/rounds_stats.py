from math import pi as M_PI
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import csv
from tqdm import tqdm
import pandas as pd

first_round = []
total_rounds = []


def create_stats(dataset):
    samples = 1000

    if dataset == 'real':
        foldername = "real_data/polar/polar"

    elif dataset == 'dgan':
        foldername = "synthetic_data/dgan/wrapped/wrapped_synthetic_sample"

    elif dataset == 'dgan_scaling':
        foldername = "synthetic_data/dgan/dgan_scaling/dgan_synthetic_sample"
    
    elif dataset == 'timegan':
        foldername = "synthetic_data/timegan/synthetic_sample"

    elif dataset == 'timegan_unwrapped':
        foldername = "synthetic_data/timegan/unwrapped/synthetic_sample"


    first_round = []
    total_rounds = []


    for s in tqdm(range(0, samples)):
        filepath = f"{foldername}_{s:03d}.csv" 
        df = pd.read_csv(filepath)

        theta_2 = df['theta2']

        if dataset == 'dgan' or dataset == 'timegan':
            theta_2 = np.unwrap(theta_2)  #Make it continuous, instead of cyclic, to count the number of cycles

        theta_2_i = theta_2[0]
        theta_2_r = theta_2[0]
        theta_2_chaos_count = 0

        theta_2_first_round = 0

        for step in range(0, samples):

            if abs(theta_2_r - theta_2[step]) > 2*M_PI:
                    theta_2_r = theta_2[step]
                    theta_2_chaos_count += 1

        first_round.append(theta_2_first_round)
        total_rounds.append(theta_2_chaos_count)

    filename = f'stats/{dataset}_stats.csv'
    header1 = 'first_round'
    header2 = 'total_rounds'

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([header1, header2])  # Write headers
        for a, b in zip(first_round, total_rounds):
            writer.writerow([a, b])
            




def first_round_histogram(dataset):
    # dataset = 'real', 'dgan' or 'timegan'
    if dataset == 'real':
        name = 'real'

    elif dataset == 'dgan':
        name = 'DoppelGANger'

    elif dataset == 'dgan_scaling':
        name = 'DoppelGANger unwrapped'
    
    elif dataset == 'timegan':
        name = 'TimeGAN'

    elif dataset == 'timegan_unwrapped':
        name = 'TimeGAN unwrapped'

    df = pd.read_csv(f'stats/{dataset}_stats.csv')
    first_round = df['first_round']

    non_zero_first = [x for x in first_round if x > 0]


    # Histogram of first round
    fig = go.Figure(
        data=[go.Histogram(x=first_round, 
                        xbins=dict(
                start=0,
                size=10
            ), 
            marker_color='blue')],
        layout=go.Layout(
            title=dict(
                text = f'Histogram of time for first round,<br>{name} data',
                font = dict(size = 28)
            ),
            xaxis=dict(title='Time', 
                    title_font = dict(size = 22)),
            yaxis=dict(title='Frequency, logarithmic scale',
                    title_font = dict(size = 22),
                    type = 'log'),
            bargap=0.1
        )
    )

    img_bytes = fig.to_image(format="png", scale=3)
    with open(f'stats/histogram_first_round_{dataset}.png', "wb") as f:
        f.write(img_bytes)


    # Histogram of first round
    fig = go.Figure(
        data=[go.Histogram(x=non_zero_first, 
                        xbins=dict(
                start=0,
                size=10
            ),
            marker_color='blue')],
        layout=go.Layout(
            title=dict(
                text = f'Histogram of non-zero time for first round,<br>{name} data',
                font = dict(size = 28)
            ),
            xaxis=dict(title='Time', 
                    title_font = dict(size = 22)),
            yaxis=dict(title='Frequency, logarithmic scale',
                    title_font = dict(size = 22),
                    type = 'log'),
            bargap=0.1
        )
    )

    img_bytes = fig.to_image(format="png", scale=3)
    with open(f'stats/histogram_first_round_non_zero_{dataset}.png', "wb") as f:
        f.write(img_bytes)



def count_histograms(dataset):
    # dataset = 'real', 'dgan' or 'timegan'
    if dataset == 'real':
        name = 'real'
        c = 'firebrick'

    elif dataset == 'dgan':
        name = 'DoppelGANger'
        c = 'lightseagreen'

    elif dataset == 'dgan_scaling':
        name = 'DoppelGANger unwrapped'
        c = 'lightseagreen'
    
    elif dataset == 'timegan':
        name = 'TimeGAN'
        c = 'cornflowerblue'

    elif dataset == 'timegan_unwrapped':
        name = 'TimeGAN unwrapped'
        c = 'cornflowerblue'

    df = pd.read_csv(f'stats/{dataset}_stats.csv')
    total_rounds = df['total_rounds']

    non_zero_total = [x for x in total_rounds if x > 0]

    if dataset == 'dgan_scaling':
        # Histogram of count
        fig = go.Figure(
            data=[go.Histogram(x=total_rounds, nbinsx=10, marker_color=c)],
            layout=go.Layout(
                title=dict(
                    text = f'Histogram of number of rounds,<br>{name} data',
                    font = dict(size = 28)
                ),
                xaxis=dict(title='No. of rounds', 
                        title_font = dict(size = 22),
                        tickfont=dict(size=20)),
                yaxis=dict(title='Frequency, logarithmic scale',
                        title_font = dict(size = 22),
                        tickfont=dict(size=20),
                        type = 'log'),
                bargap=0.1
            )
        )


        # Save figure
        img_bytes = fig.to_image(format="png", scale=3)
        with open(f'stats/histogram_total_rounds_{dataset}.png', "wb") as f:
            f.write(img_bytes)

        # Histogram of count
        fig = go.Figure(
            data=[go.Histogram(x=non_zero_total, nbinsx=10, marker_color=c)],
            layout=go.Layout(
                title=dict(
                    text = f'Histogram of number of non-zero rounds,<br>{name} data',
                    font = dict(size = 28)
                ),
                xaxis=dict(title='No. of rounds', 
                        title_font = dict(size = 22),
                        tickfont=dict(size=20)),
                yaxis=dict(title='Frequency, logarithmic scale',
                        title_font = dict(size = 22),
                        tickfont=dict(size=20),
                        type = 'log'),
                bargap=0.1
            )
        )


        img_bytes = fig.to_image(format="png", scale=3)
        with open(f'stats/histogram_total_rounds_non_zero_{dataset}.png', "wb") as f:
            f.write(img_bytes)

    else:
    
        # Histogram of count
        fig = go.Figure(
            data=[go.Histogram(x=total_rounds, nbinsx=10, marker_color=c)],
            layout=go.Layout(
                title=dict(
                    text = f'Histogram of number of rounds,<br>{name} data',
                    font = dict(size = 28)
                ),
                xaxis=dict(title='No. of rounds', 
                        title_font = dict(size = 22),
                        tickfont=dict(size=20)),
                yaxis=dict(title='Frequency',
                        title_font = dict(size = 22),
                        tickfont=dict(size=20)),
                bargap=0.1
            )
        )


        # Save figure
        img_bytes = fig.to_image(format="png", scale=3)
        with open(f'stats/histogram_total_rounds_{dataset}.png', "wb") as f:
            f.write(img_bytes)

        # Histogram of count
        fig = go.Figure(
            data=[go.Histogram(x=non_zero_total, nbinsx=10, marker_color=c)],
            layout=go.Layout(
                title=dict(
                    text = f'Histogram of number of non-zero rounds,<br>{name} data',
                    font = dict(size = 28)
                ),
                xaxis=dict(title='No. of rounds', 
                        title_font = dict(size = 22),
                        tickfont=dict(size=20)),
                yaxis=dict(title='Frequency',
                        title_font = dict(size = 22),
                        tickfont=dict(size=20)),
                bargap=0.1
            )
        )


        img_bytes = fig.to_image(format="png", scale=3)
        with open(f'stats/histogram_total_rounds_non_zero_{dataset}.png', "wb") as f:
            f.write(img_bytes)

dataset = 'real'

#create_stats(dataset)
count_histograms(dataset)

dataset = 'dgan'
count_histograms(dataset)

dataset = 'dgan_scaling'
count_histograms(dataset)

dataset = 'timegan'
count_histograms(dataset)

dataset = 'timegan_unwrapped'
count_histograms(dataset)


