from math import pi as M_PI
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import csv
from tqdm import tqdm
import pandas as pd


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


    percentage_top_1 = []
    percentage_bottom_1 = []

    percentage_top_2 = []
    percentage_bottom_2 = []

    for s in tqdm(range(0, samples)):
        filepath = f"{foldername}_{s:03d}.csv" 
        df = pd.read_csv(filepath)

        theta_1 = df['theta1']
        theta_2 = df['theta2']
        

        if dataset == 'real' or dataset == 'dgan_scaling' or dataset == 'timegan_unwrapped':
            theta_1 = np.mod(theta_1, 2 * np.pi)  #Make it be between 0 and 2pi
            theta_2 = np.mod(theta_2, 2 * np.pi)  #Make it be between 0 and 2pi


        lower = np.pi / 2
        upper = 3 * np.pi / 2

        mask_1 = (theta_1 >= lower) & (theta_1 <= upper)
        percentage_1 = np.mean(mask_1) * 100

        mask_2 = (theta_2 >= lower) & (theta_2 <= upper)
        percentage_2 = np.mean(mask_2) * 100

        percentage_top_1.append(percentage_1)
        percentage_bottom_1.append(100 - percentage_1)

        percentage_top_2.append(percentage_2)
        percentage_bottom_2.append(100 - percentage_2)


    filename = f'stats/{dataset}_position_stats.csv'
    header1 = 'percentage_top_1'
    header2 =  'percentage_bottom_1'
    header3 = 'percentage_top_2'
    header4 =  'percentage_bottom_2'
    

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([header1, header2, header3, header4])  # Write headers
        for a, b, c, d in zip(percentage_top_1, percentage_bottom_1, percentage_top_2, percentage_bottom_2):
            writer.writerow([a, b, c, d])
            


def position_histogram(dataset):
    samples = 1000

    if dataset == 'real':
        name = 'real'
        c1 = 'firebrick'
        c2 = 'lightsalmon'

    elif dataset == 'dgan':
        name = 'DoppelGANger'
        c1 = 'lightseagreen'
        c2 = 'paleturquoise'

    elif dataset == 'dgan_scaling':
        name = 'DoppelGANger unwrapped'
        c1 = 'lightseagreen'
        c2 = 'paleturquoise'
    
    elif dataset == 'timegan':
        name = 'TimeGAN'
        c1 = 'cornflowerblue'
        c2 = 'powderblue'

    elif dataset == 'timegan_unwrapped':
        name = 'TimeGAN unwrapped'
        c1 = 'cornflowerblue'
        c2 = 'powderblue'


    filename = f'stats/{dataset}_position_stats.csv'

    df = pd.read_csv(filename)

    percentage_top_1 = df['percentage_top_1']
    percentage_bottom_1 = df['percentage_bottom_1']

    percentage_top_2 = df['percentage_top_2']
    percentage_bottom_2 = df['percentage_bottom_2']


    # --- Precompute histogram bins & counts ---
    bins = np.linspace(0, 100, 21)  # 20 bins: 0%, 5%, ..., 100%
    counts_upper, _ = np.histogram(percentage_top_1, bins=bins)
    counts_lower, _ = np.histogram(percentage_bottom_1, bins=bins)

    # Compute bin centers for plotting
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Mirror lower counts
    counts_lower = -counts_lower

    # --- Build mirrored histogram in Plotly ---
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts_upper,
        name="Upper half",
        marker_color=c1
    ))

    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts_lower,
        name="Lower half",
        marker_color=c2
    ))

    fig.update_layout(
        barmode='overlay',
        bargap=0.05,
        title=dict(
            text = f"Average time spent in upper vs lower half of circle,<br>inner pendulum, {name} data",
            font = dict(size = 26)
        ),
        xaxis=dict(
            title = "Percentage of time",
            title_font = dict(size = 20),
            tickfont=dict(size=20)
        ),
        yaxis=dict(
            title = "Number of samples (mirrored)",
            title_font = dict(size=20),
            tickfont = dict(size = 20),
            tickmode='array',
            tickvals=[-max(counts_upper), 0, max(counts_upper)],
            ticktext=[max(counts_upper), 0, max(counts_upper)]
        ),
        legend=dict(title=dict(text= "Position",
                                font = dict(size=22)),
                            font=dict(size=20))
    )

    # Save figure
    img_bytes = fig.to_image(format="png", scale=3)
    with open(f'stats/histogram_position_m1_{dataset}.png', "wb") as f:
        f.write(img_bytes)





    # --- Precompute histogram bins & counts ---
    bins = np.linspace(0, 100, 21)  # 20 bins: 0%, 5%, ..., 100%
    counts_upper, _ = np.histogram(percentage_top_2, bins=bins)
    counts_lower, _ = np.histogram(percentage_bottom_2, bins=bins)

    # Compute bin centers for plotting
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Mirror lower counts
    counts_lower = -counts_lower

    # --- Build mirrored histogram in Plotly ---
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts_upper,
        name="Upper half",
        marker_color=c1
    ))

    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts_lower,
        name="Lower half",
        marker_color=c2
    ))

    fig.update_layout(
        barmode='overlay',
        bargap=0.05,
        title=dict(
            text = f"Average time spent in upper vs lower half of circle,<br>outer pendulum, {name} data",
            font = dict(size = 26)
        ),
        xaxis=dict(
            title = "Percentage of time",
            title_font = dict(size = 20),
            tickfont=dict(size=20)
        ),
        yaxis=dict(
            title = "Number of samples (mirrored)",
            title_font = dict(size=20),
            tickfont = dict(size = 20),
            tickmode='array',
            tickvals=[-max(counts_upper), 0, max(counts_upper)],
            ticktext=[max(counts_upper), 0, max(counts_upper)]
        ),
        legend=dict(title=dict(text= "Position",
                                font = dict(size=22)),
                            font=dict(size=20))
    )

    # Save figure
    img_bytes = fig.to_image(format="png", scale=3)
    with open(f'stats/histogram_position_m2_{dataset}.png', "wb") as f:
        f.write(img_bytes)





dataset = 'timegan_unwrapped'

#create_stats(dataset)

position_histogram(dataset)

dataset = 'real'
position_histogram(dataset)

dataset = 'timegan'
position_histogram(dataset)

dataset = 'dgan'
position_histogram(dataset)

dataset = 'dgan_scaling'
position_histogram(dataset)
