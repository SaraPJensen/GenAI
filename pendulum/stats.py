from math import pi as M_PI
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import csv
from tqdm import tqdm
import pandas as pd

first_round = []
total_rounds = []


def create_stats():
    first_round = []
    total_rounds = []

    samples = 1000

    for s in tqdm(range(0, samples)):
        filepath = f"real_data/polar/polar_{s:03d}.csv" 
        df = pd.read_csv(filepath)

        time = df['t']
        theta_1 = df['theta1']
        theta_2 = df['theta2']

        theta_2_i = theta_2[0]
        theta_2_r = theta_2[0]

        theta_2_chaos_count = 0

        theta_2_first_round = 0

        for step in range(0, len(time)):


            if theta_2_chaos_count == 0:
                if abs(theta_2_i - theta_2[step]) > 2*M_PI:
                    theta_2_first_round = time[step].item()
                    theta_2_r = theta_2[step]
                    theta_2_chaos_count += 1

            elif abs(theta_2_r - theta_2[step]) > 2*M_PI:
                    theta_2_r = theta_2[step]
                    theta_2_chaos_count += 1

        first_round.append(theta_2_first_round)
        total_rounds.append(theta_2_chaos_count)

    filename = 'stats/real_stats.csv'
    header1 = 'first_round'
    header2 = 'total_rounds'

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([header1, header2])  # Write headers
        for a, b in zip(first_round, total_rounds):
            writer.writerow([a, b])
            
    return first_round, total_rounds
                

#first_round, total_rounds = create_stats()



df = pd.read_csv('real_stats.csv')
first_round = df['first_round']
total_rounds = df['total_rounds']


non_zero_first = [x for x in first_round if x > 0]
non_zero_total = [x for x in total_rounds if x > 0]


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
             text = 'Histogram of time for first round, real data',
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

# Save figure (as PNG)
pio.write_image(fig, 'stats/histogram_first_round_real.pdf', width=800, height=600)


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
             text = 'Histogram of non-zero time for first round, real data',
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

# Save figure (as PNG)
pio.write_image(fig, 'stats/histogram_first_round_non_zero_real.pdf', width=800, height=600)




# Histogram of first round
fig = go.Figure(
    data=[go.Histogram(x=total_rounds, nbinsx=10, marker_color='blue')],
    layout=go.Layout(
        title=dict(
             text = 'Histogram of number of rounds, real_data',
             font = dict(size = 28)
        ),
        xaxis=dict(title='No. of rounds', 
                   title_font = dict(size = 22)),
        yaxis=dict(title='Frequency',
                   title_font = dict(size = 22)),
        bargap=0.1
    )
)



# Save figure (as PNG)
pio.write_image(fig, 'stats/histogram_total_rounds_real.pdf', width=800, height=600)


# Histogram of first round
fig = go.Figure(
    data=[go.Histogram(x=non_zero_total, nbinsx=10, marker_color='blue')],
    layout=go.Layout(
        title=dict(
             text = 'Histogram of number of non_zero rounds, real data',
             font = dict(size = 28)
        ),
        xaxis=dict(title='No. of rounds', 
                   title_font = dict(size = 22)),
        yaxis=dict(title='Frequency',
                   title_font = dict(size = 22)),
        bargap=0.1
    )
)



# Save figure (as PNG)
pio.write_image(fig, 'stats/histogram_total_rounds_non_zero_real.pdf', width=800, height=600)