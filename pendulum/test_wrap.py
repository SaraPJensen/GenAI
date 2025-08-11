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

samples = 6

for s in range(0, samples):
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

print(first_round, total_rounds)






first_round_upwrapped = []
total_rounds_unwrapped = []

samples = 6

for s in range(0, samples):
    filepath = f"real_data/polar/polar_{s:03d}.csv" 
    df = pd.read_csv(filepath)

    theta_2 = df['theta2']

    wrapped_data = np.mod(theta_2, 2 * np.pi)
    
    theta2_unwrapped = np.unwrap(theta_2)

    #print(theta2_unwrapped[:10])

    theta_2_chaos_count = 0
    theta_2_first_round = 0

    theta_2_i = theta_2[0]
    theta_2_r = theta_2[0]

    for step in range(0, len(time)):

        if theta_2_chaos_count == 0:
            if abs(theta_2_i - theta_2[step]) > 2*M_PI:
                theta_2_first_round = time[step].item()
                theta_2_r = theta_2[step]
                theta_2_chaos_count += 1

        elif abs(theta_2_r - theta_2[step]) > 2*M_PI:
                theta_2_r = theta_2[step]
                theta_2_chaos_count += 1

    first_round_upwrapped.append(theta_2_first_round)
    total_rounds_unwrapped.append(theta_2_chaos_count)


print(first_round_upwrapped, total_rounds_unwrapped)


