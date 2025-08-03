#!/bin/env python

import sys
from math import pi as M_PI
import numpy as np
import time

from double_pendulum import DoublePendulum

def update_progress_bar(current, total, bar_length=40):
    progress = (current + 1) / total
    filled_length = int(bar_length * progress)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    print(f"\rProgress: |{bar}| {current + 1}/{total} simulations", end='', flush=True)

def main():

    if len(sys.argv) > 1:
        num_sims = int(sys.argv[1])
    else:
        print("Usage: python generate.py num_sims")
        sys.exit(1)

    num_sims = max(1, num_sims)      # Ensure at least one simulation
    dt = 0.01                        # Time step for the simulation
    max_time = 60.0                  # Maximum time for each simulation
    time_steps = int(max_time / dt)  # Number of time steps

    num_digits = len(str(num_sims - 1))  # Number of digits for zero-padding

    start = time.time()

    l1 = 1.0  # Length of the first pendulum
    l2 = 1.0  # Length of the second pendulum
    m1 = 1.0  # Mass of the first pendulum
    m2 = 1.0  # Mass of the second pendulum

    for sim in range(num_sims):
        dp = DoublePendulum(l1, l2, m1, m2)
        theta01, theta02 = np.random.random(size=2) * 2 * M_PI  # Random initial angles
        dp.set_initial_conditions([theta01, 0.0, theta02, 0.0])
        dp.simulate(dt, time_steps)
        dp.write_trajectory_to_csv(f"real_data/polar/polar_{sim:0{num_digits}d}.csv")
        dp.write_trajectory_to_csv_cartesian(f"real_data/cartesian/cartesian_{sim:0{num_digits}d}.csv")

        update_progress_bar(sim, num_sims)

    print()  # Move to next line after progress bar

    end = time.time()
    print(f"Total simulation time: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
