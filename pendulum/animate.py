#!/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
    parser = argparse.ArgumentParser(description='Animate double pendulum simulation data.')
    parser.add_argument('index', type = str, help='CSV file index')
    parser.add_argument('--fast-forward', type=float, default=1.0, help='Fast forward factor for the animation.')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second for the animation.')
    args = parser.parse_args()

    cartesian_filepath = 'real_data/cartesian/cartesian_'
    polar_filepath = 'real_data/polar/polar_'

    cartesian_filename = cartesian_filepath + args.index + '.csv'
    polar_filename = polar_filepath + args.index + '.csv'

    fast_forward = args.fast_forward
    fps = args.fps

    t_array, theta1_array, omega1_array, theta2_array, omega2_array = \
        read_csv(polar_filename)
    _, x1_array, y1_array, x2_array, y2_array = read_csv_cartesian(cartesian_filename)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    fig.set_tight_layout(True)
    axes[0].set_aspect('equal')
    xmin = min([x1_array.min(), x2_array.min()])
    xmax = max([x1_array.max(), x2_array.max()])
    ymin = min([y1_array.min(), y2_array.min()])
    ymax = max([y1_array.max(), y2_array.max(), 0.0])
    axes[0].set_xlim(xmin - 0.2, xmax + 0.2)
    axes[0].set_ylim(ymin - 0.2, ymax + 0.2)

    axes[1].set_aspect('equal')
    theta1_min = min(theta1_array)
    theta1_max = max(theta1_array)
    theta2_min = min(theta2_array)
    theta2_max = max(theta2_array)
    axes[1].set_xlim(theta1_min - 0.2, theta1_max + 0.2)
    axes[1].set_ylim(theta2_min - 0.2, theta2_max + 0.2)
    axes[1].set_xlabel(r'$\theta_1$ [rad]')
    axes[1].set_ylabel(r'$\theta_2$ [rad]')

    axes[2].set_aspect('equal')
    omega1_min = min(omega1_array)
    omega1_max = max(omega1_array)
    omega2_min = min(omega2_array)
    omega2_max = max(omega2_array)
    axes[2].set_xlim(omega1_min - 0.2, omega1_max + 0.2)
    axes[2].set_ylim(omega2_min - 0.2, omega2_max + 0.2)
    axes[2].set_xlabel(r'$\omega_1$ [rad/s]')
    axes[2].set_ylabel(r'$\omega_2$ [rad/s]')

    pendulum_line, = axes[0].plot([], [], 'o-', lw=2)
    phase_line1, = axes[1].plot([], [], lw=1, color='orange')
    phase_line2, = axes[2].plot([], [], lw=1, color='green')
    time_text = axes[0].text(0.02, 0.95, '', transform=axes[0].transAxes)

    def init():
        pendulum_line.set_data([], [])
        phase_line1.set_data([], [])
        phase_line2.set_data([], [])
        time_text.set_text('')
        return pendulum_line, time_text, phase_line1, phase_line2

    def update(frame):
        # index = int(frame / (fps * dt))
        index = int(frame / fps * fast_forward / dt)
        # index = np.argmin(abs(t_array - frame / fps * fast_forward))

        x = [0, x1_array[index], x2_array[index]]
        y = [0, y1_array[index], y2_array[index]]
        theta1 = theta1_array[:index]
        theta2 = theta2_array[:index]
        omega1 = omega1_array[:index]
        omega2 = omega2_array[:index]
        pendulum_line.set_data(x, y)
        phase_line1.set_data(theta1, theta2)
        phase_line2.set_data(omega1, omega2)
        time_text.set_text(f'time = {t_array[index]:.2f} s')
        return pendulum_line, time_text, phase_line1, phase_line2

    # fast_forward = 1
    # fps = 20
    interval = int(1000 / fps) # milliseconds
    dt = (t_array[-1] - t_array[0]) / (len(t_array) - 1)

    ani = FuncAnimation(fig, update, frames=int(len(t_array) *  dt * fps / fast_forward),
                        init_func=init, blit=True, interval=interval, repeat=False)

    plt.show()

def read_csv(filename):
    trajectory = np.loadtxt(filename, delimiter=',', skiprows=1)
    t_array = trajectory[:, 0]
    theta1_array = trajectory[:, 1]
    omega1_array = trajectory[:, 2]
    theta2_array = trajectory[:, 3]
    omega2_array = trajectory[:, 4]
    return t_array, theta1_array, omega1_array, theta2_array, omega2_array

def read_csv_cartesian(filename):
    trajectory = np.loadtxt(filename, delimiter=',', skiprows=1)
    t_array = trajectory[:, 0]
    x1_array = trajectory[:, 1]
    y1_array = trajectory[:, 2]
    x2_array = trajectory[:, 3]
    y2_array = trajectory[:, 4]
    return t_array, x1_array, y1_array, x2_array, y2_array


if __name__ == "__main__":
    main()
