#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:20:23 2024

@author: florentcalvayrac
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
N = 16  # Grid size for N x N
T = 0.05  # Temperature
steps = 1  # Total number of animation frames
steps_per_spin = 1  # Monte Carlo steps per spin
total_steps = N * N * steps_per_spin


# Initial configuration
spins = np.random.choice([-1, 1], size=(N, N))

def delta_energy(spins, i, j, N):
    """Calculate the energy change from flipping a spin at position (i, j)."""
    total_spin = 0
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = (i + di) % N, (j + dj) % N
        total_spin += spins[ni, nj]
    return 2 * spins[i, j] * total_spin

def update(frame, spins, img, N, T):
    """Perform a single Monte Carlo update and refresh the image."""
    for _ in range(total_steps):
        for _ in range(N*N):
            i, j = np.random.randint(0, N, 2)
            dE = delta_energy(spins, i, j, N)
            if dE <= 0 or np.random.rand() < np.exp(-dE / T):
                spins[i, j] *= -1
    img.set_data(spins)
    return img,

fig, ax = plt.subplots()
img = ax.imshow(spins, cmap='coolwarm', interpolation='nearest')
ani = animation.FuncAnimation(fig, update, fargs=(spins, img, N, T), frames=steps, interval=50, blit=False)

plt.show()
