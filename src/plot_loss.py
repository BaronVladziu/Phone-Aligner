#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

# Settings
log_path = 'train.log'
smooth_order = 100
smoothing_runs = 1

# Load log
loss = np.zeros(0)
with open(log_path) as log_file:
    for line in log_file:
        loss = np.concatenate([
            loss,
            [float(line.split()[7])]
        ])
x = np.arange(len(loss))

# Compute smoothed loss
pre_smoothed_loss = np.array(loss)
for j in range(smoothing_runs):
    smoothed_loss = np.zeros(len(loss))
    for i in range(len(loss)):
        if i < smooth_order:
            smoothed_loss[i] = np.mean(
                pre_smoothed_loss[:i+1]
            )
        else:
            smoothed_loss[i] = np.mean(
                pre_smoothed_loss[i-smooth_order:i+1]
            )
    pre_smoothed_loss = np.array(smoothed_loss)

# Plot loss
plt.figure()
plt.plot(x, loss, 'b-', x, smoothed_loss, 'r-')
plt.grid()
plt.yscale('log')
plt.show()
