#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

T = 2
f = np.arange(-2.0, 2.0, 0.01)
x = np.sin(2 * np.pi * T * f) / (np.pi * f)

fig, ax = plt.subplots()
ax.plot(f, x)

ax.set(xlabel='f', ylabel='A',)
ax.grid()

fig.savefig('images/plot4.png')
