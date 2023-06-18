# Script to plot data.csv for comparison with PINN results

import numpy as np
import matplotlib.pyplot as plt
import dataset

ds = dataset.PINN_Dataset("./data.csv", ["x", "y"], ["u"])

inputs, output = ds[:]

assert output.squeeze().ndim == 1  # Confirm output is a vector
n = int(np.sqrt(len(output)))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
cs = ax.contourf(*[arr.reshape(n, n) for arr in (inputs[:, 0], inputs[:, 1], output)])
cbar = fig.colorbar(cs)

ax.set_title("Reference solution (ground truth)\n" +
             r"$ -\Delta u(x,y) = 2 \pi ^{2} cos(\pi x) cos(\pi y) $ in $ \Omega $" + "\n" +
             r"$ u(x, y) = cos(\pi x) cos(\pi y) $ on $ \partial \Omega $" + "\n" +
             r"$ \Omega = (0, 1) \times (0, 1) $"
             )
ax.set_xlabel("x")
ax.set_ylabel("y")
cbar.set_label("u")

plt.show()
