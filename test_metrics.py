import torch
import numpy as np

from dataset import PINN_Dataset  # for type-checking

from typing import TypeVar

Axes = TypeVar("Axes")


class PoissonErrorCalculator():
    """
    Class to calculate error in model prediction for Poisson Equation
    Takes input and ground truth values from Dataset class, requires that data is ordered
        i.e. input coordinates must be sorted
    """
    def __init__(self,
                 dataset: PINN_Dataset) -> None:
        n = int(np.sqrt(len(dataset)))
        assert np.isclose(n, np.sqrt(len(dataset)))  # len(dataset) must be a perfect square?
        self._inputs, self._output = dataset[:]  # Assumes dataset is of type PINN_Dataset
        # Reshape to meshgrids for plotting
        self.x, self.y = [arr.reshape(n, n) for arr in (self._inputs[:, 0], self._inputs[:, 1])]
        self._inputs = torch.from_numpy(self._inputs).float()  # self._inputs to be passed to model

    def __call__(self, model) -> np.ndarray:
        # Returns absolute error in prediction
        # TODO: Add tensorboard hooks to arguments for plotting?
        # u_hat = model(self._inputs).detach().numpy()
        # error = self._output - u_hat
        return self._output - model(self._inputs).detach().numpy()


class PredictionPlotter():
    """
    Class to evaluate model over a fixed test domain and plot the results, for visualisation of
    training progress.
    """
    def __init__(self,
                 extents_x: tuple[float, float],
                 spacing_x: float,
                 extents_y: tuple[float, float],
                 spacing_y: float) -> None:
        self._grid = torch.meshgrid(torch.linspace(*extents_x, spacing_x),
                                 torch.linspace(*extents_y, spacing_y),
                                 indexing='xy')
        self._batched_domain = torch.hstack([t.flatten()[:, None] for t in self._grid])
        self._grid = [t.detach().numpy() for t in self._grid]  # Convert from torch to numpy for plotting

    def __call__(self, ax: Axes, model) -> Axes:
        ax.plot_surface(self._grid[0], 
                        self._grid[1],
                        model(self._batched_domain).detach().numpy().reshape(self._grid[0].shape))
        return ax
