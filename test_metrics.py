import numpy as np

from dataset import PINN_Dataset  # for type-checking

from typing import Callable, Sequence


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

    def __call__(self,
                 model,
                 line_plotter: Callable[[float], None],
                 meshgrid_plotter: Callable[[Sequence[np.ndarray]], None]) -> None:
        # TODO: Add tensorboard hooks to arguments for plotting?
        u_hat = model(self._inputs).numpy()
        error = self._output - u_hat
        line_plotter(np.linalg.norm(error))
        meshgrid_plotter(self.x, self.y, error.reshape(self.x.shape))
