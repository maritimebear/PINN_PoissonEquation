import torch

from typing import TypeAlias, Callable

Tensor: TypeAlias = torch.Tensor


class PoissonEquation():
    """
    Callable class, returns residuals of Poisson Equation in a
    2-D rectangular domain:

        -(u_xx + u_yy) = f
    """
    def __init__(self, source_fn: Callable[[Tensor], Tensor]) -> None:
        self.source_fn = source_fn
        self._ones = None

    def _grad(self, f: Tensor, x: Tensor) -> Tensor:
        if self._ones is None:
            self._ones = torch.ones_like(f)
        return torch.autograd.grad(f, x, grad_outputs=self._ones, create_graph=True)[0]

    def __call__(self, prediction: Tensor, domain: Tensor) -> Tensor:
        # prediction: Tensor[u]
        # domain: Tensor[x, y]
        # Return: residual = -(u_xx + u_yy) - f
        u_x, u_y = [self._grad(prediction, domain)[:, col].reshape(prediction.shape[0], -1)
                    for col in (0, 1)]
        # Reshape to stop pytorch complaining about shape mismatch
        u_xx = self._grad(u_x, domain)[:, 0]
        u_yy = self._grad(u_y, domain)[:, 1]
        return ( -(u_xx - u_yy) - self.source_fn(domain) )
