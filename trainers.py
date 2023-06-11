import torch
from typing import Protocol, TypeAlias, Callable

Tensor: TypeAlias = torch.Tensor


class Trainer(Protocol):
    def __call__(self) -> Tensor: ...


class DataTrainer():
    # Callable class, calculates data loss from ground truth
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn

    def __call__(self, x, y) -> Tensor:
        # x: input, y: ground truth at x
        prediction = self.model(x)
        return self.loss_fn(prediction, y)


class PhysicsTrainer():
    # Callable class, calculates physics loss wrt residuals
    def __init__(self,
                 sampler: Callable[[], Tensor],
                 model: Callable[[Tensor], Tensor],
                 residual_fn: Callable[[Tensor], Tensor],
                 loss_fn: Callable[[Tensor], Tensor]) -> None:
        self.sampler = sampler
        self.model = model
        self.residual_fn = residual_fn
        self.loss_fn = loss_fn

    def __call__(self) -> Tensor:
        domain = self.sampler()
        prediction = self.model(domain)
        residuals = self.residual_fn(prediction, domain)
        return self.loss_fn(residuals)  # Return loss


# class BoundaryTrainer 
