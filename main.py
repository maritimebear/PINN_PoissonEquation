"""
2D square Poisson equation using PINN
Geometry: [0, 1] x [0, 1], non-dimensional units

Governing equation:
    -(u_xx + u_yy) = f in domain
    u = g on all boundaries

    f(x, y) = (2 * pi**2) * cos(pi * x) * cos(pi * y)
    g(x, y) = cos(pi * x) * cos(pi * y)
"""

import network
import dataset
import trainers
import loss
import sampler
import physics
import test_metrics
import plotters
import plot_logger

import torch
import matplotlib.pyplot as plt
import numpy as np

from typing import Sequence, TypeAlias, Union
Tensor: TypeAlias = Union[np.ndarray, torch.Tensor]

# Is it possible to create a "main_trainer" class and set an RNG engine for
# each instance of that class?
# If yes, is it possible to create an array of such objects with different
# n_data_samples and train all of those in parallel?
torch.manual_seed(7673345)
torch.set_default_dtype(torch.float64)  # Double precision required for L-BFGS? https://stackoverflow.com/questions/73019486/torch-optim-lbfgs-does-not-change-parameters
plt.ioff()
PI = np.pi

# Parameters
batch_size = 64
n_data_samples = 1024
n_residual_points = 10_000
n_boundary_points = 1000

# Learning rate and decay control
lr_Adam = 1e-3
lr_decay_exp = 1 - 1e-4  # Exponent for exponential learning rate decay

# Domain definition
extents_x = (0.0, 1.0)
extents_y = (0.0, 1.0)

# Loss weights
w_dataloss = 0.0
w_residualloss = 1.0
w_boundaryloss = 1.0

# Training loop control
n_epochs = 10_000  # Use with for-loop
converged = False  # Use with while-loop with convergence control
convergence_threshold = 1e0
convergence_sustain_duration = 10
n_converged = 0


# Grid for plotting residuals and fields during testing
test_gridspacing = 100
_gridx, _gridy = torch.meshgrid(*[torch.linspace(*ext, test_gridspacing, requires_grad=True) for ext in (extents_x, extents_y)], indexing='xy')
res_domain = torch.hstack( [t.flatten()[:, None] for t in (_gridx, _gridy)] )

# Set up model
model = network.FCN(2,  # inputs: x, y
                    1,  # outputs: u
                    32,  # number of neurons per hidden layer
                    4)  # number of hidden layers

# Set up optimiser
optimiser_Adam = torch.optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler_Adam = torch.optim.lr_scheduler.ExponentialLR(optimiser_Adam, gamma=lr_decay_exp)

# Set up losses
lossfn_data = loss.WeightedScalarLoss(torch.nn.MSELoss(), weight=w_dataloss)
lossfn_residual = loss.WeightedScalarLoss(torch.nn.MSELoss(), weight=w_residualloss)
lossfn_boundary = loss.WeightedScalarLoss(torch.nn.MSELoss(), weight=w_boundaryloss)

# Set up trainers
# Data trainer
sampling_idxs = np.random.randint(0, 1024**2, n_data_samples)  # 1024**2 == total number of ground truth samples from multigrid solution
ds = dataset.Interior_Partial_Dataset("./data.csv", ["x", "y"], ["u"], sampling_idxs=sampling_idxs)
dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
trainer_data = trainers.DataTrainer(model, loss_fn=lossfn_data)

# Residual trainer
sampler_residual = sampler.UniformRandomSampler(n_points=n_residual_points, extents=[extents_x, extents_y])
residual_fn = physics.PoissonEquation()
trainer_residual = trainers.ResidualTrainer(sampler_residual, model, residual_fn, lossfn_residual)

# Boundary trainers
bottom, top = [ [extents_x, (y, y)] for y in extents_y[:] ]
left, right = [ [(x, x), extents_y] for x in extents_x[:] ]
samplers_boundaries = [sampler.UniformRandomSampler(n_points=n_boundary_points,
                                                    extents=ext) for
                       ext in (bottom, right, top, left)]

# Dirichlet BC
boundary_fn = lambda x: (torch.cos(PI * x[:, 0]) * torch.cos(PI * x[:, 1])).reshape(x.shape[0], -1)

trainers_boundaries = [trainers.BoundaryTrainer(sampler, model, boundary_fn, lossfn_boundary) for
                       sampler in samplers_boundaries]


# Test-metrics
pred_plotter = test_metrics.PredictionPlotter(extents_x, test_gridspacing, extents_y, test_gridspacing)
error_calculator = test_metrics.PoissonErrorCalculator(dataset.PINN_Dataset("./data.csv", ["x", "y"], ["u"]))

# Setup plot-loggers for loss and error curves
# Track losses and norms of error and residual through dicts
# TODO: dict-key-lookup impact on performance?
losses_dict = {key: list() for key in ("data", "residual", "boundaries", "total")}
errors_dict = {key: list() for key in ("l2", "max")}
residuals_dict = {key: list() for key in ("l2", "max")}

logger_loss = plot_logger.Plot_and_Log_Scalar("losses", losses_dict,
                                              plot_xlabel="Iteration", plot_ylabel="Loss", plot_title="Loss curves")
logger_error = plot_logger.Plot_and_Log_Scalar("absolute_error", errors_dict,
                                               plot_xlabel="Epoch", plot_ylabel="||error||", plot_title="Absolute error at test points")
logger_residual = plot_logger.Plot_and_Log_Scalar("residuals", residuals_dict,
                                                  plot_xlabel="Epoch", plot_ylabel="||residual||", plot_title="Residuals at test points")


def train_iteration(optimiser, step: bool, lr_scheduler=None) -> torch.Tensor:
    # Can be used as closure function for L-BFGS
    # Returns total loss (scalar)
    # step: whether or not to optimiser.step()
    for batch in dataloader:
        optimiser.zero_grad()
        # Data loss
        x, y = [tensor.to(torch.get_default_dtype()) for tensor in batch]  # Network weights have dtype torch.float32
        loss_data = trainer_data(x, y)
        # Residual loss
        loss_residual = trainer_residual()
        # Boundary losses
        loss_boundaries = sum([trainer() for trainer in trainers_boundaries])  # Not considering each boundary separately
        # Total loss
        loss_total = loss_data + loss_residual + loss_boundaries
        loss_total.backward()
        if step:
            optimiser.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Append losses to corresponding lists in dict
        for key, _loss in zip(["data", "residual", "boundaries", "total"],
                              [loss_data, loss_residual, loss_boundaries, loss_total]):
            losses_dict[key].append(_loss.detach())
        logger_loss.update_log()

    return loss_total  # For future L-BFGS compatibility


def test() -> tuple[Sequence[Tensor], float]:
    # Calculate error and residual fields over a fixed set of points in the domain
    # Returns a tuple of resulting tensors and a metric of the error/residual (a norm or a mean),
    # to determine solution convergence

    # Calculate errors, update error logger
    error = error_calculator(model)
    errors_dict["l2"].append(np.linalg.norm(error.flatten()))
    errors_dict["max"].append(np.linalg.norm(error.flatten(), ord=np.inf))
    logger_error.update_log()

    # Residuals
    res_test_fn = physics.PoissonEquation()
    u_h = model(res_domain)  # res_domain from main namespace
    residuals = res_test_fn(u_h, res_domain)
    residuals_dict["l2"].append(np.linalg.norm(residuals.detach().numpy()))
    residuals_dict["max"].append(np.linalg.norm(residuals.detach().numpy(), ord=np.inf))
    logger_residual.update_log()

    convergence_control = residuals_dict["max"][-1]  # Using inf-norm of residuals for convergence control

    return ([u_h, error, residuals], convergence_control)


def plot(u_h, error, residuals) -> None:
    # Update scalar plots
    _ = [logger.update_plot() for logger in (logger_loss, logger_error, logger_residual)]

    # Prediction surface plot
    fig_predcf = plt.figure(figsize=(8, 8))
    ax_predsurf = fig_predcf.add_subplot(1, 1, 1, projection="3d")
    ax_predsurf = pred_plotter(ax_predsurf, model)
    ax_predsurf.set_xlabel("x")
    ax_predsurf.set_ylabel("y")
    ax_predsurf.set_zlabel("u\u0302")  # \u0302: unicode for hat symbol
    ax_predsurf.set_title("Predicted solution")

    # Plot error contours
    fig_errorcf = plt.figure(figsize=(8, 8))
    ax_errorcf = fig_errorcf.add_subplot(1, 1, 1)
    ax_errorcf = plotters.contourf(ax_errorcf,
                                   error_calculator.x, error_calculator.y, error.reshape(error_calculator.x.shape),
                                   xlabel="x", ylabel="y", fieldlabel="error", title="Absolute error in prediction")

    # Residuals contour plot
    fig_rescf = plt.figure(figsize=(8, 8))
    ax_rescf = fig_rescf.add_subplot(1, 1, 1)
    ax_rescf = plotters.contourf(ax_rescf,
                                 *[t.detach().numpy() for t in (_gridx, _gridy, residuals.reshape(_gridx.shape))],
                                 xlabel="x", ylabel="y", fieldlabel="residual", title="Residual field")

    # Prediction contour plot
    fig_predcf = plt.figure(figsize=(8, 8))
    ax_predcf = fig_predcf.add_subplot(1, 1, 1)
    ax_predcf = plotters.contourf(ax_predcf,
                                  *[t.detach().numpy() for t in (_gridx, _gridy, u_h.reshape(_gridx.shape))],
                                  xlabel="x", ylabel="y", fieldlabel="u\u0302", title="Predicted solution")

    plt.show()


# for-loop to train for a specified number of epochs
# epochs wrt dataset size and batch size of ground truth data
# for i in range(n_epochs):
    # print(f"Epoch: {i}")
    # _ = train_iteration(optimiser_Adam, step=True, lr_scheduler=lr_scheduler_Adam)  # Discard return value, losses appended to lists
    # test_tensors, _ = test()  # Discard convergence control in for-loop
    # plot(*test_tensors)

# while-loop to train until converged wrt convergence control returned by test()
epoch_ctr = 0
while not converged:
    _ = train_iteration(optimiser_Adam, step=True, lr_scheduler=lr_scheduler_Adam)  # Discard return value, losses appended to lists
    test_tensors, convergence_control = test()

    epoch_ctr += 1
    # plot(*test_tensors)

    if convergence_control <= convergence_threshold:  # threshold defined in main namespace
        n_converged += 1
        print(f"Epoch: {epoch_ctr}\t" +
              f"Convergence control: {convergence_control}\t" +
              f"Threshold: {convergence_threshold}\t" +
              f"Remaining: {convergence_sustain_duration - n_converged}")
    else:
        n_converged = 0
        print(f"Epoch: {epoch_ctr}\t" +
              f"Convergence control: {convergence_control}\t" +
              f"Threshold: {convergence_threshold}")

    if n_converged >= convergence_sustain_duration:
        converged = True
        print(f"Training converged in {epoch_ctr} epochs")

# Plot final results
final_test, _ = test()
plot(*final_test)
