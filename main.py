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

import torch
import matplotlib.pyplot as plt
import numpy as np

# Is it possible to create a "main_trainer" class and set an RNG engine for
# each instance of that class?
# If yes, is it possible to create an array of such objects with different
# n_data_samples and train all of those in parallel?
torch.manual_seed(7673345)
plt.ioff()
PI = np.pi

# Parameters
batch_size = 64
n_data_samples = 1024
n_residual_points = 10_000
n_boundary_points = 1000

extents_x = (0.0, 1.0)
extents_y = (0.0, 1.0)

# Loss weights
w_dataloss = 0.0
w_residualloss = 1.0
w_boundaryloss = 10.0

# Grid for plotting residuals and fields during testing
test_gridspacing = 100

# Set up model
model = network.FCN(2,  # inputs: x, y
                    1,  # outputs: u
                    32,  # number of neurons per hidden layer
                    4)  # number of hidden layers

optimiser_Adam = torch.optim.Adam(model.parameters(), lr=1e-3)

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

# Training loop
n_epochs = 10_000

# Lists to store losses/errors
loss_total_list = list()
loss_data_list = list()
loss_residual_list = list()
loss_boundaries_list = list()
epoch_error_l2 = list()
epoch_error_inf = list()

# TODO: Remove after fixing residuals
_res_list = list()

def train_iteration(optimiser, step: bool) -> torch.Tensor:
    # Can be used as closure function for L-BFGS
    # Returns total loss (scalar)
    # step: whether or not to optimiser.step()
    for batch in dataloader:
        optimiser.zero_grad()
        # Data loss
        x, y = [tensor.float() for tensor in batch]  # Network weights have dtype torch.float32
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
        # Append losses to lists
        for _loss, _list in zip([loss_data, loss_residual, loss_boundaries, loss_total],
                                   [loss_data_list, loss_residual_list, loss_boundaries_list, loss_total_list]):
            _list.append(_loss.detach())

    return loss_total  # For future L-BFGS compatibility

def postprocess():
    # Calculate error, test steps and plotting
    error = error_calculator(model)
    epoch_error_l2.append(np.linalg.norm(error.flatten()))
    epoch_error_inf.append(np.linalg.norm(error.flatten(), ord=np.inf))

    # Plotting
    fig_loss, ax_loss = plt.subplots(1, 1, figsize=(4, 4))
    for _list, label in zip([loss_data_list, loss_residual_list, loss_boundaries_list, loss_total_list],
                            ["Data", "Residual", "Boundaries", "Total"]):
        ax_loss = plotters.semilogy_plot(ax_loss, _list, label=label, xlabel="Iterations", ylabel="Loss", title="Losses")

    # Calculate and plot error in prediction
    fig2 = plt.figure(figsize=(8, 8))
    ax_error_l2norm = fig2.add_subplot(2, 2, 1)
    ax_errorcontours = fig2.add_subplot(2, 2, 2)
    ax_surf = fig2.add_subplot(2, 2, 3, projection="3d")
    ax_error_infnorm = fig2.add_subplot(2, 2, 4)

    ax_error_l2norm, ax_error_infnorm = [plotters.semilogy_plot(ax, errorlist, xlabel="Iteration", ylabel="||error||", title=title) for
                                         ax, errorlist, title in zip([ax_error_l2norm, ax_error_infnorm],
                                                                     [epoch_error_l2, epoch_error_inf],
                                                                     ["L2 norm of error", "inf-norm of error"])]

    cs = ax_errorcontours.contourf(error_calculator.x, error_calculator.y,
                                   error.reshape(error_calculator.x.shape))
    fig2.colorbar(cs)
    ax_surf = pred_plotter(ax_surf, model)
    fig2.tight_layout()

    # TODO: Remove after fixing residual errors not reducing
    # Track residual learning
    _res_test_fn = physics.PoissonEquation()
    _gridx, _gridy = torch.meshgrid(*[torch.linspace(*ext, 100, requires_grad=True) for ext in (extents_x, extents_y)], indexing='xy')

    _res_domain = torch.hstack( [t.flatten()[:, None] for t in (_gridx, _gridy)] )
    u_h = model(_res_domain)

    _residuals = _res_test_fn(u_h, _res_domain)
    _res_list.append(np.linalg.norm(_residuals.detach().numpy()))


    fig_res = plt.figure(figsize=(4, 8))
    ax_resnorm = fig_res.add_subplot(2, 1, 1)
    ax_rescontours = fig_res.add_subplot(2, 1, 2)
    ax_resnorm = plotters.semilogy_plot(ax_resnorm, _res_list, xlabel="Epochs", ylabel="||res||", title="L2 norm of residuals")

    cs_r = ax_rescontours.contourf(_gridx.detach().numpy(), _gridy.detach().numpy(), _residuals.reshape(_gridx.shape).detach().numpy())
    fig_res.colorbar(cs_r)

    fig_pred = plt.figure(figsize=(8,4))
    ax_pred = fig_pred.add_subplot(1, 2, 1)
    cs_pred = ax_pred.contourf(_gridx.detach().numpy(), _gridy.detach().numpy(), u_h.reshape(_gridx.shape).detach().numpy())
    fig_pred.colorbar(cs_pred)
    ax_predsurf = fig_pred.add_subplot(1, 2, 2, projection="3d")
    ax_predsurf = pred_plotter(ax_predsurf, model)

    plt.show()

for i in range(n_epochs):
    print(f"Epoch: {i}")
    _ = train_iteration(optimiser_Adam, step=True)  # Discard return value, losses appended to lists
    postprocess()



