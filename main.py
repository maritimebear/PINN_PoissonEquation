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

import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(767)
plt.ioff()
PI = np.pi

# Parameters
batch_size = 64
n_data_samples = 1024
n_residual_points = 1000
n_boundary_points = 1000

extents_x = (0.0, 1.0)
extents_y = (0.0, 1.0)

# Loss weights
w_dataloss = 1.0
w_residualloss = 1.0
w_boundaryloss = 2.0

# Grid for plotting residuals and fields during testing
test_gridspacing = 100
# Nx, Ny = (100, 100) # Nx and Ny must multiply to produce batch_size


# grid_x, grid_y = torch.meshgrid(torch.linspace(*extents_x, Nx, requires_grad=False),
                                # torch.linspace(*extents_y, Ny, requires_grad=False),
                                # indexing='xy')

# batched_domain = torch.hstack([grid_x.flatten()[:,None], grid_y.flatten()[:,None]])


# Set up model
model = network.FCN(2,  # inputs: x, y
                    1,  # outputs: u
                    64,  # number of neurons per hidden layer
                    4)  # number of hidden layers

optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimiser = torch.optim.SGD(model.parameters(), lr=1e-2)


# Set up losses
lossfn_data = loss.WeightedScalarLoss(torch.nn.MSELoss(), weight=w_dataloss)
lossfn_residual = loss.WeightedScalarLoss(torch.nn.MSELoss(), weight=w_residualloss)
lossfn_boundary = loss.WeightedScalarLoss(torch.nn.MSELoss(), weight=w_boundaryloss)


# Set up trainers

# Data trainer
# ds = dataset.PINN_Dataset("./data.csv", ["x", "y"], ["u"])
sampling_idxs = np.random.randint(0, 1024**2, n_data_samples)
ds = dataset.Interior_Partial_Dataset("./data.csv", ["x", "y"], ["u"], sampling_idxs=sampling_idxs)
dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
trainer_data = trainers.DataTrainer(model, loss_fn=lossfn_data)

# Residual trainer
sampler_residual = sampler.UniformRandomSampler(n_points=n_residual_points, extents=[extents_x, extents_y])
# Source term of Poisson equation
poisson_source = lambda x: ((2 * PI**2) * torch.cos(PI * x[:, 0]) * torch.cos(PI * x[:, 1])).reshape(x.shape[0], -1)
residual_fn = physics.PoissonEquation(poisson_source)
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

# Convenience variables for plotting and monitoring
# batched_domain = torch.hstack([torch.linspace(*extents_x, test_gridspacing),
#                                torch.linspace(*extents_y, test_gridspacing)])

# Training loop
n_epochs = 100
epoch_loss = torch.zeros((n_epochs,))
epoch_error = list()

for i in range(n_epochs):
    loss_list = list()

    for nbatch, batch in enumerate(dataloader):
        print(f"{nbatch=}")

        optimiser.zero_grad()

        # Data loss
        x, y = [tensor.float() for tensor in batch]  # Network weights have dtype torch.float32
        loss_data = trainer_data(x, y)
        # Residual loss
        loss_residual = trainer_residual()

        # Boundary losses
        losses_boundaries = [trainer() for trainer in trainers_boundaries]

        loss_total = loss_data + loss_residual + sum(losses_boundaries)

        loss_total.backward()
        optimiser.step()
        loss_list.append(loss_total.detach())
        if nbatch == 1000:
            break

    epoch_loss[i] = torch.stack(loss_list).mean().item()

    # test_pred = model(batched_domain)

    fig1 = plt.figure(figsize=(4, 8))
    ax_loss = fig1.add_subplot(2, 1, 1)
    ax_surf = fig1.add_subplot(2, 1, 2, projection="3d")
    ax_loss.semilogy(epoch_loss)
    # ax_surf.plot_surface(error_calculator.x, error_calculator.y, model(batched_domain).reshape_as(error_calculator.x).detach())
    ax_surf = pred_plotter(ax_surf, model)

    # Calculate and plot error in prediction
    fig2 = plt.figure(figsize=(4, 8))
    ax_errornorm = fig2.add_subplot(2, 1, 1)
    ax_errorcontours = fig2.add_subplot(2, 1, 2)

    error = error_calculator(model)
    epoch_error.append(np.linalg.norm(error))
    ax_errornorm.semilogy(epoch_error)
    cs = ax_errorcontours.contourf(error_calculator.x, error_calculator.y,
                                   error.reshape(error_calculator.x.shape))
    fig2.colorbar(cs)

    plt.show()
