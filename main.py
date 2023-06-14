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
w_boundaryloss = 1.0

# Grid for plotting residuals and fields during testing
test_gridspacing = 100

# Set up model
model = network.FCN(2,  # inputs: x, y
                    1,  # outputs: u
                    64,  # number of neurons per hidden layer
                    4)  # number of hidden layers

optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
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

# Training loop
n_epochs = 100

# Lists to store losses/errors
loss_total_list = list()
loss_data_list = list()
loss_residual_list = list()
loss_boundaries_list = list()
epoch_error_l2 = list()
epoch_error_inf = list()

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
        loss_boundaries = sum([trainer() for trainer in trainers_boundaries])  # Not considering each boundary separately

        loss_total = loss_data + loss_residual + loss_boundaries

        loss_total.backward()
        optimiser.step()
        
        # Append losses to lists
        for _loss, _list in zip([loss_data, loss_residual, loss_boundaries, loss_total],
                                   [loss_data_list, loss_residual_list, loss_boundaries_list, loss_total_list]):
            _list.append(_loss.detach())
            
        if nbatch == 1000:
            break

    # loss_total_list.append(torch.stack(loss_list).mean().item())
    
    # Post-processing        
    error = error_calculator(model)
    epoch_error_l2.append(np.linalg.norm(error.flatten()))
    epoch_error_inf.append(np.linalg.norm(error.flatten(), ord=np.inf))
    
    # Plotting
    fig_loss, ax_loss = plt.subplots(1, 1, figsize=(4, 4))
    for _list, label in zip([loss_data_list, loss_residual_list, loss_boundaries_list, loss_total_list],
                            ["Data", "Residual", "Boundaries", "Total"]):
        ax_loss = plotters.semilogy_plot(ax_loss, _list, label=label, xlabel="Iterations", ylabel="Loss", title="Losses")
    


    # fig1 = plt.figure(figsize=(4, 4))
    # ax_loss = fig1.add_subplot(1, 1, 1)
    # ax_loss.semilogy(loss_total_list)


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
    
    
    # ax_error_l2norm.semilogy(epoch_error_l2)
    cs = ax_errorcontours.contourf(error_calculator.x, error_calculator.y,
                                   error.reshape(error_calculator.x.shape))
    fig2.colorbar(cs)
    ax_surf = pred_plotter(ax_surf, model)
    # ax_error_infnorm.semilogy(epoch_error_inf)
    
    fig2.tight_layout()

    plt.show()
