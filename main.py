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

import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(767)
plt.ioff()


# Parameters
batch_size = 64
n_data_samples = 1024

extents_x = (0.0, 1.0)
extents_y = (0.0, 1.0)

# Loss weights
w_dataloss = 1.0
w_residualloss = 1.0
w_boundaryloss = 1.0

# Grid for plotting residuals and fields
Nx, Ny = (100, 100) # Nx and Ny must multiply to produce batch_size


grid_x, grid_y = torch.meshgrid(torch.linspace(*extents_x, Nx, requires_grad=False),
                                torch.linspace(*extents_y, Ny, requires_grad=False),
                                indexing='xy')

batched_domain = torch.hstack([grid_x.flatten()[:,None], grid_y.flatten()[:,None]])


# Set up model
model = network.FCN(2,  # inputs: x, y
                    1,  # outputs: u
                    256,  # number of neurons per hidden layer
                    4)  # number of hidden layers

optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimiser = torch.optim.SGD(model.parameters(), lr=1e-2)


# Set up losses
lossfn_data = loss.WeightedScalarLoss(torch.nn.MSELoss(), weight=w_dataloss)
lossfn_residual = loss.WeightedScalarLoss(torch.nn.MSELoss(), weight=w_residualloss)
lossfn_boundary = loss.WeightedScalarLoss(torch.nn.MSELoss(), weight=w_boundaryloss)


# Set up data trainer
# ds = dataset.PINN_Dataset("./data.csv", ["x", "y"], ["u"])
sampling_idxs = np.random.randint(0, 1024**2, n_data_samples)
ds = dataset.Interior_Partial_Dataset("./data.csv", ["x", "y"], ["u"], sampling_idxs=sampling_idxs)
dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
data_trainer = trainers.DataTrainer(model, loss_fn=lossfn_data)



# Training loop
n_epochs = 100
epoch_loss = torch.zeros((n_epochs,))

for i in range(n_epochs):
    loss_list = list()

    for nbatch, batch in enumerate(dataloader):
        print(f"{nbatch=}")

        optimiser.zero_grad()

        # Data loss
        x, y = [tensor.float() for tensor in batch]  # Network weights have dtype torch.float32
        loss_data = data_trainer(x, y)
        # Residual loss

        # Boundary losses

        loss_total = loss_data

        loss_total.backward()
        optimiser.step()
        loss_list.append(loss_total.detach())
        if nbatch == 1000:
            break

    epoch_loss[i] = torch.stack(loss_list).mean().item()

    test_pred = model(batched_domain)


    fig = plt.figure(figsize=(4, 8))
    ax_loss = fig.add_subplot(2, 1, 1)
    ax_surf = fig.add_subplot(2, 1, 2, projection="3d")
    ax_loss.semilogy(epoch_loss)
    ax_surf.plot_surface(grid_x, grid_y, test_pred.reshape_as(grid_x).detach())
    plt.show()


