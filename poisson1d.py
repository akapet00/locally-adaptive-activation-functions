r"""Implementation of locally adaptive activation functions (LAAFs)
with slope recovery for deep and physics-informed neural networks
(PINNs) in PyTorch.

Original implementation of LAAFs in TensorFlow is open source and
available on the first author's GitHub at:
AmeyaJagtap/Locally-Adaptive-Activation-Functions-Neural-Networks-

The LAAFs are introduced in the paper by Jagtap et al. available at:
https://doi.org/10.1098/rspa.2020.0334

Author: Ante Lojic Kapetanovic
"""

import argparse
import datetime
import logging
import os
import sys
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import tqdm

from utils import SUPPORTED_OPTIMIZERS
from utils import optimizer_dispatcher
from utils import Swish
from utils import AdaptiveLinear


def parse_arguments():
    r"""Return parsed arguments.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Parsed input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda',
        action='store_true', help='Use CUDA GPU for training if available')
    parser.add_argument('--domain',
        type=float, nargs=2, default=[0.0, 1.0], help='Boundaries of the solution domain')
    parser.add_argument('--boundary_conditions',
        type=float, nargs=2, default=[1.0, 1.0], help='Boundary conditions on boundaries of the domain')
    parser.add_argument('--rhs',
        type=float, default=-10.0, help='Right-hand-side forcing function')
    parser.add_argument('--n_layers',
        type=int, default=3, help='The number of hidden layers of the neural network')
    parser.add_argument('--n_units',
        type=int, default=50, help='The number of neurons per hidden layer')
    parser.add_argument('--activation',
        type=str, default='tanh', help='activation function')
    parser.add_argument('--optimizer',
        type=str, default='adam', choices=SUPPORTED_OPTIMIZERS, help='Optimization procedure')
    parser.add_argument('--n_epochs',
        type=int, default=1000, help='The number of training epochs')
    parser.add_argument('--batch_size',
        type=int, default=101, help='The number of data points for optimization per epoch')
    parser.add_argument('--linspace',
        action='store_true', help='Space the batch of data linearly, otherwise random')
    parser.add_argument('--learning_rate',
        type=float, default=1e-3, help='Learning rate applied for gradient based optimizers')
    parser.add_argument('--dropout_rate',
        type=float, default=0.0, help='Dropout regularization rate')
    parser.add_argument('--apply_mcdropout',
        action='store_true', help='Apply MCdropout for uncertainty quantification')
    parser.add_argument('--adaptive_rate',
        type=float, help='Add additional adaptive rate parameter to activation function')
    parser.add_argument('--adaptive_rate_scaler',
        type=float, help='Apply constant scaler to the adaptive rate')
    parser.add_argument('--save_fig',
        type=str, help='Save figure with specified name')
    args = parser.parse_args()
    return args


def solve_poisson(x, rhs, boundary_conditions):
    r"""Solve 1-D Poisson equation of simple form as follows:
    .. math::
        \frac{\mathrm{d} \phi^2}{\mathrm{d} x^2} = f(x)

    with known Dirichlet boundary conditions on arbitrary solution
    domain.

    Parameters
    ----------
    x : numpy.ndarray
        Independent variable to solve Poisson equation with respect to
    rhs : float
        Value of the scalar right hand side function
    boundary_conditions : tuple or list
        Boundary conditions

    Returns
    -------
    numpy.ndarray
        Poisson equation analytic solution, :math:`phi(x)`
    """
    x0 = x.min()
    x1 = x.max()
    C1 = (
        1 / (x1 - x0) 
        * (boundary_conditions[1] - rhs / 2 * x1**2 + rhs / 2 * x0**2 - boundary_conditions[0])
    )
    C2 = boundary_conditions[0] - rhs / 2 * x0**2 - C1 * x0 
    return rhs / 2 * x**2 + C1 * x + C2


class Net(nn.Module):
    r"""Neural approximator for the unknown function that is supposed
    to be solved.

    More details available in Raissi, M. et al. Physics-informed neural
    networks: A deep learning framework for solving forward and inverse
    problems involving nonlinear partial differential equations, J.
    Comput. Phys. 2019.

    Parameters
    ----------
    sizes : list
        Each element represents the number of neuron per layer
    activation : callable 
        Activation function
    dropout_rate : float, optional
        Dropout rate for regulrization during training process and
        uncertainty quantification by means of Monte Carlo dropout
        procedure while performing evaluation
    adaptive_rate : float, optional
        Scalable adaptive rate parameter for activation function that
        is added layer-wise for each neuron separately. It is treated
        as learnable parameter and will be optimized using a optimizer
        of choice
    adaptive_rate_scaler : float, optional
        Fixed, pre-defined, scaling factor for adaptive activation
        functions
    """
    def __init__(self, sizes, activation, dropout_rate=0.0, adaptive_rate=None, adaptive_rate_scaler=None):
        super(Net, self).__init__()
        self.regressor = nn.Sequential(
            *[Net.linear_block(in_features, out_features, activation, dropout_rate, adaptive_rate, adaptive_rate_scaler)
            for in_features, out_features in zip(sizes[:-1], sizes[1:-1])],     
            AdaptiveLinear(sizes[-2], sizes[-1]) # output layer is regular linear transformation
            )
        
    def forward(self, x):
        return self.regressor(x)

    @staticmethod
    def linear_block(in_features, out_features, activation, dropout_rate, adaptive_rate, adaptive_rate_scaler):
        activation_dispatcher = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['relu', nn.ReLU()],
            ['tanh', nn.Tanh()],
            ['sigmoid', nn.Sigmoid()],
            ['swish', Swish()]
        ])
        return nn.Sequential(
            AdaptiveLinear(in_features, out_features, adaptive_rate=adaptive_rate, adaptive_rate_scaler=adaptive_rate_scaler),
            activation_dispatcher[activation],
            nn.Dropout(dropout_rate),
            )


def train(
        device, domain, boundary_conditions, rhs,
        sizes, activation, optimizer, n_epochs, batch_size, linspace, learning_rate,
        dropout_rate,
        adaptive_rate, adaptive_rate_scaler
        ):
    r"""Train PINN and return trained network alongside loss over time.

    Parameters
    ----------
    device : str
        Specifiy `cuda` if CUDA-enabled GPU is available, otherwise
        specify `cpu`
    domain : tuple or list
        Boundaries of the solution domain
    boundary_conditions : tuple or list
        Boundary conditions
    rhs : float
        Value of the scalar right hand side function
    sizes : list
        Each element represents the number of neuron per layer
    activation : callable 
        Activation function
    optimizer : callable
        Optimization procedure
    n_epochs : int
        The number of training epochs
    batch_size : int
        The number of data points for optimization per epoch
    linspace : bool
        Space the batch of data linearly, otherwise random
    learning_rate : float
        Learning rate
    dropout_rate : float, optional
        Dropout rate for regulrization during training process and
        uncertainty quantification by means of Monte Carlo dropout
        procedure while performing evaluation
    adaptive_rate : float, optional
        Scalable adaptive rate parameter for activation function that
        is added layer-wise for each neuron separately. It is treated
        as learnable parameter and will be optimized using a optimizer
        of choice
    adaptive_rate_scaler : float, optional
        Fixed, pre-defined, scaling factor for adaptive activation
        functions
    
    Returns
    -------
    net : Net
        Trained function approximator
    loss_list : list
        Loss values during training process
    """
    net = Net(sizes, activation, dropout_rate, adaptive_rate, adaptive_rate_scaler).to(device=device)
    optimizer = optimizer_dispatcher(optimizer, net.parameters(), learning_rate)
    loss_list = []
    logging.info(f'{net}\n')
    logging.info(f'Training started at {datetime.datetime.now()}\n')
    start_time = timer()
    for _ in tqdm.tqdm(range(n_epochs), desc='[Training procedure]', ascii=True, total=n_epochs):
        def closure():
            if linspace:
                x = torch.linspace(*domain, steps=batch_size, device=device).unsqueeze(-1)
            else:
                x = (domain[0] - domain[1]) * torch.rand(size=(batch_size, ), device=device).unsqueeze(-1) + domain[1]
            x.requires_grad = True

            phi = net(x)
            x.grad = None
            phi.backward(torch.ones_like(x, device=device), create_graph=True)
            phi_x = x.grad
            x.grad = None
            phi_x.backward(torch.ones_like(x, device=device), create_graph=True)
            phi_xx = x.grad
            domain_residual = phi_xx - rhs(x)

            boundaries = torch.tensor(domain, device=device).unsqueeze(-1)
            boundaries.requires_grad = True
            boundary_residual = net(boundaries) - torch.tensor(boundary_conditions, device=device).unsqueeze(-1)

            if adaptive_rate:
                local_recovery_terms = torch.tensor([torch.mean(net.regressor[layer][0].A.data) for layer in range(len(net.regressor) - 1)])
                slope_recovery_term = 1 / torch.mean(torch.exp(local_recovery_terms))
                loss = (torch.mean(domain_residual ** 2) + torch.mean(boundary_residual ** 2)) + slope_recovery_term
            else:
                loss = (torch.mean(domain_residual ** 2) + torch.mean(boundary_residual ** 2))
            loss_list.append(loss)
            optimizer.zero_grad()
            loss.backward()
            return loss
        optimizer.step(closure)
    elapsed = timer() - start_time
    logging.info(f'Training finished. Elapsed time: {elapsed} s\n')
    return net, loss_list


def eval_and_viz(
        device, domain, boundary_conditions, rhs,
        net, loss_list,
        apply_mcdropout,
        save_fig
        ):
    r"""Evaluate and visualize.

    Parameters
    ----------
    device : str
        Specifiy `cuda` if CUDA-enabled GPU is available, otherwise
        specify `cpu`
    domain : tuple or list
        Boundaries of the solution domain
    boundary_conditions : tuple or list
        Boundary conditions
    rhs : float
        Value of the scalar right hand side function
    net : Net
        Trained function approximator
    loss_list : list
        Loss values during training process
    apply_mcdropout : bool
        Apply Monte Carlo dropout for uncertainty quantification if set
        to `True`
    save_fig : bool
        Save figure

    Returns
    -------
    None
    """
    x = torch.linspace(*domain, 101, device=device).unsqueeze(-1)
    rhs = rhs(x).cpu().detach().numpy().ravel()
    if apply_mcdropout:
        y_pred_mc = np.empty((1000, x.shape[0]))
        for i in range(1000):
            y_pred = net(x)
            y_pred_mc[i, :] = y_pred.cpu().detach().numpy().ravel()
        net.eval()
        y_pred = np.mean(y_pred_mc, axis=0)
        y_ci = np.std(y_pred_mc, axis=0)
    else:
        net.eval() 
        y_pred = net(x)
        y_pred = y_pred.cpu().detach().numpy().ravel()
    x = x.cpu().detach().numpy().ravel()
    y = solve_poisson(x, rhs, boundary_conditions)
    rmse_val = np.sqrt(np.mean((y - y_pred)**2))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax[0].plot(x, y, 'k-', linewidth=2, label='Analytic solution')
    ax[0].plot(x, y_pred, 'r--', dashes=(3, 4), linewidth=3, label='PINN solution')
    if apply_mcdropout:
        ax[0].fill_between(x, y_pred + 2*y_ci, y_pred - 2*y_ci, color='r', alpha=0.1, label='95% CI')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title(f'RMSE = {rmse_val:.6f}')
    ax[0].legend()
    ax[1].plot(loss_list, 'r-')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('training epoch')
    ax[1].set_ylabel('loss value')
    plt.tight_layout()
    plt.show()
    if save_fig:
        fig.savefig(os.path.join('figs', f'{save_fig}.png'), format='png', bbox_inches='tight', dpi=200)
    

def main():
    torch.set_default_dtype(torch.float32)
    args = parse_arguments()
    if args.cuda:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')
    logging.info(f'Device: {device}\n')
    rhs = lambda x: torch.tensor([args.rhs], device=device)
    domain = args.domain
    boundary_conditions = args.boundary_conditions

    # configure neural network
    sizes = [1] + args.n_layers * [args.n_units] + [1]
    activation = args.activation
    optimizer = args.optimizer
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    linspace = args.linspace
    learning_rate = args.learning_rate
    dropout_rate = args.dropout_rate
    apply_mcdropout = args.apply_mcdropout
    adaptive_rate = args.adaptive_rate
    adaptive_rate_scaler = args.adaptive_rate_scaler
    save_fig = args.save_fig

    # trainining process
    net, loss_list = train(
        device, domain, boundary_conditions, rhs,
        sizes, activation, optimizer, n_epochs, batch_size, linspace, learning_rate,
        dropout_rate,
        adaptive_rate, adaptive_rate_scaler
        )
    
    # evaluation
    eval_and_viz(
        device, domain, boundary_conditions, rhs,
        net, loss_list,
        apply_mcdropout,
        save_fig
        )


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main()