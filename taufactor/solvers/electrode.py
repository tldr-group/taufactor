"""Electrode tortuosity solvers."""

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from ..utils import compute_impedance_batched
from .base import SORSolver


class ElectrodeSolver(SORSolver):
    """
    Default solver for two phase images. Once solve method is
    called, tau, D_eff and D_rel are available as attributes.
    """
    connectivity_open_end = False

    def __init__(self, img, conductive_label=1, reactive_label=0, \
                 omega=None, spacing=None, device='cuda'):
        """
        Initialise parameters, solution field and other tools that can be re-used
        for multiple solves.

        :param img: input img with 1s conductive and 0s non-conductive
        :param bc: Upper and lower boundary conditions. Leave as default.
        :param D_0: reference material diffusivity
        :param device: pytorch device, can be cuda or cpu 
        """
        self.left_bc = 1.0
        self.electrode_bc = 0.0
        self.conductive_labels = [conductive_label]
        self.reac_label=reactive_label
        self.dx = spacing or 1
        super().__init__(img, omega=omega, device=device)
        self.c_x = 0
        # ElectrodeSolver never reads cpu_img after init (unlike through-transport)
        self.cpu_img = None

    def init_field(self, mask):
        x = np.arange(self.Nx)+0.5
        c_init = self.electrode_bc + (self.left_bc-self.electrode_bc)*np.cosh(1-x/self.Nx)/np.cosh(1)
        vec = torch.tensor(c_init, dtype=self.precision, device=self.device)
        for i in range(2):
            vec = torch.unsqueeze(vec, -1)
        vec = torch.unsqueeze(vec, 0)
        return self._pad(mask * vec, [self.left_bc * 2, 0])

    def init_conductive_neighbours(self, img, mask):
        padded = self._pad(mask, [2, 0])
        cond_nn = torch.empty_like(mask)
        self._neighbour_sum_from_padded(padded, cond_nn)
        del padded
        cond_nn.masked_fill_(mask == 0, torch.inf)
        return cond_nn

    def init_reactive_neighbours(self, img, mask):
        reac = (img == self.reac_label).to(dtype=self.precision)
        padded = self._pad(reac)
        del reac
        reac_nn = torch.empty(
            (self.batch_size, self.Nx, self.Ny, self.Nz),
            dtype=self.precision,
            device=self.device,
        )
        self._neighbour_sum_from_padded(padded, reac_nn)
        del padded
        reac_nn.masked_fill_(mask == 0, 0)
        return reac_nn

    def compute_metrics(self):
        c_x = torch.mean(self.field[:, 1:-1, 1:-1, 1:-1], (2, 3)).cpu().numpy()
        c_x = np.divide(c_x, self.conn_vol_x, out=np.zeros_like(self.conn_vol_x),
                        where=self.conn_vol_x != 0)
        # Largest deviation to previous check as conv crit
        relative_error = np.max(np.abs(c_x-self.c_x), axis=1)
        self.c_x = c_x

        fluxes = -self.field[:, 1:-1, 1:-1, 1:-1] + self.field[:, :-2, 1:-1, 1:-1]
        fluxes[:, 0, :, :] = (self.left_bc-self.field[:, 1, 1:-1, 1:-1])/0.5
        fluxes[self.field[:, 1:-1, 1:-1, 1:-1] == 0] = 0
        fluxes[self.field[:, :-2, 1:-1, 1:-1] == 0] = 0
        fluxes = torch.mean(fluxes, (2, 3)).cpu().numpy()
        fluxes_1d = np.concatenate((2*(self.left_bc-c_x[:,:1]), (-c_x[:,1:]+c_x[:,:-1])), axis=1)
        fluxes_1d[:,1:][self.conn_vol_x[:,1:]==0] = 0
        fluxes_1d[:,1:][self.conn_vol_x[:,:-1]==0] = 0

        # Make some quantities visible to user
        # Porosity at voxel faces (arithmetic mean between voxel centers)
        eps = np.concatenate((self.vol_x[:,:1], 0.5*(self.vol_x[:,:-1]+self.vol_x[:,1:])), axis=1)
        self.tau_x = np.divide(eps * fluxes_1d, fluxes,
            out=np.full_like(fluxes_1d, np.nan), where=fluxes != 0)
        # Difference of in- and out-going fluxes equals reactive fluxes
        fluxes[:,:-1] -= fluxes[:,1:]
        self.k_x = np.divide(fluxes, c_x - self.electrode_bc, out=np.zeros_like(c_x),
                        where=(c_x - self.electrode_bc) != 0) / self.k_0[:, None]

        freq = np.mean(eps, axis=1, keepdims=True) / np.mean(self.a_x*self.dx, axis=1, keepdims=True) / self.Nx**2 * 2**-3
        R = self.tau_x/eps
        R[eps == 0] = 1e30
        R[np.isnan(self.tau_x)] = 1e30
        self.Z_sim = compute_impedance_batched(R, self.a_x*self.dx, freq)
        R_ideal = 1/np.mean(eps, axis=1)[:, None].repeat(self.Nx, axis=1)
        C_ideal = np.mean(self.a_x*self.dx, axis=1)[:, None].repeat(self.Nx, axis=1)
        self.Z_ideal = compute_impedance_batched(R_ideal, C_ideal, freq)
        tau = self.Z_sim[:,0].real/self.Z_ideal[:,0].real
        return tau, relative_error
    
    def plot_stats(self, relative_error):
        i = np.argmax(relative_error)
        if not hasattr(self, '_plot_fig'):
            self._plot_fig, self._plot_ax = plt.subplots()
        ax = self._plot_ax
        ax.clear()
        x = np.arange(0, self.Nx)+0.5
        ax.plot(x, self.vol_x[i], label='$\\epsilon(x)$', color='gray', linestyle='--')

        # Analytical solution for ideal c profile
        c = self.electrode_bc + (self.left_bc-self.electrode_bc)*np.cosh(1-x/self.Nx)/np.cosh(1)
        ax.plot(x, c, label='$c_\\text{ideal}(x)$', color='black', linestyle=':')
        
        ax.plot(x, self.c_x[i], label='$c(x)$', color='blue', linestyle='-')
        ax.plot(x-0.5, 1/self.tau_x[i], label='$\\tau^{-1}(x)$', color='red', linestyle='-')
        ax.plot(x, np.abs(self.k_x[i]/(self.a_x[i]*self.dx)-1), label='rel_error', color='lime', linestyle='-.')

        ax.set_xlabel('voxels in x')
        ax.set_ylabel('$\\epsilon(x)$, $c(x)$, $\\tau^{-1}(x)$')
        ax.set_title(f'Homogenized quantities in iter {self.iter}')
        self._set_plot_subtitle(
            ax, f'Iter: {self.iter}, conv error: {abs(relative_error[i]):.3E}, '
            f'tau: {self.tau[i]:.5f} (batch element {i})'
        )
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.grid()
        self._plot_fig.canvas.draw_idle()
        self._plot_fig.canvas.flush_events()
        self._display_figure(self._plot_fig, '_plot_display')


class PeriodicElectrodeSolver(ElectrodeSolver):
    """
    Solver with periodic boundary conditions in y and z direction.
    """
    connectivity_periodic = (False, True, True)

    def init_conductive_neighbours(self, img, mask):
        padded = self._pad(mask, [2, 0])[:, :, 1:-1, 1:-1]
        cond_nn = torch.empty_like(mask)
        self._periodic_yz_neighbour_sum_from_padded(padded, cond_nn)
        del padded
        cond_nn.masked_fill_(mask == 0, torch.inf)
        return cond_nn

    def init_reactive_neighbours(self, img, mask):
        reac = (img == self.reac_label).to(dtype=self.precision)
        padded = self._pad(reac)[:, :, 1:-1, 1:-1]
        del reac
        reac_nn = torch.empty(
            (self.batch_size, self.Nx, self.Ny, self.Nz),
            dtype=self.precision,
            device=self.device,
        )
        self._periodic_yz_neighbour_sum_from_padded(padded, reac_nn)
        del padded
        reac_nn.masked_fill_(mask == 0, 0)
        return reac_nn

    def apply_boundary_conditions(self):
        self.field[:,:,0,:] = self.field[:,:,-2,:]
        self.field[:,:,-1,:] = self.field[:,:,1,:]
        self.field[:,:,:,0] = self.field[:,:,:,-2]
        self.field[:,:,:,-1] = self.field[:,:,:,1]
