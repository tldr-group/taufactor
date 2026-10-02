"""Electrochemical impedance spectroscopy solvers."""

from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from ..utils import compute_impedance
from .base import SORSolver


class ImpedanceSolver(SORSolver):
    """
    Electrode Solver - solves the electrode tortuosity factor system (migration and capacitive current between current collector and solid/electrolyte interface)
    Once solve method is called, tau, D_eff and D_rel are available as attributes.
    """
    connectivity_open_end = False

    def __init__(self, img, conductive_label=1, reactive_label=0, \
                 omega=None, spacing=None, mode='tau_e', device='cuda'):
        self.left_bc = 1.0
        self.electrode_bc = 0.0
        self.conductive_labels = [conductive_label]
        self.reac_label=reactive_label
        self.dx = spacing or 1
        super().__init__(img, omega=omega, device=device, precision=torch.double)
        if self.batch_size > 1:
            raise TypeError(
                "Error: The ImpedanceSolver is only implemented for batch_size=1! "
                "TODO: vectorize impedance solver for batching."
            )
        
        torch_img = torch.as_tensor(self.cpu_img, device=self.device)
        mask_f = self.mask.to(self.precision)
        self.cond_nn, self.reac_nn = self.count_neighbours(torch_img, mask_f)
        del torch_img, mask_f

        # Volume fraction (slice-wise)
        self.vol_x = self.vol_x[0]
        self.a_x = (torch.sum(self.reac_nn, (0, 2, 3)) / (self.Ny*self.Nz*self.dx)).cpu().numpy()

        # Define frequency, resistance and capacitance
        self.resistance = 1
        self.capacitance = 1
        self.freq_0 = np.mean(self.vol_x) / np.mean(self.a_x*self.dx) / self.Nx**2
        if mode == 'tau_e':
            self.frequency = np.array([self.freq_0/2, self.freq_0/4])
        elif mode == 'nyquist':
            freq_range = self.freq_0 * 2 ** (np.arange(-3, 10, 0.5))
            self.frequency = freq_range[::-1]
        self.Z_mean = compute_impedance(np.full(self.Nx, 1/np.mean(self.vol_x)), \
                                        np.full(self.Nx, np.mean(self.a_x*self.dx)), \
                                        self.frequency)

        # solving params
        self.sum_iter = 0
        self.c_x = 0

    def init_field(self, mask):
        self.mask = mask.to(dtype=torch.bool)
    
    def init_conductive_neighbours(self, img, mask):
        return None

    def init_field_internal(self, mask):
        """
        Initialise field with analytical solution
        :param img: input image, with 1s conductive and 0s non-conductive
        :type img: torch.array
        :return: initial field
        :rtype: torch.array
        """
        x = np.arange(0, self.Nx)+0.5
        lambda_0 = np.sqrt(1j * self.frc * np.mean(self.a_x[1:-1]*self.dx) / np.mean(self.vol_x))
        phi = self.left_bc*np.cosh(lambda_0*(self.Nx-x))/np.cosh(lambda_0*self.Nx)
        if self.precision == torch.float:
            vec = torch.tensor(phi, dtype=torch.complex64, device=self.device)
        else:
            vec = torch.tensor(phi, dtype=torch.complex128, device=self.device)
        for i in range(2):
            vec = torch.unsqueeze(vec, -1)
        vec = torch.unsqueeze(vec, 0)
        mask_f = mask.to(vec.dtype) if mask.dtype == torch.bool else mask
        return self._pad(mask_f * vec, [2 * self.left_bc, 0]).to(self.device)

    def count_neighbours(self, img, mask):
        """
        Initialise factor based on conductive and capacitive neighbours
        factor = (N_i + 2*S_i* j*r*w*c / (1 + j*r*frequency*c)
        N_i: amount of conductive neighbours (cond_nn)
        S_i: amount of capacitive neighbours (reac_nn)

        :param img: input image, with 1s conductive and 0s non-conductive
        :type img: cp.array
        :return: prefactor
        :rtype: cp.array
        """      
        # Conducting nearest neighbours
        padded = self._pad(mask, [2, 0])
        cond_nn = torch.empty_like(mask)
        self._neighbour_sum_from_padded(padded, cond_nn)
        del padded

        # Capacitive nearest neighbours
        reac = (img == self.reac_label).to(dtype=self.precision)
        padded = self._pad(reac)
        del reac
        reac_nn = torch.empty_like(mask)
        self._neighbour_sum_from_padded(padded, reac_nn)
        del padded

        # Masking conducting voxels
        cond_nn[mask == 0] = 0.0
        reac_nn[mask == 0] = 0.0
        return cond_nn, reac_nn

    def solve(self, iter_limit=10000, verbose=True, conv_crit=1e-3, plot_interval=10):
        """
        run a solve simulation

        :param iter_limit: max iterations before aborting, will attemtorch double for the same no. iterations
        if initialised as singles
        :param verbose: Whether to print tau. Can be set to 'per_iter' for more feedback
        :param conv_crit: convergence criteria - running standard deviation of tau_e
        :param conv_crit_2: convergence criteria - maximum difference between tau_e in consecutive frequency solves
        :return: tau
        """
        if (verbose) and (self.device.type == 'cuda'):
            torch.cuda.reset_peak_memory_stats(device=self.device)

        self.tau_t = []

        self.impedance = []
        self.converged_freq = []
        self.taus = []

        # init field
        self.frc = self.frequency[0]*self.resistance*self.capacitance
        self.field = self.init_field_internal(self.mask)

        with torch.no_grad():
            increment = torch.empty(
                (self.batch_size, self.Nx, self.Ny, self.Nz),
                dtype=self.field.dtype,
                device=self.device,
            )
            start = timer()
            for f in self.frequency:
                self.frc = f*self.resistance*self.capacitance
                if torch.any(torch.isnan(self.field)):
                    self.field = self.init_field_internal(self.mask)
                # Assemble total prefactor
                # Variant 1: (N_i + 2*S_i* j*r*frequency*c / (1 + j*r*frequency*c) )**-1
                # factor = (self.cond_nn + 2*self.reac_nn * 1j*self.frc/(2+1j*self.frc))**-1
                # Variant 2: (N_i + S_i * j*r*frequency*c)**-1
                factor = (self.cond_nn + self.reac_nn * 1j*self.frc)**-1
                factor.masked_fill_(~self.mask, 0)
                factor[(self.cond_nn+self.reac_nn) == 0] = 0

                self.impedance.append(0)
                self.converged = False
                self.tau = 0
                self.iter = 0
                while not self.converged and self.iter < iter_limit:
                    self.apply_boundary_conditions()
                    self.sum_weighted_neighbours(increment)
                    increment *= factor
                    increment -= self.field[:, 1:-1, 1:-1, 1:-1]
                    self._apply_chequerboard(increment)
                    self.field[:, 1:-1, 1:-1, 1:-1] += increment
                    self.iter += 1

                    if self.iter % 100 == 0:
                        self.converged = self.check_convergence(verbose, conv_crit, plot_interval)
                self.converged_freq.append(self.converged)
                self.sum_iter += self.iter
                # Compute tau_e from last impedance (at lowest frquency)
                self.taus.append(self.tau)
            print(self.taus)
            self.walltime = timer() - start
            self._end_simulation(self.sum_iter, verbose)

    def calc_input_impedance(self):
        # Calculate total current on left boundary
        influx = 2*self.left_bc - 2*self.field[:, 1:2, 1:-1, 1:-1]
        influx[self.field[:, 1:2, 1:-1, 1:-1] == 0] = 0
        influx = torch.mean(influx, (0, 2, 3)).cpu().numpy()
        return self.left_bc/influx[0]
    
    def plot_stats(self, relative_error):
        i = np.argmax(relative_error)
        if not hasattr(self, '_plot_fig'):
            self._plot_fig, self._plot_ax = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
        ax = self._plot_ax
        for axis in ax:
            axis.clear()
        x = np.arange(0, self.Nx)+0.5
        ax[0].plot(x, self.vol_x, label='vol_x', color='gray', linestyle='--')
        ax[0].plot(x, self.c_x, label='c_x', color='blue', linestyle='-')

        # Analytical solution for ideal c profile
        lambda_0 = np.sqrt(1j * self.frc * np.mean(self.a_x[1:-1]*self.dx) / np.mean(self.vol_x))
        phi = np.abs(self.left_bc*np.cosh(lambda_0*(self.Nx-x))/np.cosh(lambda_0*self.Nx))
        ax[0].plot(x, phi, label='phi_ideal', color='black', linestyle=':')
        ax[0].set_xlabel('voxels in x')
        ax[0].set_ylabel('vol/c')
        ax[0].set_title(f'Homogenized quantities in iter {self.iter}')
        self._set_plot_subtitle(
            ax[0], f'Iter: {self.iter}, conv error: {abs(relative_error[i]):.3E}, '
            f'tau: {self.tau[i]:.5f} (batch element {i})'
        )
        ax[0].set_ylim(0, 1.2)
        ax[0].legend()
        ax[0].grid()

        ax[1].plot(np.ones(100), np.linspace(0,10,100), color='gray', linestyle='--', label='ref')
        
        # freq = 2 ** (np.arange(-3, 10, 0.5))
        # freq_ref = 1
        # scale = np.cosh(np.sqrt(1j*freq_ref)) / (np.sqrt(1j*freq_ref) * np.sinh(np.sqrt(1j*freq_ref)))
        # Z_pde_ref = np.cosh(np.sqrt(1j * freq)) / (np.sqrt(1j * freq) * np.sinh(np.sqrt(1j * freq))) / scale.real
        # Z_point_ref = np.cosh(np.sqrt(1j*freq_ref)) / (np.sqrt(1j*freq_ref) * np.sinh(np.sqrt(1j*freq_ref))) / scale.real
        # ax[1].plot(np.real(Z_pde_ref), -np.imag(Z_pde_ref), color='black', linestyle='-', label='ref')
        # ax[1].scatter(np.real(Z_point_ref), -np.imag(Z_point_ref), label='freq', facecolors='none', edgecolors='black', s=60)

        scale = self.Z_mean[-1].real
        ax[1].plot(np.real(self.Z_mean)/scale, -np.imag(self.Z_mean)/scale, 'k-', label='Ref')
        ax[1].plot(np.real(self.impedance)/scale, -np.imag(self.impedance)/scale, 'rx-', label='tau(x)')
        # ax[1].scatter(np.real(self.impedance)/scale, -np.imag(self.impedance)/scale, facecolors='blue', edgecolors='blue', s=40, label='sim')

        ref_idx = np.argmin(np.abs(self.frequency - self.freq_0))
        ax[1].scatter(self.Z_mean.real[ref_idx]/scale, -self.Z_mean.imag[ref_idx]/scale, c='gray', edgecolors='k', s=80, label='f_0')
        if len(self.impedance) > 19:
            Z_point = np.array(self.impedance)[19]
            ax[1].scatter(np.real(Z_point)/scale, -np.imag(Z_point)/scale, label='freq', facecolors='red', edgecolors='blue', s=50)
        ax[1].set_xlim([0, max(1.5, np.real(self.impedance[-1])/scale+0.1)])
        ax[1].set_ylim([0, max(3.5, -np.imag(self.impedance[-1])/scale+0.1)])
        ax[1].legend()
        # ax[1].set_aspect('equal', 'box')
        ax[1].set_xlabel("Z'")
        ax[1].set_ylabel("-Z''")
        ax[1].set_title('Nyquist plot of TLM')
        self._plot_fig.canvas.draw_idle()
        self._plot_fig.canvas.flush_events()
        self._display_figure(self._plot_fig, '_plot_display')

    def compute_metrics(self):
        # TODO: vectorize for batching
        c_av = torch.mean(torch.abs(self.field[:, 1:-1, 1:-1, 1:-1]), (0, 2, 3)).cpu().numpy()
        c_av[self.vol_x > 0] /= self.vol_x[self.vol_x > 0]
        relative_error = np.array(np.max(np.abs(c_av-self.c_x)) / (np.max(c_av)-np.min(c_av)))
        if np.any(np.isnan(c_av)):
            # tau = inf, relative_error = 0
            return np.expand_dims(0.0, axis=0), np.expand_dims(0.0, axis=0)
        self.c_x = c_av
        self.impedance[-1] = self.calc_input_impedance()
        tau = np.array(self.impedance[-1].real / self.Z_mean[-1].real)
        return np.expand_dims(tau, axis=0), np.expand_dims(relative_error, axis=0)


class PeriodicImpedanceSolver(ImpedanceSolver):
    """
    Solver with periodic boundary conditions in y and z direction.
    """
    connectivity_periodic = (False, True, True)

    def count_neighbours(self, img, mask):
        img2 = self._pad(mask, [2, 0])[:, :, 1:-1, 1:-1]
        cond_nn = torch.zeros_like(img2)
        # iterate through shifts in the spatial dimensions
        for dim in range(1, 4):
            for dr in [1, -1]:
                cond_nn += torch.roll(img2, dr, dim)
        cond_nn = cond_nn[:, 1:-1]

        img2 = torch.zeros_like(img)
        img2[img==self.reac_label] = 1
        img2 = self._pad(img2)[:, :, 1:-1, 1:-1]
        reac_nn = torch.zeros_like(img2)
        # iterate through shifts in the spatial dimensions
        for dim in range(1, 4):
            for dr in [1, -1]:
                reac_nn += torch.roll(img2, dr, dim)
        reac_nn = reac_nn[:, 1:-1]

        # Masking conducting voxels
        cond_nn[mask == 0] = 0.0
        reac_nn[mask == 0] = 0.0
        return cond_nn, reac_nn

    def apply_boundary_conditions(self):
        self.field[:,:,0,:] = self.field[:,:,-2,:]
        self.field[:,:,-1,:] = self.field[:,:,1,:]
        self.field[:,:,:,0] = self.field[:,:,:,-2]
        self.field[:,:,:,-1] = self.field[:,:,:,1]
