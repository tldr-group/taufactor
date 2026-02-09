"""Main module."""
from IPython.display import clear_output
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
try:
    import torch
except Exception:
    torch = None
from .taufactor import SORSolver
from .utils import compute_impedance, compute_impedance_batched

class ElectrodeSolver(SORSolver):
    """
    Default solver for two phase images. Once solve method is
    called, tau, D_eff and D_rel are available as attributes.
    """
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
        self.cond_label=conductive_label
        self.reac_label=reactive_label
        self.dx = spacing or 1
        super().__init__(img, omega=omega, device=device)
        self.c_x = 0

    def return_mask(self, img):
        mask = torch.zeros_like(img)
        mask[img==self.cond_label] = 1
        return mask

    def init_field(self, mask):
        x = np.arange(self.Nx)+0.5
        c_init = self.electrode_bc + (self.left_bc-self.electrode_bc)*np.cosh(1-x/self.Nx)/np.cosh(1)
        vec = torch.tensor(c_init, dtype=self.precision, device=self.device)
        for i in range(2):
            vec = torch.unsqueeze(vec, -1)
        vec = torch.unsqueeze(vec, 0)
        vec = vec.repeat(self.batch_size, 1, self.Ny, self.Nz, )
        return self._pad(mask * vec, [self.left_bc * 2, 0])

    def init_conductive_neighbours(self, img):
        mask = self.return_mask(img)
        img2 = self._pad(mask, [2, 0])
        cond_nn = self._sum_by_rolling(img2)
        cond_nn = self._crop(cond_nn, 1)
        cond_nn[mask == 0] = torch.inf
        return cond_nn

    def init_reactive_neighbours(self, img):
        img2 = torch.zeros_like(img)
        img2[img==self.reac_label] = 1
        img2 = self._pad(img2)
        reac_nn = self._sum_by_rolling(img2)
        reac_nn = self._crop(reac_nn, 1)
        reac_nn[img != self.cond_label] = 0.0
        return reac_nn

    def compute_metrics(self):
        c_x = torch.mean(self.field[:, 1:-1, 1:-1, 1:-1], (2, 3)).cpu().numpy()/self.vol_x
        c_x[self.vol_x == 0] = 0
        # Largest deviation to previous check as conv crit
        relative_error = np.max(np.abs(c_x-self.c_x), axis=1)
        self.c_x = c_x

        fluxes = -self.field[:, 1:-1, 1:-1, 1:-1] + self.field[:, :-2, 1:-1, 1:-1]
        fluxes[:, 0, :, :] = (self.left_bc-self.field[:, 1, 1:-1, 1:-1])/0.5
        fluxes[self.field[:, 1:-1, 1:-1, 1:-1] == 0] = 0
        fluxes[self.field[:, :-2, 1:-1, 1:-1] == 0] = 0
        fluxes = torch.mean(fluxes, (2, 3)).cpu().numpy()
        fluxes_1d = np.concatenate((2*(self.left_bc-c_x[:,:1]), (-c_x[:,1:]+c_x[:,:-1])), axis=1)
        fluxes_1d[:,1:][self.vol_x[:,1:]==0] = 0
        fluxes_1d[:,1:][self.vol_x[:,:-1]==0] = 0

        # Make some quantities visible to user
        eps = np.concatenate((self.vol_x[:,:1], 0.5*(self.vol_x[:,:-1]+self.vol_x[:,1:])), axis=1)
        self.tau_x = eps*fluxes_1d/fluxes
        fluxes[:,:-1] -= fluxes[:,1:] 
        self.k_x = (fluxes) / (c_x - self.electrode_bc) / self.k_0[:, None]

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
        clear_output(wait=True)
        i = np.argmax(relative_error)
        print(f'Iter: {self.iter}, conv error: {abs(relative_error[i]):.3E}, tau: {self.tau[i]:.5f} (batch element {i})')
        fig, ax = plt.subplots() #figsize=(10, 4), dpi=200)
        x = np.arange(0, self.Nx)+0.5
        ax.plot(x, self.vol_x[i], label='vol_x', color='gray', linestyle='--')
        ax.plot(x, self.c_x[i], label='c_x', color='blue', linestyle='-')
        ax.plot(x-0.5, 1/self.tau_x[i], label='1/tau_x', color='red', linestyle='-')
        ax.plot(x, self.k_x[i]/(self.a_x[i]*self.dx), label='k_x', color='lime', linestyle='-.')
        
        # Analytical solution for ideal c profile
        c = self.electrode_bc + (self.left_bc-self.electrode_bc)*np.cosh(1-x/self.Nx)/np.cosh(1)
        ax.plot(x, c, label='c_ideal', color='black', linestyle=':')

        ax.set_xlabel('voxels in x')
        ax.set_ylabel('vol/a/c/tau/reaction')
        ax.set_title(f'Homogenized quantities in iter {self.iter}')
        ax.set_ylim(0, 1.2)
        ax.legend()
        ax.grid()
        plt.show()


class PeriodicElectrodeSolver(ElectrodeSolver):
    """
    Solver with periodic boundary conditions in y and z direction.
    """
    def init_conductive_neighbours(self, img):
        mask = self.return_mask(img)
        img2 = self._pad(mask, [2, 0])[:, :, 1:-1, 1:-1]
        cond_nn = self._sum_by_rolling(img2)
        cond_nn = cond_nn[:, 1:-1]
        cond_nn[mask == 0] = torch.inf
        return cond_nn

    def init_reactive_neighbours(self, img):
        img2 = torch.zeros_like(img)
        img2[img==self.reac_label] = 1
        img2 = self._pad(img2)[:, :, 1:-1, 1:-1]
        reac_nn = self._sum_by_rolling(img2)
        reac_nn = reac_nn[:, 1:-1]
        reac_nn[img != self.cond_label] = 0.0
        return reac_nn

    def apply_boundary_conditions(self):
        self.field[:,:,0,:] = self.field[:,:,-2,:]
        self.field[:,:,-1,:] = self.field[:,:,1,:]
        self.field[:,:,:,0] = self.field[:,:,:,-2]
        self.field[:,:,:,-1] = self.field[:,:,:,1]


class ImpedanceSolver(SORSolver):
    """
    Electrode Solver - solves the electrode tortuosity factor system (migration and capacitive current between current collector and solid/electrolyte interface)
    Once solve method is called, tau, D_eff and D_rel are available as attributes.
    """

    def __init__(self, img, conductive_label=1, reactive_label=0, \
                 omega=None, spacing=None, mode='tau_e', device='cuda'):
        self.left_bc = 1.0
        self.electrode_bc = 0.0
        self.cond_label=conductive_label
        self.reac_label=reactive_label
        self.dx = spacing or 1
        super().__init__(img, omega=omega, device=device, precision=torch.double)
        if self.batch_size > 1:
            raise TypeError(
                "Error: The ImpedanceSolver is only implemented for batch_size=1! "
                "TODO: vectorize impedance solver for batching."
            )
        
        torch_img = torch.tensor(self.cpu_img, dtype=self.precision, device=self.device)
        self.mask = torch.zeros_like(torch_img)
        self.mask[torch_img==self.cond_label] = 1
        self.cond_nn, self.reac_nn = self.count_neighbours(torch_img, self.mask)

        # Volume fraction (slice-wise)
        self.vol_x = torch.mean(self.mask, (0, 2, 3)).cpu().numpy()
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

    def return_mask(self, img):
        mask = torch.zeros_like(img)
        mask[img==self.cond_label] = 1
        return mask

    def init_field(self, img):
        return None
    
    def init_conductive_neighbours(self, img):
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
        vec = vec.repeat(self.batch_size, 1, self.Ny, self.Nz, )
        return self._pad(mask * vec, [2 * self.left_bc, 0]).to(self.device)

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
        img2 = self._pad(mask, [2, 0])
        cond_nn = self._sum_by_rolling(img2)
        cond_nn = self._crop(cond_nn, 1)

        # Capacitive nearest neighbours
        img2 = torch.zeros_like(img)
        img2[img==self.reac_label] = 1
        img2 = self._pad(img2)
        reac_nn = self._sum_by_rolling(img2)
        reac_nn = self._crop(reac_nn, 1)

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

        if verbose == 'debug':
            self.tau_t = []

        self.tau
        self.impedance = []
        self.converged_freq = []
        self.taus = []

        # init field
        self.frc = self.frequency[0]*self.resistance*self.capacitance
        self.field = self.init_field_internal(self.mask)

        with torch.no_grad():
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
                factor[self.mask == 0] = 0
                factor[(self.cond_nn+self.reac_nn) == 0] = 0

                self.impedance.append(0)
                self.converged = False
                self.tau = 0
                self.iter = 0
                while not self.converged and self.iter < iter_limit:
                    self.apply_boundary_conditions()
                    increment = self.sum_weighted_neighbours()
                    increment *= factor
                    increment -= self.field[:, 1:-1, 1:-1, 1:-1]
                    increment *= self.cb[self.iter % 2]
                    self.field[:, 1:-1, 1:-1, 1:-1] += increment
                    self.iter += 1

                    if self.iter % 100 == 0:
                        self.converged = self.check_convergence(verbose, conv_crit, plot_interval)
                self.converged_freq.append(self.converged)
                self.sum_iter += self.iter
                # Compute tau_e from last impedance (at lowest frquency)
                self.taus.append(self.tau)
            print(self.taus)
            self._end_simulation(self.sum_iter, verbose, start)

    def calc_input_impedance(self):
        # Calculate total current on left boundary
        influx = 2*self.left_bc - 2*self.field[:, 1:2, 1:-1, 1:-1]
        influx[self.field[:, 1:2, 1:-1, 1:-1] == 0] = 0
        influx = torch.mean(influx, (0, 2, 3)).cpu().numpy()
        return self.left_bc/influx[0]
    
    def plot_stats(self, relative_error):
        clear_output(wait=True)
        i = np.argmax(relative_error)
        print(f'Iter: {self.iter}, conv error: {abs(relative_error[i]):.3E}, tau: {self.tau[i]:.5f} (batch element {i})')
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
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
        plt.show()

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
