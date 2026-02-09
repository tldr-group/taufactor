"""Main module."""
import numpy as np
from abc import ABC, abstractmethod
from IPython.display import clear_output
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import psutil
try:
    import torch
except Exception:
    torch = None
import warnings
from .metrics import extract_through_feature


class SORSolver(ABC):
    """
    A minimal, clean template for SOR solvers.
    Subclasses override a few well-defined hooks.
    Args:
            img: labelled input image defining (non-)conducting phases.
            oemga: Over-relaxation factor for SOR scheme.
            device: The device to perform computations ('cpu' or 'cuda').
    """
    def __init__(self, img: np.ndarray, omega: float | None = None, precision=None, device='cuda'):
        if torch is None:
            raise ImportError(
                "PyTorch is required to use TauFactor solvers. Install pytorch following "
                "https://taufactor.readthedocs.io/en/latest/installation.html"
            )
        self.cpu_img = self._expand_to_4d(img)
        self.batch_size, self.Nx, self.Ny, self.Nz = self.cpu_img.shape
        self.device = self._init_device(device)
        self.precision = precision or torch.float

        # Overrelaxation factor for SOR
        if omega is None:
            omega = 2 - torch.pi / (1.5 * self.Nx)

        # Initialise pytorch tensors
        torch_img = torch.tensor(self.cpu_img, dtype=self.precision, device=self.device)
        mask = self.return_mask(torch_img)
        vol_x = torch.mean(mask, (2, 3)) # volume fraction
        self.field = self.init_field(mask)
        self.factor = self.init_conductive_neighbours(torch_img)

        # Optional for electrode simulations
        reac_nn = self.init_reactive_neighbours(torch_img)
        if reac_nn is not None:
            a_x = (torch.sum(reac_nn, (2, 3)) / (self.Ny*self.Nz*self.dx)) # surface area
            # Pre-compute reaction prefactor
            k_0 = torch.mean(vol_x, 1) / torch.mean(a_x*self.dx, 1) / self.Nx**2
            reac_nn = reac_nn * k_0[:, None, None, None]
            self.factor += reac_nn
            self.factor[self.factor == 0] = torch.inf
            self.a_x = a_x.cpu().numpy()
            self.k_0 = k_0.cpu().numpy()

        self.vol_x = vol_x.cpu().numpy()
        self.cb = self._init_chequerboard(omega)

        # Init params
        self.converged = False
        self.old_tau = -1
        self.iter = 0
        self.tau = None
        self.tau_x = None
        self.D_eff = None

    # ---------------- required hook ----------------
    @abstractmethod
    def return_mask(self, img: torch.Tensor) -> torch.Tensor:
        """Return conductive mask."""
    
    @abstractmethod
    def init_field(self, img: torch.Tensor) -> torch.Tensor:
        """Return initial padded field [bs,Nx+2,Ny+2,Nz+2]."""

    @abstractmethod 
    def init_conductive_neighbours(self, img: torch.Tensor) -> torch.Tensor:
        """N_i: amount of conductive neighbours (cond_nn)"""

    @abstractmethod 
    def compute_metrics(self):
        """Defines tau and relative error"""

    # ---------------- optional hooks --------------
    def init_reactive_neighbours(self, img: torch.Tensor) -> torch.Tensor:
        """S_i: amount of reactive neighbours (reac_nn)"""
        return None
    
    def apply_boundary_conditions(self):
        """Default: Dirichlet in x and no-flux in y and z direction."""
        pass

    def sum_weighted_neighbours(self) -> torch.Tensor:
        """Default: isotropic 6-neighbor SOR increment on interior."""
        sum = self.field[:, 2:, 1:-1, 1:-1] + \
              self.field[:, :-2, 1:-1, 1:-1] + \
              self.field[:, 1:-1, 2:, 1:-1] + \
              self.field[:, 1:-1, :-2, 1:-1] + \
              self.field[:, 1:-1, 1:-1, 2:] + \
              self.field[:, 1:-1, 1:-1, :-2]
        return sum
    
    def plot_stats(self, relative_error):
        """Default: No plotting output."""
        pass
    
    def check_convergence(self, verbose, conv_crit, plot_interval):
        self.tau, relative_error = self.compute_metrics()

        if verbose == 'per_iter':
            # Print stats for slowest converging microstructure
            i = np.argmax(relative_error)
            print(f'Iter: {self.iter}, conv error: {abs(relative_error[i]):.3E}, tau: {self.tau[i]:.5f} (batch element {i})')
            
        if (verbose == 'plot') and (self.iter % (100*plot_interval) == 0):
            self.plot_stats(relative_error)

        if verbose == 'debug':
            self.tau_t.append(self.tau)
            if (self.iter % (100*plot_interval) == 0):
                clear_output(wait=True)
                i = np.argmax(np.abs(relative_error))
                print(f'Iter: {self.iter}, conv error: {np.abs(relative_error[i]):.3E}, tau: {self.tau[i]:.5f} (batch element {i})')
                fig, ax = plt.subplots(figsize=(8,2), dpi=200)
                taus = np.array(self.tau_t)
                x = np.arange(0, taus.shape[0])*100
                for b in range(self.batch_size):
                    ax.plot(x, taus[:,b], label=f'batch_{b}', linestyle='-')
                ax.set_xlabel('iters')
                ax.set_ylabel('tau')
                ax.set_title('Tau convergence')
                ax.set_ylim(np.min(taus)-0.1, np.max(taus)+0.1)
                ax.legend()
                ax.grid()
                plt.show()

        if not np.all(relative_error < conv_crit):
            self.old_tau = self.tau
            return False

        tau_error = np.max(np.abs(self.tau - self.old_tau))
        if not tau_error < 2e-3:
            self.old_tau = self.tau
            return False

        self.tau[self.tau == 0] = np.inf
        return True
    
    # ---------------- main loop -------------------
    def solve(self, iter_limit=10000, verbose=True, conv_crit=1e-2, plot_interval=10):
        """
        Solve steady-state with SOR solver

        :param iter_limit: max iterations before aborting
        :param verbose: Set to 'True', 'per_iter' or 'plot' for more feedback
        :param conv_crit: convergence criteria, minimum percent difference between
        max and min flux through a given layer
        :return: tau
        """
        if (verbose) and (self.device.type == 'cuda'):
            torch.cuda.reset_peak_memory_stats(device=self.device)

        if verbose == 'debug':
            self.tau_t = []

        with torch.no_grad():
            start = timer()
            while not self.converged and self.iter < iter_limit:
                self.apply_boundary_conditions()
                increment = self.sum_weighted_neighbours()
                increment /= self.factor
                increment -= self.field[:, 1:-1, 1:-1, 1:-1]
                # Multiply with checkerboard and over-relaxation factor
                increment *= self.cb[self.iter % 2]
                self.field[:, 1:-1, 1:-1, 1:-1] += increment
                self.iter += 1

                if self.iter % 100 == 0:
                    self.converged = self.check_convergence(verbose, conv_crit, plot_interval)

            self._end_simulation(self.iter, verbose, start)
            if self.tau_x is None:
                return self.tau
            return self.tau_x

    # ---------------- helpers ----------------
    @staticmethod
    def _expand_to_4d(img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            raise TypeError("Error: input image must be a NumPy array!")
        if img.ndim == 2:
            img = img[..., None]
        if img.ndim == 3:
            img = img[None, ...]
        if img.ndim != 4:
            raise ValueError("expected [B, X, Y, Z]")
        return img
    
    @staticmethod
    def _init_device(device: torch.device) -> torch.device:
        # check device is available
        if torch.device(device).type.startswith('cuda') and not torch.cuda.is_available():
            device = torch.device('cpu')
            warnings.warn("CUDA not available, defaulting device to cpu. "
                          "To avoid this warning, explicitly set the device when "
                          "initialising the solver with device='cpu' ")
        else:
            device = torch.device(device)
        return device
    
    @staticmethod
    def _check_binary_labels(img):
        if len(np.unique(img)) > 2 or \
           np.unique(img).max() not in [0, 1] or \
           np.unique(img).min() not in [0, 1]:
            raise ValueError(
                "Input image must only contain 0s and 1s. "
                "Your image must be segmented to use this tool. "
                "If your image has been segmented, ensure your labels are "
                "0 for non-conductive and 1 for conductive phase. "
                f"Your image has the following labels: {np.unique(img)}. "
                "If you have more than one conductive phase, use the multi-phase solver.")

    def _init_chequerboard(self, omega: float):
        """Creates a chequerboard to ensure neighbouring pixels dont update,
        which can cause instability"""
        cb = np.zeros([self.Nx, self.Ny, self.Nz])
        a, b, c = np.meshgrid(range(self.Nx), range(self.Ny), range(self.Nz), indexing='ij')
        cb[(a + b + c) % 2 == 0] = 1
        return [torch.tensor(omega*cb, dtype=self.precision, device=self.device),
                torch.tensor(omega*(1-cb), dtype=self.precision, device=self.device)]

    @staticmethod
    def _pad(img: torch.Tensor, vals=(0,0,0,0,0,0)) -> torch.Tensor:
        """Pads a volume with values"""
        while len(vals) < 6:
            vals.append(0)
        to_pad = [1]*8
        to_pad[-2:] = (0, 0)
        img = torch.nn.functional.pad(img, to_pad, 'constant')
        img[:, 0], img[:, -1] = vals[:2]
        img[:, :, 0], img[:, :, -1] = vals[2:4]
        img[:, :, :, 0], img[:, :, :, -1] = vals[4:]
        return img

    @staticmethod
    def _crop(img: torch.Tensor, c: int=1):
        """removes a layer from the volume edges"""
        return img[:, c:-c, c:-c, c:-c]
    
    @staticmethod
    def _sum_by_rolling(tensor: torch.Tensor):
        """Sum up active neighbours and return new tensor"""
        sum = torch.zeros_like(tensor)
        # iterate through shifts in the spatial dimensions
        for dim in range(1, 4):
            for dr in [1, -1]:
                sum += torch.roll(tensor, dr, dim)
        return sum

    def _end_simulation(self, iterations: int, verbose: bool, start):
        if self.converged:
            msg = 'converged to'
        else:
            print('Warning: not converged')
            msg = 'unconverged value of tau'

        if verbose:
            print(f'{msg}: {self.tau} after: {iterations} iterations in: {np.around(timer() - start, 4)}s ({np.around((timer() - start)/(iterations), 4)} s/iter)')
            if self.device.type == 'cuda':
                print(f"GPU-RAM currently {torch.cuda.memory_allocated(device=self.device) / 1e6:.2f} MB "
                      f"(max allocated {torch.cuda.max_memory_allocated(device=self.device) / 1e6:.2f} MB; "
                      f"{torch.cuda.max_memory_reserved(device=self.device) / 1e6:.2f} MB reserved)")
            elif self.device.type == 'cpu':
                memory_info = psutil.virtual_memory()
                print(f"CPU total memory: {memory_info.total / 1e6:.2f} MB")
                print(f"CPU available memory: {memory_info.available / 1e6:.2f} MB")
                print(f"CPU used memory: {memory_info.used / 1e6:.2f} MB")


class Solver(SORSolver):
    """Two-phase (binary) SOR solver.

    Solves steady-state potential/diffusion on a binary microstructure
    (1 = conductive, 0 = non-conductive) using a Jacobi-like SOR sweep
    with alternating checkerboards. Reports batchwise tortuosity and
    effective diffusivity.

    Args:
        img (numpy.ndarray): Binary image with labels in ``{0, 1}``.
        bc (tuple[float, float], optional): Boundary values
            ``(top_bc, bot_bc)``. Defaults to ``(-0.5, 0.5)``.
        D_0 (float, optional): Reference (mean) diffusivity. Defaults to ``1``.
        device (str | torch.device, optional): Compute device. Defaults to ``'cuda'``.

    Attributes:
        D_0 (float): Reference diffusivity.
        D_mean (float | None): Mean diffusivity used for scaling.
        VF (numpy.ndarray): Volume fraction per batch element.
        D_rel (numpy.ndarray): Relative diffusivity per batch (set during solve).

    Raises:
        ValueError: If labels are not strictly in ``{0, 1}``.
    """

    def __init__(self, img, omega=None, D_0=1, device='cuda'):
        self._check_binary_labels(img)
        self.top_bc, self.bot_bc = (-0.5, 0.5) # boundary conditions
        super().__init__(img, omega=omega, device=device)

        self.D_0 = D_0
        self.VF = np.mean(self.vol_x, axis=1)

    def return_mask(self, img):
        return img

    def init_field(self, mask):
        """Sets an initial linear field across the volume"""
        sh = 1 / (2 * self.Nx)
        vec = torch.linspace(self.top_bc + sh, self.bot_bc - sh, self.Nx,
                             dtype=self.precision, device=self.device)
        for i in range(2):
            vec = torch.unsqueeze(vec, -1)
        vec = torch.unsqueeze(vec, 0)
        vec = vec.repeat(self.batch_size, 1, self.Ny, self.Nz, )
        return self._pad(mask * vec, [2*self.top_bc, 2*self.bot_bc])

    def init_conductive_neighbours(self, mask):
        """Saves the number of conductive neighbours for flux calculation"""
        img2 = self._pad(mask, [2, 2])
        nn = self._sum_by_rolling(img2)
        nn = self._crop(nn, 1)
        # avoid div 0 errors
        nn[mask == 0] = torch.inf
        nn[nn == 0] = torch.inf
        return nn

    def vertical_flux(self) -> torch.Tensor:
        '''Calculates the vertical flux through the volume'''
        # Indexing removes boundary layers (1 layer at every boundary)
        vert_flux = self.field[:, 2:-1, 1:-1, 1:-1] - \
            self.field[:, 1:-2, 1:-1, 1:-1]
        vert_flux[self.factor[:, 0:-1] > 8] = 0
        vert_flux[self.factor[:, 1:] > 8] = 0
        return vert_flux
    
    def compute_metrics(self):
        vertical_flux = self.vertical_flux()
        # Sum over the y and z dimensions only, leaving a (bs, x) result.
        self.flux_1d = torch.sum(vertical_flux, (2, 3)).cpu().numpy() # (bs, x)
        fl_max = np.max(self.flux_1d, axis=1)  # shape: (bs,)
        fl_min = np.min(self.flux_1d, axis=1)  # shape: (bs,)
        mean_fl = np.mean(self.flux_1d, axis=1)   # shape: (bs,)
        relative_error = np.divide(fl_max - fl_min, fl_max,
            out=np.full_like(fl_max, np.nan), where=fl_max != 0)

        self.D_rel = mean_fl * self.Nx / (self.Ny * self.Nz)\
                     / abs(self.top_bc - self.bot_bc)
        tau = np.divide(self.VF, self.D_rel,
            out=np.full_like(self.D_rel, np.nan), where=self.D_rel != 0)

        fluxes = -self.field[:, 2:-1, 1:-1, 1:-1] + self.field[:, 1:-2, 1:-1, 1:-1]
        fluxes[self.field[:, 2:-1, 1:-1, 1:-1] == 0] = 0
        fluxes[self.field[:, 1:-2, 1:-1, 1:-1] == 0] = 0
        fluxes = torch.mean(fluxes, (2, 3)).cpu().numpy()
        c_x = torch.mean(self.field[:, 1:-1, 1:-1, 1:-1], (2, 3)).cpu().numpy()
        c_x = np.divide(c_x, self.vol_x, out=np.zeros_like(self.vol_x),
                        where=self.vol_x != 0)
        self.c_x = c_x
        fluxes_1d = c_x[:,:-1] - c_x[:,1:]
        fluxes_1d[:,:][self.vol_x[:,1:]==0] = 0
        fluxes_1d[:,:][self.vol_x[:,:-1]==0] = 0
        eps = 0.5*(self.vol_x[:,:-1] + self.vol_x[:,1:])
        self.tau_x = np.divide(eps * fluxes_1d, fluxes,
            out=np.full_like(fluxes_1d, np.nan), where=fluxes != 0)

        for b in range(self.batch_size):
            if (fl_min[b] == 0) or (mean_fl[b] == 0):
                _ , frac = extract_through_feature(self.cpu_img[b]>0, 1, 'x')
                if frac == 0:
                    print(f"Warning: batch element {b} has no percolating path!")
                    relative_error[b] = 0 # Set to converged
                    self.D_rel[b] = 0
                    tau[b] = 0
                    self.tau_x[b,:] = 0
        # If NaN values occuring set to converged to stop
        relative_error[np.isnan(mean_fl)] = 0
        self.D_eff = self.D_0*self.D_rel
        return tau, relative_error
    
    def plot_stats(self, relative_error):
        clear_output(wait=True)
        i = np.argmax(relative_error)
        print(f'Iter: {self.iter}, conv error: {abs(relative_error[i]):.3E}, tau: {self.tau[i]:.5f} (batch element {i})')
        mean = np.expand_dims(np.mean(self.flux_1d, axis=1), 1)
        rel_fluxes = ((self.flux_1d - mean)/mean)
        fig, ax = plt.subplots(figsize=(8,2), dpi=200)
        x = np.arange(0, rel_fluxes.shape[1])+0.5
        for b in range(self.batch_size):
            ax.plot(x, rel_fluxes[b], label=f'batch_{b}', linestyle='-')

        ax.set_xlabel('voxels in x')
        ax.set_ylabel('relative fluxes')
        ax.set_title(f'Relative flux convergence in flux direction in iter {self.iter}')
        ax.set_ylim(-0.1, 0.1)
        ax.legend()
        ax.grid()
        plt.show()


class AnisotropicSolver(Solver):
    """Anisotropic SOR solver with voxel-spacing corrections.

    Scales neighbour contributions to account for non-cubic voxels such
    as in FIB-SEM stacks (different spacing in cutting direction).
    Y-neighbors are scaled by ``(dx/dy)^2`` and Z-neighbors by
    ``(dx/dz)^2``.

    Args:
        img (numpy.ndarray): Binary input image.
        spacing (tuple[float, float, float]): Voxel spacing ``(dx, dy, dz)``.
        bc (tuple[float, float], optional): Boundary values.
            Defaults to ``(-0.5, 0.5)``.
        D_0 (float, optional): Reference diffusivity. Defaults to ``1``.
        device (str | torch.device, optional): Compute device. Defaults to ``'cuda'``.

    Attributes:
        Ky (float): Anisotropy weight for Y neighbors (``(dx/dy)^2``).
        Kz (float): Anisotropy weight for Z neighbors (``(dx/dz)^2``).

    Raises:
        ValueError: If ``spacing`` is not a length-3 numeric tuple.
        UserWarning: If spacing anisotropy is very large.
    """

    def __init__(self, img, spacing, omega=None, D_0=1, device=torch.device('cuda:0')):
        if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
            raise ValueError("spacing must be a list or tuple with three elements (dx, dy, dz)")
        if not all(isinstance(x, (int, float)) for x in spacing):
            raise ValueError("All elements in spacing must be integers or floats")
        if (np.max(spacing)/np.min(spacing) > 10):
            warnings.warn("This computation is very questionable for largely different spacings e.g. dz >> dx.")
        dx, dy, dz = spacing
        self.Ky = (dx/dy)**2
        self.Kz = (dx/dz)**2
        super().__init__(img, omega=omega, D_0=D_0, device=device)

    def init_conductive_neighbours(self, img):
        """Saves the number of conductive neighbours for flux calculation"""
        img2 = self._pad(img, [2, 2])
        nn = torch.zeros_like(img2, dtype=self.precision)
        # iterate through shifts in the spatial dimensions
        factor = [1.0, self.Ky, self.Kz]
        for dim in range(1, 4):
            for dr in [1, -1]:
                nn += torch.roll(img2, dr, dim)*factor[dim-1]
        nn = self._crop(nn, 1)
        nn[img == 0] = torch.inf
        nn[nn == 0] = torch.inf
        return nn
    
    def sum_weighted_neighbours(self):
        """Default: isotropic 6-neighbor SOR increment on interior."""
        sum = self.field[:, 2:, 1:-1, 1:-1] + self.field[:, :-2, 1:-1, 1:-1] + \
              self.Ky*(self.field[:, 1:-1, 2:, 1:-1] + self.field[:, 1:-1, :-2, 1:-1]) + \
              self.Kz*(self.field[:, 1:-1, 1:-1, 2:] + self.field[:, 1:-1, 1:-1, :-2])
        return sum


class PeriodicSolver(Solver):
    """Two-phase SOR solver with periodic Y/Z boundaries.

    Uses periodic wrapping for neighbor evaluation in Y and Z and
    reapplies periodic boundary conditions to the field each iteration.
    X remains the flux/open direction.

    Notes:
        Overrides ``init_nn`` and ``apply_boundary_conditions`` from
        :class:`Solver`.
    """

    def init_conductive_neighbours(self, img):
        img2 = self._pad(img, [2, 2])[:, :, 1:-1, 1:-1]
        nn = self._sum_by_rolling(img2)
        nn = nn[:, 1:-1]
        nn[img == 0] = torch.inf
        nn[nn == 0] = torch.inf
        return nn

    def apply_boundary_conditions(self):
        self.field[:,:,0,:] = self.field[:,:,-2,:]
        self.field[:,:,-1,:] = self.field[:,:,1,:]
        self.field[:,:,:,0] = self.field[:,:,:,-2]
        self.field[:,:,:,-1] = self.field[:,:,:,1]


class MultiPhaseSolver(SORSolver):
    """Multi-phase SOR solver with per-phase conductivities.

    Supports multiple conductive labels with different conductivities
    and uses harmonic-mean pair weights in the update stencil. Currently
    implemented for batch size of 1.

    Args:
        img (numpy.ndarray): Labeled image; 0 = non-conductive.
        cond (dict[int, float], optional): Map ``label -> conductivity``.
            Defaults to ``{1: 1}``.
        bc (tuple[float, float], optional): Boundary values.
            Defaults to ``(-0.5, 0.5)``.
        device (str | torch.device, optional): Compute device. Defaults to ``'cuda'``.

    Attributes:
        cond (dict[int, float]): Internal map of label to resistance half-weights.
        pre_factors (list[torch.Tensor]): Directional pre-factors for the stencil.
        VF (dict[int, float]): Volume fraction per label.
        D_mean (float): Phase-weighted mean diffusivity.
        D_eff (torch.Tensor | float | None): Effective diffusivity.
        tau (torch.Tensor | float | None): Tortuosity.

    Raises:
        ValueError: If conductivity for any label is 0, or if label 0
            is included as conductive.
        TypeError: If batch size is greater than 1.
    """

    def __init__(self, img, cond={1: 1}, device='cuda'):
        if (0 in cond.values()):
            raise ValueError(
                '0 conductivity phase: non-conductive phase should be labelled 0 in the input image and ommitted from the cond argument')
        if (0 in cond.keys()):
            raise ValueError(
                '0 cannot be used as a conductive phase label, please use a positive integer and leave 0 for non-conductive phase')
        self.cond = {ph: 0.5 / c for ph, c in cond.items()}

        # Boundary conditions
        self.top_bc, self.bot_bc = (-0.5, 0.5)
        super().__init__(img, device=device)
        if self.batch_size > 1:
            raise TypeError('Error: The MultiPhaseSolver is only implemented for batch_size=1!')
        # TODO: what is this?
        self.pre_factors = self.factor[1:]
        self.factor = self.factor[0]

        # TODO: counts of unique instead of loop
        self.VF = {p: np.mean(img == p)
                   for p in np.unique(img)}

        if len(np.array([self.VF[z] for z in self.VF.keys() if z != 0])) > 0:
            self.D_mean = np.sum(
                np.array([self.VF[z]*(1/(2*self.cond[z])) for z in self.VF.keys() if z != 0]))
        else:
            self.D_mean = 0

    def return_mask(self, img):
        mask = torch.zeros_like(img)
        mask[img > 0] = 1
        return mask

    def init_field(self, mask):
        sh = 1 / (self.Nx + 1)
        vec = torch.linspace(self.top_bc + sh, self.bot_bc - sh, self.Nx,
                             dtype=self.precision, device=self.device)
        for i in range(2):
            vec = torch.unsqueeze(vec, -1)
        vec = torch.unsqueeze(vec, 0)
        vec = vec.repeat(self.batch_size, 1, self.Ny, self.Nz)
        return self._pad(mask * vec, [self.top_bc, self.bot_bc])

    def init_conductive_neighbours(self, img):
        # conductivity map
        img2 = torch.zeros_like(img)
        for ph in self.cond:
            c = self.cond[ph]
            img2[img == ph] = c
        img2 = self._pad(self._pad(img2))
        img2[:, 1] = img2[:, 2]
        img2[:, -2] = img2[:, -3]
        nn = torch.zeros_like(img2, dtype=self.precision)
        # iterate through shifts in the spatial dimensions
        nn_list = []
        for dim in range(1, 4):
            for dr in [1, -1]:
                shift = torch.roll(img2, dr, dim)
                sum = img2 + shift
                sum[shift == 0] = 0
                sum[img2 == 0] = 0
                sum = 1/sum
                sum[sum == torch.inf] = 0
                nn += sum
                nn_list.append(self._crop(sum, 1).to(self.device))
        # remove the two paddings
        nn = self._crop(nn, 2)
        # avoid div 0 errors
        nn[img == 0] = torch.inf
        nn[nn == 0] = torch.inf
        nn_list.insert(0, nn.to(self.device))
        return nn_list
    
    def sum_weighted_neighbours(self) -> torch.Tensor:
        sum = self.field[:, 2:, 1:-1, 1:-1] * self.pre_factors[0][:, 2:, 1:-1, 1:-1] + \
              self.field[:, :-2, 1:-1, 1:-1] * self.pre_factors[1][:, :-2, 1:-1, 1:-1] + \
              self.field[:, 1:-1, 2:, 1:-1] * self.pre_factors[2][:, 1:-1, 2:, 1:-1] + \
              self.field[:, 1:-1, :-2, 1:-1] * self.pre_factors[3][:, 1:-1, :-2, 1:-1] + \
              self.field[:, 1:-1, 1:-1, 2:] * self.pre_factors[4][:, 1:-1, 1:-1, 2:] + \
              self.field[:, 1:-1, 1:-1, :-2] * self.pre_factors[5][:, 1:-1, 1:-1, :-2]
        return sum

    def vertical_flux(self):
        '''Calculates the vertical flux through the volume'''
        vert_flux = (self.field[:, 2:-1, 1:-1, 1:-1] - \
                     self.field[:, 1:-2, 1:-1, 1:-1]) * \
                    self.pre_factors[1][:, 1:-2, 1:-1, 1:-1]
        vert_flux[self.factor[:,1:] == torch.inf] = 0
        return vert_flux
    
    def compute_metrics(self):
        vertical_flux = self.vertical_flux()
        # Sum over the y and z dimensions only, leaving a (bs, x) result.
        self.flux_1d = torch.sum(vertical_flux, (2, 3)).cpu().numpy() # (bs, x)
        fl_max = np.max(self.flux_1d, axis=1)  # shape: (bs,)
        fl_min = np.min(self.flux_1d, axis=1)  # shape: (bs,)
        mean_fl = np.mean(self.flux_1d, axis=1)   # shape: (bs,)
        relative_error = np.divide(fl_max - fl_min, fl_max,
            out=np.full_like(fl_max, np.nan), where=fl_max != 0)

        self.D_eff = mean_fl * (self.Nx+1) / (self.Ny * self.Nz)\
                     / abs(self.top_bc - self.bot_bc)
        tau = np.divide(self.D_mean, self.D_eff,
                        out=np.full_like(self.D_eff, np.nan),
                        where=self.D_eff != 0)
      
        for b in range(self.batch_size):
            if (fl_min[b] == 0) or (mean_fl[b] == 0):
                _ , frac = extract_through_feature(self.cpu_img[b]>0, 1, 'x')
                if frac == 0:
                    print(f"Warning: batch element {b} has no percolating path!")
                    relative_error[b] = 0 # Set to converged
                    self.D_eff[b] = 0
                    tau[b] = 0
        # If NaN values occuring set to converged to stop
        relative_error[np.isnan(relative_error)] = 0
        return tau, relative_error
    
    def plot_stats(self, relative_error):
        clear_output(wait=True)
        i = np.argmax(relative_error)
        print(f'Iter: {self.iter}, conv error: {abs(relative_error[i]):.3E}, tau: {self.tau[i]:.5f} (batch element {i})')
        mean = np.expand_dims(np.mean(self.flux_1d, axis=1), 1)
        rel_fluxes = ((self.flux_1d - mean)/mean)
        fig, ax = plt.subplots(figsize=(8,2), dpi=200)
        x = np.arange(0, rel_fluxes.shape[1])+0.5
        for b in range(self.batch_size):
            ax.plot(x, rel_fluxes[b], label=f'batch_{b}', linestyle='-')

        ax.set_xlabel('voxels in x')
        ax.set_ylabel('relative fluxes')
        ax.set_title(f'Relative flux convergence in flux direction in iter {self.iter}')
        ax.set_ylim(-0.1, 0.1)
        ax.legend()
        ax.grid()
        plt.show()
