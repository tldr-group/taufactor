"""Main module."""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

try:
    import torch
except ImportError:
    torch = None

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
    def __init__(self, img: np.ndarray, omega: float | None = None, precision=None,
                 device='cuda', chunk_size: int | None = None):
        if torch is None:
            raise ImportError(
                "PyTorch is required to use TauFactor solvers. Install pytorch following "
                "https://taufactor.readthedocs.io/en/latest/installation.html"
            )
        self.cpu_img = self._expand_to_4d(img)
        self.batch_size, self.Nx, self.Ny, self.Nz = self.cpu_img.shape
        self.device = self._init_device(device)
        self.precision = precision or torch.float
        self._chunk_size = self._init_chunk_size(chunk_size)

        # Overrelaxation factor for SOR
        if omega is None:
            omega = 2 - torch.pi / (1.5 * self.Nx)
        self.omega = float(omega)

        # Labels as uint8 on device (4x smaller than float32) to cut init peak VRAM
        torch_img = torch.as_tensor(self.cpu_img, device=self.device)
        if torch_img.dtype != torch.uint8:
            torch_img = torch_img.to(torch.uint8)
        # Conductive mask stays boolean; the neighbour sums and the initial field
        # cast it on the fly, so no float copy of the volume is ever held.
        mask = self.return_mask(torch_img)
        vol_x = self._slice_volume_fraction(mask)

        # Reactive neighbours before field so we can free torch_img early
        reac_nn = self.init_reactive_neighbours(torch_img)
        self.factor = self.init_conductive_neighbours(torch_img, mask)
        del torch_img
        if reac_nn is not None:
            a_x = torch.sum(reac_nn, (2, 3), dtype=self.precision) / (self.Ny * self.Nz * self.dx)
            k_0 = torch.mean(vol_x, 1) / torch.mean(a_x * self.dx, 1) / self.Nx**2
            # Scale into factor via alpha so reac_nn is never widened to float
            for b in range(self.batch_size):
                self.factor[b].add_(reac_nn[b], alpha=float(k_0[b]))
            del reac_nn
            self.factor.masked_fill_(self.factor == 0, torch.inf)
            self.a_x = a_x.cpu().numpy()
            self.k_0 = k_0.cpu().numpy()

        # Init allocates and frees several differently sized volumes, which leaves
        # the caching allocator fragmented; for large images the reserved pool ends
        # up far above what is live. Return the freed blocks before the largest
        # allocation, and again once the temporaries are gone.
        self._release_cached_memory()
        self.field = self.init_field(mask)
        self.vol_x = vol_x.cpu().numpy()
        del mask, vol_x
        self._release_cached_memory()

        # Init params
        self.converged = False
        self.old_tau = 0
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
    def init_conductive_neighbours(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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

    def sum_weighted_neighbours(self, out: torch.Tensor, x0: int = 0) -> None:
        """Isotropic 6-neighbor sum into a preallocated interior buffer.

        ``out`` covers interior x slices ``[x0, x0 + out.shape[1])``.
        """
        x1 = x0 + out.shape[1]
        torch.add(self.field[:, x0 + 2:x1 + 2, 1:-1, 1:-1],
                  self.field[:, x0:x1, 1:-1, 1:-1], out=out)
        out.add_(self.field[:, x0 + 1:x1 + 1, 2:, 1:-1])
        out.add_(self.field[:, x0 + 1:x1 + 1, :-2, 1:-1])
        out.add_(self.field[:, x0 + 1:x1 + 1, 1:-1, 2:])
        out.add_(self.field[:, x0 + 1:x1 + 1, 1:-1, :-2])

    def _apply_chequerboard(self, increment: torch.Tensor, x0: int = 0) -> None:
        """Zero the inactive colour and scale by omega, in-place.

        The active colour is (i+j+k) % 2 == iter % 2 in global interior indices,
        selected through strided views so no chequerboard mask is stored.
        """
        for a in (0, 1):
            for b in (0, 1):
                z0 = (a + b + x0 + self.iter + 1) % 2
                increment[:, a::2, b::2, z0::2] = 0
        increment.mul_(self.omega)

    def plot_stats(self, relative_error):
        """Default: No plotting output."""

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
                _, ax = plt.subplots(figsize=(8,2), dpi=200)
                taus = np.array(self.tau_t)
                x = np.arange(0, taus.shape[0])*100
                min_tau, max_tau = 1, 1
                for b in range(self.batch_size):
                    if relative_error[b] > 0:
                        ax.plot(x, taus[:,b], label=f'batch_{b}', linestyle='-')
                        min_tau = np.min([np.min(taus[:,b]), min_tau])
                        max_tau = np.max([np.max(taus[:,b]), max_tau])
                ax.set_xlabel('iters')
                ax.set_ylabel('tau')
                ax.set_title('Tau convergence')
                ax.set_ylim(min_tau-0.1, max_tau+0.1)
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
            chunk = self._chunk_size
            increment = torch.empty(
                (self.batch_size, chunk, self.Ny, self.Nz),
                dtype=self.field.dtype,
                device=self.device,
            )
            start = timer()
            while not self.converged and self.iter < iter_limit:
                self.apply_boundary_conditions()
                # Sweeping in x slabs is exact for a chequerboard update: every
                # neighbour of an active voxel has the opposite colour, so it is
                # never written during the same iteration.
                for x0 in range(0, self.Nx, chunk):
                    x1 = min(self.Nx, x0 + chunk)
                    buf = increment[:, :x1 - x0]
                    self.sum_weighted_neighbours(buf, x0)
                    buf /= self.factor[:, x0:x1]
                    buf -= self.field[:, x0 + 1:x1 + 1, 1:-1, 1:-1]
                    self._apply_chequerboard(buf, x0)
                    self.field[:, x0 + 1:x1 + 1, 1:-1, 1:-1] += buf
                self.iter += 1

                if self.iter % 100 == 0:
                    self.converged = self.check_convergence(verbose, conv_crit, plot_interval)

            self.walltime = timer() - start
            self._end_simulation(self.iter, verbose)
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
    def _init_device(device) -> torch.device:
        # check device is available
        if torch.device(device).type.startswith('cuda') and not torch.cuda.is_available():
            device = torch.device('cpu')
            warnings.warn("CUDA not available, defaulting device to cpu. "
                          "To avoid this warning, explicitly set the device when "
                          "initialising the solver with device='cpu' ")
        else:
            device = torch.device(device)
        return device

    def _release_cached_memory(self):
        """Hand cached-but-unused device blocks back to the driver."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _init_chunk_size(self, chunk_size: int | None) -> int:
        """Number of x slices handled per sweep chunk.

        Chunking bounds the scratch buffers to a fixed budget rather than a
        multiple of the volume, which is what lets large volumes fit at all.
        """
        if chunk_size is not None:
            return max(1, min(self.Nx, int(chunk_size)))
        budget = 64 * 2**20  # bytes of scratch per sweep chunk
        per_slice = max(1, self.batch_size * self.Ny * self.Nz * self.precision.itemsize)
        n = min(self.Nx, max(2, budget // per_slice))
        # Keep chunks even so a chunk's local parity matches the global one
        return int(n) if n % 2 == 0 or n == self.Nx else int(n) - 1

    def _init_field_from_profile(self, mask: torch.Tensor, profile: torch.Tensor,
                                x_bcs) -> torch.Tensor:
        """Allocate the padded field and fill its interior with ``mask * profile``.

        ``profile`` is the 1D initial profile along x. The product is written
        straight into the padded buffer, so no full-volume temporary is created.
        """
        field = torch.zeros(
            (self.batch_size, self.Nx + 2, self.Ny + 2, self.Nz + 2),
            dtype=profile.dtype,
            device=self.device,
        )
        interior = field[:, 1:-1, 1:-1, 1:-1]
        interior.copy_(mask)  # casts in the kernel, no widened copy of mask
        interior.mul_(profile[:, None, None])
        field[:, 0], field[:, -1] = x_bcs
        field[:, :, 0], field[:, :, -1] = 0, 0
        field[:, :, :, 0], field[:, :, :, -1] = 0, 0
        return field

    @staticmethod
    def _slice_volume_fraction(mask: torch.Tensor) -> torch.Tensor:
        """Per-x-slice volume fraction of a boolean mask, without a float copy."""
        ny, nz = mask.shape[2], mask.shape[3]
        return torch.sum(mask, (2, 3), dtype=torch.float32) / (ny * nz)

    @staticmethod
    def _neighbour_sum(src: torch.Tensor, out: torch.Tensor, x_ghosts=(0, 0)) -> None:
        """6-neighbour sum of ``src`` into ``out``, both shaped [bs,Nx,Ny,Nz].

        Equivalent to summing over a +1 constant pad with ``x_ghosts`` on the x
        faces and zero on y/z, but without materialising the padded copy.
        ``src`` may be a narrower dtype than ``out`` (e.g. bool into uint8).
        """
        lo, hi = x_ghosts
        out[:, :-1].copy_(src[:, 1:])
        out[:, -1] = hi
        out[:, 1:].add_(src[:, :-1])
        out[:, 0] += lo
        out[:, :, 1:].add_(src[:, :, :-1])
        out[:, :, :-1].add_(src[:, :, 1:])
        out[:, :, :, 1:].add_(src[:, :, :, :-1])
        out[:, :, :, :-1].add_(src[:, :, :, 1:])

    @staticmethod
    def _periodic_yz_neighbour_sum(src: torch.Tensor, out: torch.Tensor,
                                   x_ghosts=(0, 0)) -> None:
        """As :meth:`_neighbour_sum` but with periodic y/z instead of zero pad."""
        lo, hi = x_ghosts
        out[:, :-1].copy_(src[:, 1:])
        out[:, -1] = hi
        out[:, 1:].add_(src[:, :-1])
        out[:, 0] += lo
        out[:, :, 1:].add_(src[:, :, :-1])
        out[:, :, :-1].add_(src[:, :, 1:])
        out[:, :, 0].add_(src[:, :, -1])
        out[:, :, -1].add_(src[:, :, 0])
        out[:, :, :, 1:].add_(src[:, :, :, :-1])
        out[:, :, :, :-1].add_(src[:, :, :, 1:])
        out[:, :, :, 0].add_(src[:, :, :, -1])
        out[:, :, :, -1].add_(src[:, :, :, 0])

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
    
    def _end_simulation(self, iterations: int, verbose: bool):
        if self.converged:
            msg = "converged to"
        else:
            print("Warning: not converged")
            msg = "unconverged value of tau"

        if verbose:
            print(f"{msg}: {self.tau} after: {iterations} iterations in: "
                  f"{np.around(self.walltime, 4)} s "
                  f"({np.around(self.walltime/iterations, 4)} s/iter)")
            if self.device.type == 'cuda':
                print(f"GPU-RAM currently {torch.cuda.memory_allocated(device=self.device) / 1e6:.2f} MB "
                      f"(max allocated {torch.cuda.max_memory_allocated(device=self.device) / 1e6:.2f} MB; "
                      f"{torch.cuda.max_memory_reserved(device=self.device) / 1e6:.2f} MB reserved)")


class ThroughTransportSolver(SORSolver):
    """Solver for through-transport with open boundaries in x direction.

    Uses Dirichlet boundary conditions in x to calculate tortuosity from
    staedy-state fluxes.
    """
    def __init__(self, img, omega = None, precision=None, device='cuda'):
        self.top_bc, self.bot_bc = (-0.5, 0.5) # boundary conditions
        super().__init__(img, omega, precision, device)

    def init_field(self, mask):
        """Sets an initial linear field across the volume"""
        sh = 1 / (2 * self.Nx)
        vec = torch.linspace(self.top_bc + sh, self.bot_bc - sh, self.Nx,
                             dtype=self.precision, device=self.device)
        return self._init_field_from_profile(
            mask, vec, (2 * self.top_bc, 2 * self.bot_bc))

    def compute_metrics(self):
        vertical_flux = self.vertical_flux()
        # Sum over the y and z dimensions only, leaving a (bs, x) result.
        self.flux_1d = torch.mean(vertical_flux, (2, 3)).cpu().numpy() # (bs, x)
        fl_max = np.max(self.flux_1d, axis=1)  # shape: (bs,)
        fl_min = np.min(self.flux_1d, axis=1)  # shape: (bs,)
        mean_fl = np.mean(self.flux_1d, axis=1)   # shape: (bs,)
        relative_error = np.divide(fl_max - fl_min, fl_max,
            out=np.full_like(fl_max, np.nan), where=fl_max != 0)

        D_rel = mean_fl * self.Nx / abs(self.top_bc - self.bot_bc)
        tau = np.divide(self.D_mean, D_rel,
            out=np.full_like(D_rel, np.nan), where=D_rel != 0)

        c_x = torch.mean(self.field[:, 1:-1, 1:-1, 1:-1], (2, 3)).cpu().numpy()
        c_x = np.divide(c_x, self.vol_x, out=np.zeros_like(self.vol_x),
                        where=self.vol_x != 0)
        self.c_x = c_x
        flux_from_c = c_x[:,1:] - c_x[:,:-1]
        flux_from_c[:,:][self.vol_x[:,1:]==0] = 0
        flux_from_c[:,:][self.vol_x[:,:-1]==0] = 0
        eps = 0.5*(self.vol_x[:,:-1] + self.vol_x[:,1:])
        self.tau_x = np.divide(eps * flux_from_c, self.flux_1d,
            out=np.full_like(flux_from_c, np.nan), where=self.flux_1d != 0)

        for b in range(self.batch_size):
            if (fl_min[b] == 0) or (fl_max[b] == 0) or (mean_fl[b] == 0):
                conductive_mask = np.isin(self.cpu_img[b], self.conductive_labels)
                _ , frac = extract_through_feature(conductive_mask, 1, 'x')
                if frac == 0:
                    print(f"Warning: batch element {b} has no percolating path!")
                    relative_error[b] = 0 # Set to converged
                    D_rel[b] = 0
                    tau[b] = 0
                    self.tau_x[b,:] = 0
        # If NaN values occuring set to converged to stop
        relative_error[np.isnan(mean_fl)] = 0
        self.D_eff = self.D_0*D_rel
        return tau, relative_error

    def plot_stats(self, relative_error):
        """Plot relative fluxes across x direction to visualize convergence."""
        clear_output(wait=True)
        i = np.argmax(relative_error)
        print(f'Iter: {self.iter}, conv error: {abs(relative_error[i]):.3E}, tau: {self.tau[i]:.5f} (batch element {i})')
        mean = np.expand_dims(np.mean(self.flux_1d, axis=1), 1)
        rel_fluxes = ((self.flux_1d - mean)/mean)
        _, ax = plt.subplots(figsize=(8,2), dpi=200)
        x = np.arange(0, rel_fluxes.shape[1])+0.5
        for b in range(self.batch_size):
            if relative_error[b] > 0:
                ax.plot(x, rel_fluxes[b], label=f'batch_{b}', linestyle='-')

        ax.set_xlabel('voxels in x')
        ax.set_ylabel('relative fluxes')
        ax.set_title(f'Relative flux convergence in flux direction in iter {self.iter}')
        ax.set_ylim(-0.1, 0.1)
        ax.legend()
        ax.grid()
        plt.show()


class Solver(ThroughTransportSolver):
    """Two-phase (binary) through-transport solver.

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
        self.conductive_labels = [1]
        super().__init__(img, omega=omega, device=device)
        self.D_0 = D_0
        self.D_mean = np.mean(self.vol_x, axis=1)

    def _check_binary_labels(self, img):
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

    def return_mask(self, img):
        return img.to(dtype=torch.bool)

    def init_conductive_neighbours(self, img, mask):
        """Saves the number of conductive neighbours for flux calculation"""
        nn = torch.empty(mask.shape, dtype=self.precision, device=self.device)
        self._neighbour_sum(mask, nn, (2, 2))
        # avoid div 0 errors
        nn.masked_fill_(~mask, torch.inf)
        nn.masked_fill_(nn == 0, torch.inf)
        return nn

    def vertical_flux(self) -> torch.Tensor:
        '''Calculates the vertical flux through the volume'''
        # Indexing removes boundary layers (1 layer at every boundary)
        vert_flux = self.field[:, 2:-1, 1:-1, 1:-1] - \
            self.field[:, 1:-2, 1:-1, 1:-1]
        vert_flux[self.factor[:, 0:-1] > 8] = 0
        vert_flux[self.factor[:, 1:] > 8] = 0
        return vert_flux


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

    def __init__(self, img, spacing, omega=None, D_0=1, device='cuda:0'):
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

    def init_conductive_neighbours(self, img, mask):
        """Saves the number of conductive neighbours for flux calculation"""
        nn = torch.empty(mask.shape, dtype=self.precision, device=self.device)
        nn[:, :-1].copy_(mask[:, 1:])
        nn[:, -1] = 2
        nn[:, 1:].add_(mask[:, :-1])
        nn[:, 0] += 2
        nn[:, :, 1:].add_(mask[:, :, :-1], alpha=self.Ky)
        nn[:, :, :-1].add_(mask[:, :, 1:], alpha=self.Ky)
        nn[:, :, :, 1:].add_(mask[:, :, :, :-1], alpha=self.Kz)
        nn[:, :, :, :-1].add_(mask[:, :, :, 1:], alpha=self.Kz)
        nn.masked_fill_(~mask, torch.inf)
        nn.masked_fill_(nn == 0, torch.inf)
        return nn

    def sum_weighted_neighbours(self, out, x0: int = 0):
        """Anisotropic 6-neighbor sum into a preallocated interior buffer."""
        x1 = x0 + out.shape[1]
        torch.add(self.field[:, x0 + 2:x1 + 2, 1:-1, 1:-1],
                  self.field[:, x0:x1, 1:-1, 1:-1], out=out)
        out.add_(self.field[:, x0 + 1:x1 + 1, 2:, 1:-1], alpha=self.Ky)
        out.add_(self.field[:, x0 + 1:x1 + 1, :-2, 1:-1], alpha=self.Ky)
        out.add_(self.field[:, x0 + 1:x1 + 1, 1:-1, 2:], alpha=self.Kz)
        out.add_(self.field[:, x0 + 1:x1 + 1, 1:-1, :-2], alpha=self.Kz)


class PeriodicSolver(Solver):
    """Two-phase SOR solver with periodic Y/Z boundaries.

    Uses periodic wrapping for neighbor evaluation in Y and Z and
    reapplies periodic boundary conditions to the field each iteration.
    X remains the flux/open direction.

    Notes:
        Overrides ``init_nn`` and ``apply_boundary_conditions`` from
        :class:`Solver`.
    """

    def init_conductive_neighbours(self, img, mask):
        nn = torch.empty(mask.shape, dtype=self.precision, device=self.device)
        self._periodic_yz_neighbour_sum(mask, nn, (2, 2))
        nn.masked_fill_(~mask, torch.inf)
        nn.masked_fill_(nn == 0, torch.inf)
        return nn

    def apply_boundary_conditions(self):
        self.field[:,:,0,:] = self.field[:,:,-2,:]
        self.field[:,:,-1,:] = self.field[:,:,1,:]
        self.field[:,:,:,0] = self.field[:,:,:,-2]
        self.field[:,:,:,-1] = self.field[:,:,:,1]


class MultiPhaseSolver(ThroughTransportSolver):
    """Multi-phase SOR solver with per-phase conductivity/diffusivity.

    Supports multiple labels with user-defined diffusivities and uses
    harmonic-mean pair weights in the update stencil. Labels omitted from
    ``diffusivities`` are treated as isolating with a warning.

    Args:
        img (numpy.ndarray): Labeled image.
        diffusivities (dict[int, float], optional): Map ``label -> diffusivity``.
            Diffusivity can be zero for any label (including label 0).
            Labels not provided are assumed isolating.
            Defaults to ``{1: 1}``.
        device (str | torch.device, optional): Compute device. Defaults to ``'cuda'``.

    Attributes:
        diffusivities (dict[int, float]): Internal map of label to diffusivity.
        pre_factors (list[torch.Tensor]): Directional pre-factors for the stencil.
        VF (dict[int, numpy.ndarray]): Volume fraction per label and batch.
        D_mean (numpy.ndarray): Phase-weighted mean diffusivity per batch.
        D_eff (torch.Tensor | float | None): Effective diffusivity.
        tau (torch.Tensor | float | None): Tortuosity.

    Raises:
        ValueError: If any diffusivity is negative or non-finite.
    """

    def __init__(self, img, diffusivities=None, D_scaling=1, omega=None, device='cuda'):
        # Validate diffusivities
        if diffusivities is None:
            diffusivities = {0: 0, 1: 1}
        if not isinstance(diffusivities, dict):
            raise TypeError("diffusivities must be a dictionary mapping phase labels to diffusivities")
        for phase, D_p in diffusivities.items():
            if not isinstance(phase, (int, np.integer)):
                raise TypeError(f"Phase label must be integer, got {type(phase).__name__}")
            D_p = float(D_p)
            if (not np.isfinite(D_p)) or (D_p < 0):
                raise ValueError(f"Diffusivity for label {phase} must be finite and >= 0, got {D_p}")
        self.Ds = diffusivities

        # Check for missing labels in cond and warn
        missing_labels = sorted(int(lbl) for lbl in np.unique(img) if lbl not in self.Ds)
        if missing_labels:
            warnings.warn(
                "No diffusivity provided for phase label(s) "
                f"{missing_labels}; assuming these phases are isolating.",
                UserWarning,
            )
            for lbl in missing_labels:
                self.Ds[lbl] = 0.0
        # Make list of conductive labels
        self.conductive_labels = [lbl for lbl, D_p in self.Ds.items() if D_p > 0]

        # Boundary conditions
        super().__init__(img, omega=omega, device=device)
        self.VF = {
            int(p): np.mean(self.cpu_img == p, axis=(1, 2, 3))
            for p in np.unique(self.cpu_img)
        }
        self.D_0 = D_scaling
        self.D_mean = np.sum([self.VF[z] * self.Ds.get(z, 0.0) for z in self.VF], axis=0)

    def return_mask(self, img):
        if len(self.conductive_labels) == 0:
            return torch.zeros(img.shape, dtype=torch.bool, device=img.device)
        conductive = torch.tensor(self.conductive_labels, dtype=img.dtype, device=self.device)
        return torch.isin(img, conductive)

    
    def _harmonic_mean(self, a, b):
        """Calculate the harmonic mean of two tensors, avoiding div-by-zero."""
        denom = a + b
        hm = torch.zeros_like(denom)
        valid = denom > 0
        hm[valid] = 2 * a[valid] * b[valid] / denom[valid]
        return hm

    def init_conductive_neighbours(self, img, mask):
        diff_map = torch.zeros(img.shape, dtype=self.precision, device=img.device)
        for phase, D_p in self.Ds.items():
            diff_map[img == phase] = D_p

        diff_map = self._pad(diff_map)
        diff_map[:, 0] = diff_map[:, 1]
        diff_map[:, -1] = diff_map[:, -2]

        self.D_x = self._harmonic_mean(diff_map[:, :-1, 1:-1, 1:-1], diff_map[:, 1:, 1:-1, 1:-1])
        self.D_y = self._harmonic_mean(diff_map[:, 1:-1, :-1, 1:-1], diff_map[:, 1:-1, 1:, 1:-1])
        self.D_z = self._harmonic_mean(diff_map[:, 1:-1, 1:-1, :-1], diff_map[:, 1:-1, 1:-1, 1:])

        factor = self.D_x[:, :-1, :, :] + self.D_x[:, 1:, :, :] + \
                 self.D_y[:, :, :-1, :] + self.D_y[:, :, 1:, :] + \
                 self.D_z[:, :, :, :-1] + self.D_z[:, :, :, 1:]
        factor[:, 0]  += self.D_x[:, 0, :, :]
        factor[:, -1] += self.D_x[:, -1, :, :]
        factor[factor == 0] = torch.inf
        return factor

    def sum_weighted_neighbours(self, out, x0: int = 0) -> None:
        x1 = x0 + out.shape[1]
        torch.mul(self.field[:, x0 + 2:x1 + 2, 1:-1, 1:-1],
                  self.D_x[:, x0 + 1:x1 + 1, :, :], out=out)
        out.addcmul_(self.field[:, x0:x1, 1:-1, 1:-1], self.D_x[:, x0:x1, :, :])
        out.addcmul_(self.field[:, x0 + 1:x1 + 1, 2:, 1:-1], self.D_y[:, x0:x1, 1:, :])
        out.addcmul_(self.field[:, x0 + 1:x1 + 1, :-2, 1:-1], self.D_y[:, x0:x1, :-1, :])
        out.addcmul_(self.field[:, x0 + 1:x1 + 1, 1:-1, 2:], self.D_z[:, x0:x1, :, 1:])
        out.addcmul_(self.field[:, x0 + 1:x1 + 1, 1:-1, :-2], self.D_z[:, x0:x1, :, :-1])

    def vertical_flux(self):
        '''Calculates the vertical flux through the volume'''
        vert_flux = self.D_x[:, 1:-1, :, :] * \
                    (self.field[:, 2:-1, 1:-1, 1:-1] - \
                     self.field[:, 1:-2, 1:-1, 1:-1])
        return vert_flux


class PeriodicMultiPhaseSolver(MultiPhaseSolver):
    """Multi-phase solver with periodic boundary conditions in y and z."""

    def init_conductive_neighbours(self, img, mask):
        diff_map = torch.zeros(img.shape, dtype=self.precision, device=img.device)
        for phase, D_p in self.Ds.items():
            diff_map[img == phase] = D_p

        diff_map = self._pad(diff_map)
        # Dirichlet in x direction, periodic in y and z
        diff_map[:, 0]  = diff_map[:, 1]
        diff_map[:, -1] = diff_map[:, -2]
        diff_map[:, :, 0, :] = diff_map[:, :, -2, :]
        diff_map[:, :, -1, :] = diff_map[:, :, 1, :]
        diff_map[:, :, :, 0] = diff_map[:, :, :, -2]
        diff_map[:, :, :, -1] = diff_map[:, :, :, 1]

        self.D_x = self._harmonic_mean(diff_map[:, :-1, 1:-1, 1:-1], diff_map[:, 1:, 1:-1, 1:-1])
        self.D_y = self._harmonic_mean(diff_map[:, 1:-1, :-1, 1:-1], diff_map[:, 1:-1, 1:, 1:-1])
        self.D_z = self._harmonic_mean(diff_map[:, 1:-1, 1:-1, :-1], diff_map[:, 1:-1, 1:-1, 1:])

        factor = self.D_x[:, :-1, :, :] + self.D_x[:, 1:, :, :] + \
                 self.D_y[:, :, :-1, :] + self.D_y[:, :, 1:, :] + \
                 self.D_z[:, :, :, :-1] + self.D_z[:, :, :, 1:]
        factor[:, 0]  += self.D_x[:, 0, :, :]
        factor[:, -1] += self.D_x[:, -1, :, :]
        factor[factor == 0] = torch.inf
        return factor

    def apply_boundary_conditions(self):
        self.field[:, :, 0, :] = self.field[:, :, -2, :]
        self.field[:, :, -1, :] = self.field[:, :, 1, :]
        self.field[:, :, :, 0] = self.field[:, :, :, -2]
        self.field[:, :, :, -1] = self.field[:, :, :, 1]
