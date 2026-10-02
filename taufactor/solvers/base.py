"""Shared SOR solver base class."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from ..metrics import extract_connected_network


class SORSolver(ABC):
    """
    A minimal, clean template for SOR solvers.
    Subclasses override a few well-defined hooks.
    Args:
            img: labelled input image defining (non-)conducting phases.
            oemga: Over-relaxation factor for SOR scheme.
            device: The device to perform computations ('cpu' or 'cuda').
    """
    connectivity_periodic = (False, False, False)
    connectivity_open_end = True

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

        # Compute volume fraction of conductive phases and extract connected
        # transport network for each batch element
        self.vol_x, self.conn_vol_x, mask  = self._analyse_phases_and_connectivity(self.cpu_img)

        # Overrelaxation factor for SOR
        if omega is None:
            omega = 2 - torch.pi / (1.3 * self.Nx)
        self.omega = float(omega)

        # Labels as uint8 on device (4x smaller than float32) to cut init peak VRAM
        torch_img = torch.as_tensor(self.cpu_img, device=self.device)
        if torch_img.dtype != torch.uint8:
            torch_img = torch_img.to(torch.uint8)

        # Reactive neighbours before field so we can free torch_img early
        reac_nn = self.init_reactive_neighbours(torch_img, mask)
        self.factor = self.init_conductive_neighbours(torch_img, mask)
        del torch_img
        if reac_nn is not None:
            a_x = (torch.sum(reac_nn, (2, 3)) / (self.Ny * self.Nz * self.dx))
            conn_vf = torch.as_tensor(np.mean(self.conn_vol_x, axis=1),
                                      dtype=self.precision, device=self.device)
            k_0 = conn_vf / torch.mean(a_x * self.dx, 1) / self.Nx**2
            reac_nn.mul_(k_0[:, None, None, None])
            self.factor.add_(reac_nn)
            del reac_nn
            self.factor.masked_fill_(self.factor == 0, torch.inf)
            self.a_x = a_x.cpu().numpy()
            self.k_0 = k_0.cpu().numpy()

        self.field = self.init_field(mask)
        del mask

        self.cb, self._cb_inv = self._init_chequerboard()

        # Init params
        self.converged = False
        self.old_tau = 0
        self.iter = 0
        self.tau = None
        self.tau_t = []
        self.tau_x = None
        self.D_eff = None

    # ---------------- required hook ----------------
    @abstractmethod
    def init_field(self, mask: torch.Tensor) -> torch.Tensor:
        """Return initial padded field [bs,Nx+2,Ny+2,Nz+2]."""

    @abstractmethod 
    def init_conductive_neighbours(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """N_i: amount of conductive neighbours (cond_nn)"""

    @abstractmethod 
    def compute_metrics(self):
        """Defines tau and relative error"""

    # ---------------- optional hooks --------------
    def init_reactive_neighbours(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """S_i: amount of reactive neighbours (reac_nn)"""
        return None

    def apply_boundary_conditions(self):
        """Default: Dirichlet in x and no-flux in y and z direction."""

    def sum_weighted_neighbours(self, out: torch.Tensor) -> None:
        """Isotropic 6-neighbor sum into a preallocated interior buffer."""
        torch.add(self.field[:, 2:, 1:-1, 1:-1], self.field[:, :-2, 1:-1, 1:-1], out=out)
        out.add_(self.field[:, 1:-1, 2:, 1:-1])
        out.add_(self.field[:, 1:-1, :-2, 1:-1])
        out.add_(self.field[:, 1:-1, 1:-1, 2:])
        out.add_(self.field[:, 1:-1, 1:-1, :-2])

    def _apply_chequerboard(self, increment: torch.Tensor) -> None:
        """Zero the inactive colour and scale by omega, in-place."""
        mask_zero = self._cb_inv if (self.iter % 2 == 0) else self.cb
        increment.masked_fill_(mask_zero, 0)
        increment.mul_(self.omega)

    def plot_stats(self, relative_error):
        """Default: No plotting output."""

    def _display_figure(self, fig, display_attr):
        """Display ``fig`` once in notebooks, then update it in place."""
        if not plt.isinteractive():
            plt.ion()
        try:
            from IPython.display import display
        except ImportError:
            return
        if not hasattr(self, display_attr):
            setattr(self, display_attr, display(fig, display_id=True))
            plt.close(fig)
        else:
            getattr(self, display_attr).update(fig)

    @staticmethod
    def _set_plot_subtitle(ax, subtitle):
        """Add a small status line below an axes title."""
        ax.set_title(ax.get_title(), pad=24)
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center',
                va='bottom', fontsize='small', clip_on=False)

    def check_convergence(self, verbose, conv_crit, plot_interval):
        self.tau, relative_error = self.compute_metrics()

        if verbose == 'per_iter':
            # Print stats for slowest converging microstructure
            i = np.argmax(relative_error)
            print(f'Iter: {self.iter}, conv error: {abs(relative_error[i]):.3E}, tau: {self.tau[i]:.5f} (batch element {i})')

        if (verbose == 'plot') and (self.iter % (100*plot_interval) == 0):
            self.plot_stats(relative_error)

        self.tau_t.append(self.tau)
        if (verbose == 'debug') and (self.iter % (100*plot_interval) == 0):
            i = np.argmax(np.abs(relative_error))
            if not hasattr(self, '_debug_fig'):
                self._debug_fig, self._debug_ax = plt.subplots(figsize=(8,2), dpi=200)
            ax = self._debug_ax
            ax.clear()
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
            self._set_plot_subtitle(
                ax, f'Iter: {self.iter}, conv error: {np.abs(relative_error[i]):.3E}, '
                f'tau: {self.tau[i]:.5f} (batch element {i})'
            )
            ax.set_ylim(min_tau-0.1, max_tau+0.1)
            ax.legend()
            ax.grid()
            self._debug_fig.canvas.draw_idle()
            self._debug_fig.canvas.flush_events()
            self._display_figure(self._debug_fig, '_debug_display')

        if not np.all(relative_error < conv_crit):
            self.old_tau = self.tau
            return False

        if not np.allclose(self.tau, self.old_tau, rtol=2e-3, atol=1e-3):
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

        self.tau_t = []

        with torch.no_grad():
            increment = torch.empty(
                (self.batch_size, self.Nx, self.Ny, self.Nz),
                dtype=self.field.dtype,
                device=self.device,
            )
            start = timer()
            while not self.converged and self.iter < iter_limit:
                self.apply_boundary_conditions()
                self.sum_weighted_neighbours(increment)
                increment /= self.factor
                increment -= self.field[:, 1:-1, 1:-1, 1:-1]
                self._apply_chequerboard(increment)
                self.field[:, 1:-1, 1:-1, 1:-1] += increment
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

    def _analyse_phases_and_connectivity(self, img):
        """Return spatial volume fraction of conductive phase(s) and connected transport network."""
        vol_x = np.zeros(img.shape[:2])
        conn_vol_x = np.zeros(img.shape[:2])
        mask = np.zeros_like(img, dtype=bool)
        self.percolates = np.zeros(self.batch_size, dtype=bool)

        for b, batch_img in enumerate(img):
            conn = extract_connected_network(
                batch_img, self.conductive_labels, "x",
                periodic=self.connectivity_periodic,
                connectivity=1,
                open_end=self.connectivity_open_end,
            )
            if conn:
                mask[b] = conn[1]["connected_mask"]
                vol_x[b] = conn[1]["spatial_vol_frac_all"]
                conn_vol_x[b] = conn[1]["spatial_vol_frac_connected"]
                self.percolates[b] = mask[b].any()

        return vol_x, conn_vol_x, torch.as_tensor(
            mask, dtype=self.precision, device=self.device
        )

    def _init_chequerboard(self):
        """Bool chequerboard on device (True = even i+j+k), plus precomputed inverse.

        Built in z-chunks to avoid host meshgrid / large int64 temporaries.
        """
        cb = torch.empty((self.Nx, self.Ny, self.Nz), dtype=torch.bool, device=self.device)
        xy = (
            torch.arange(self.Nx, device=self.device)[:, None]
            + torch.arange(self.Ny, device=self.device)[None, :]
        ) & 1
        chunk = 64
        for z0 in range(0, self.Nz, chunk):
            z1 = min(self.Nz, z0 + chunk)
            z = torch.arange(z0, z1, device=self.device)
            cb[:, :, z0:z1] = ((xy.unsqueeze(-1) + z) & 1) == 0
        cb_inv = torch.empty_like(cb)
        torch.logical_not(cb, out=cb_inv)
        return cb, cb_inv

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
    def _neighbour_sum_from_padded(padded: torch.Tensor, out: torch.Tensor) -> None:
        """6-neighbour sum from a +1-padded volume into an interior-sized buffer."""
        torch.add(padded[:, 2:, 1:-1, 1:-1], padded[:, :-2, 1:-1, 1:-1], out=out)
        out.add_(padded[:, 1:-1, 2:, 1:-1])
        out.add_(padded[:, 1:-1, :-2, 1:-1])
        out.add_(padded[:, 1:-1, 1:-1, 2:])
        out.add_(padded[:, 1:-1, 1:-1, :-2])

    @staticmethod
    def _periodic_yz_neighbour_sum_from_padded(
        padded: torch.Tensor, out: torch.Tensor
    ) -> None:
        """6-neighbour sum with X ghosts and periodic Y/Z boundaries."""
        center = padded[:, 1:-1]
        torch.add(padded[:, :-2], padded[:, 2:], out=out)
        out[:, :, 1:].add_(center[:, :, :-1])
        out[:, :, :-1].add_(center[:, :, 1:])
        out[:, :, 0].add_(center[:, :, -1])
        out[:, :, -1].add_(center[:, :, 0])
        out[:, :, :, 1:].add_(center[:, :, :, :-1])
        out[:, :, :, :-1].add_(center[:, :, :, 1:])
        out[:, :, :, 0].add_(center[:, :, :, -1])
        out[:, :, :, -1].add_(center[:, :, :, 0])

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
