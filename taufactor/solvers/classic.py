"""Classic through-transport solvers."""

import warnings

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .base import SORSolver


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
        for i in range(2):
            vec = torch.unsqueeze(vec, -1)
        vec = torch.unsqueeze(vec, 0)
        return self._pad(mask * vec, [2*self.top_bc, 2*self.bot_bc])

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
        c_x = np.divide(c_x, self.conn_vol_x, out=np.zeros_like(self.conn_vol_x),
                        where=self.conn_vol_x != 0)
        self.c_x = c_x
        flux_from_c = c_x[:,1:] - c_x[:,:-1]
        flux_from_c[:,:][self.conn_vol_x[:,1:]==0] = 0
        flux_from_c[:,:][self.conn_vol_x[:,:-1]==0] = 0
        eps = 0.5*(self.vol_x[:,:-1] + self.vol_x[:,1:])
        self.tau_x = np.divide(eps * flux_from_c, self.flux_1d,
            out=np.full_like(flux_from_c, np.nan), where=self.flux_1d != 0)

        for b in range(self.batch_size):
            if not self.percolates[b]:
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
        i = np.argmax(relative_error)
        mean = np.expand_dims(np.mean(self.flux_1d, axis=1), 1)
        rel_fluxes = ((self.flux_1d - mean)/mean)
        if not hasattr(self, '_plot_fig'):
            self._plot_fig, self._plot_ax = plt.subplots(figsize=(8,2), dpi=200)
        ax = self._plot_ax
        ax.clear()
        x = np.arange(0, rel_fluxes.shape[1])+0.5
        for b in range(self.batch_size):
            if relative_error[b] > 0:
                ax.plot(x, rel_fluxes[b], label=f'batch_{b}', linestyle='-')

        ax.set_xlabel('voxels in x')
        ax.set_ylabel('relative fluxes')
        ax.set_title(f'Relative flux convergence in flux direction in iter {self.iter}')
        self._set_plot_subtitle(
            ax, f'Iter: {self.iter}, conv error: {abs(relative_error[i]):.3E}, '
            f'tau: {self.tau[i]:.5f} (batch element {i})'
        )
        ax.set_ylim(-0.1, 0.1)
        ax.legend()
        ax.grid()
        self._plot_fig.canvas.draw_idle()
        self._plot_fig.canvas.flush_events()
        self._display_figure(self._plot_fig, '_plot_display')


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

    def init_conductive_neighbours(self, img, mask):
        """Saves the number of conductive neighbours for flux calculation"""
        img2 = self._pad(mask, [2, 2])
        nn = torch.empty_like(mask)
        self._neighbour_sum_from_padded(img2, nn)
        del img2
        # avoid div 0 errors
        nn.masked_fill_(mask == 0, torch.inf)
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
        img2 = self._pad(mask, [2, 2])
        nn = torch.zeros_like(img2, dtype=self.precision)
        # iterate through shifts in the spatial dimensions
        factor = [1.0, self.Ky, self.Kz]
        for dim in range(1, 4):
            for dr in [1, -1]:
                nn += torch.roll(img2, dr, dim)*factor[dim-1]
        nn = self._crop(nn, 1)
        nn[mask == 0] = torch.inf
        nn[nn == 0] = torch.inf
        return nn

    def sum_weighted_neighbours(self, out):
        """Anisotropic 6-neighbor sum into a preallocated interior buffer."""
        torch.add(self.field[:, 2:, 1:-1, 1:-1], self.field[:, :-2, 1:-1, 1:-1], out=out)
        out.add_(self.field[:, 1:-1, 2:, 1:-1], alpha=self.Ky)
        out.add_(self.field[:, 1:-1, :-2, 1:-1], alpha=self.Ky)
        out.add_(self.field[:, 1:-1, 1:-1, 2:], alpha=self.Kz)
        out.add_(self.field[:, 1:-1, 1:-1, :-2], alpha=self.Kz)


class PeriodicSolver(Solver):
    """Two-phase SOR solver with periodic Y/Z boundaries.

    Uses periodic wrapping for neighbor evaluation in Y and Z and
    reapplies periodic boundary conditions to the field each iteration.
    X remains the flux/open direction.

    Notes:
        Overrides ``init_nn`` and ``apply_boundary_conditions`` from
        :class:`Solver`.
    """
    connectivity_periodic = (False, True, True)

    def init_conductive_neighbours(self, img, mask):
        padded = self._pad(mask, [2, 2])[:, :, 1:-1, 1:-1]
        nn = torch.empty_like(mask)
        self._periodic_yz_neighbour_sum_from_padded(padded, nn)
        del padded
        nn[mask == 0] = torch.inf
        nn[nn == 0] = torch.inf
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
        diff_map.mul_(mask)

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

    def sum_weighted_neighbours(self, out) -> None:
        torch.mul(self.field[:, 2:, 1:-1, 1:-1], self.D_x[:, 1:, :, :], out=out)
        out.addcmul_(self.field[:, :-2, 1:-1, 1:-1], self.D_x[:, :-1, :, :])
        out.addcmul_(self.field[:, 1:-1, 2:, 1:-1], self.D_y[:, :, 1:, :])
        out.addcmul_(self.field[:, 1:-1, :-2, 1:-1], self.D_y[:, :, :-1, :])
        out.addcmul_(self.field[:, 1:-1, 1:-1, 2:], self.D_z[:, :, :, 1:])
        out.addcmul_(self.field[:, 1:-1, 1:-1, :-2], self.D_z[:, :, :, :-1])

    def vertical_flux(self):
        '''Calculates the vertical flux through the volume'''
        vert_flux = self.D_x[:, 1:-1, :, :] * \
                    (self.field[:, 2:-1, 1:-1, 1:-1] - \
                     self.field[:, 1:-2, 1:-1, 1:-1])
        return vert_flux


class PeriodicMultiPhaseSolver(MultiPhaseSolver):
    """Multi-phase solver with periodic boundary conditions in y and z."""
    connectivity_periodic = (False, True, True)

    def init_conductive_neighbours(self, img, mask):
        diff_map = torch.zeros(img.shape, dtype=self.precision, device=img.device)
        for phase, D_p in self.Ds.items():
            diff_map[img == phase] = D_p
        diff_map.mul_(mask)

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
