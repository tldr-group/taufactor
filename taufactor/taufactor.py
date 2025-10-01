"""Main module."""
import numpy as np
from IPython.display import clear_output
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import psutil
try:
    import torch
except Exception:
    torch = None
import warnings


class BaseSolver:
    """Base solver class to handle common functionality for solvers."""

    def __init__(self, img, bc=(-0.5, 0.5), device='cuda'):
        if torch is None:
            raise ImportError(
                "PyTorch is required to use TauFactor solvers. Install pytorch following "
                "https://taufactor.readthedocs.io/en/latest/installation.html"
            )

        if not isinstance(img, np.ndarray):
            raise TypeError("Error: input image must be a NumPy array!")
        if img.ndim == 2:
            # Convert 2D to pseudo-3D by expanding
            img = np.expand_dims(img, axis=-1)
        self.top_bc, self.bot_bc = bc
        # Add batch channel if not existent
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        self.cpu_img = img
        self.precision = torch.float

        self.device = torch.device(device)
        # check device is available
        if torch.device(device).type.startswith('cuda') and not torch.cuda.is_available():
            self.device = torch.device('cpu')
            warnings.warn(
                "CUDA not available, defaulting device to cpu. "
                "To avoid this warning, explicitly set device='cpu'"
            )
        # save original image in cuda
        img = torch.tensor(img, dtype=self.precision, device=self.device)

        # init conc
        self.conc = self.init_conc(img)
        # create nn map
        self.nn = self.init_nn(img)
        # overrelaxation factor
        self.w = 2 - torch.pi / (1.5 * img.shape[1])
        # checkerboarding to ensure stable steps
        self.cb = self.init_cb(img)
        self.batch_size, x, y, z = self.cpu_img.shape
        self.L_A = x / (z * y)

        # solving params
        self.converged = False
        self.old_fl = -1
        self.iter = 1

        # Results
        self.tau = None
        self.D_eff = None
        img = None

    def init_cb(self, img):
        """Create a checkerboard over-relaxation mask for SOR.

        Args:
            img (torch.Tensor): Batched 3D tensor of the working image.

        Returns:
            list[torch.Tensor]: Two masks (even/odd slices), each shaped
            like the spatial volume.
        """
        _, x, y, z = img.shape
        cb = np.zeros([x, y, z])
        a, b, c = np.meshgrid(range(x), range(y), range(z), indexing='ij')
        cb[(a + b + c) % 2 == 0] = 1
        cb *= self.w
        return [torch.roll(torch.tensor(cb, dtype=self.precision, device=self.device), sh, 0) for sh in [0, 1]]

    def solve(self):
        """Run solver"""
        raise NotImplementedError("You're trying to call the solve function on the generalized BaseSolver class")

    def pad(self, img, vals=[0] * 6):
        """Pad a 3D volume with constant boundary values.

        Pads one voxel on each face and sets face values from ``vals``.

        Args:
            img (torch.Tensor): Batched 3D tensor ``[B, X, Y, Z]``.
            vals (list[float], optional): Six boundary values in order
                ``[x0, x1, y0, y1, z0, z1]``. Defaults to six zeros.

        Returns:
            torch.Tensor: Padded tensor with shape ``[B, X+2, Y+2, Z+2]``.
        """
        while len(vals) < 6:
            vals.append(0)
        to_pad = [1]*8
        to_pad[-2:] = (0, 0)
        img = torch.nn.functional.pad(img, to_pad, 'constant')
        img[:, 0], img[:, -1] = vals[:2]
        img[:, :, 0], img[:, :, -1] = vals[2:4]
        img[:, :, :, 0], img[:, :, :, -1] = vals[4:]
        return img

    def crop(self, img, c=1):
        """Crop a uniform margin from all faces.

        Args:
            img (torch.Tensor): Batched 3D tensor.
            c (int, optional): Number of voxels to remove from each face.
                Defaults to ``1``.

        Returns:
            torch.Tensor: Cropped tensor.
        """
        return img[:, c:-c, c:-c, c:-c]

    def check_vertical_flux(self, conv_crit):
        """Assess vertical flux uniformity as a convergence proxy.

        Computes per-slice vertical flux, flags convergence per batch, and
        returns diagnostics.

        Args:
            conv_crit (float): Tolerance for relative flux variation across
                slices.

        Returns:
            tuple:
                flags (list[str]): One of ``{"converged","not_converged","zero_flux"}`` per batch.
                mean_fl (torch.Tensor): Mean flux per batch ``[B]``.
                err (torch.Tensor): Relative spread per batch ``[B]``.
                fl (torch.Tensor): Slice-resolved flux ``[B, X]``.
        """
        from .metrics import extract_through_feature

        vert_flux = self.calc_vertical_flux()
        # Sum over the y and z dimensions only, leaving a (bs, x) result.
        fl = torch.sum(vert_flux, (2, 3)) # (bs, x)
        fl_max, _ = torch.max(fl, dim=1)  # shape: (bs,)
        fl_min, _ = torch.min(fl, dim=1)  # shape: (bs,)
        mean_fl = torch.mean(fl, dim=1)   # shape: (bs,)
        
        # Compute the error for each batch element:
        err = (fl_max - fl_min) / fl_max

        flags = []
        for b in range(self.batch_size):
            if (fl_min[b] == 0) or (mean_fl[b] == 0):
                _ , frac = extract_through_feature(self.cpu_img[b], 1, 'x')
                if frac == 0:
                    print(f"Warning: batch element {b} has no percolating path!")
                    flags.append("zero_flux")
                else:
                    flags.append("not_converged")
            else:
                if err[b].item() < conv_crit or torch.isnan(err[b]).item():
                    flags.append("converged")
                else:
                    flags.append("not_converged")
        return flags, mean_fl, err, fl

    def _check_rolling_mean(self, conv_crit):
        err = (self.new_fl - self.old_fl) / (self.new_fl + self.old_fl)
        return torch.max(err) < conv_crit

    def _end_simulation(self, iter_limit, verbose, start):
        converged = 'converged to'
        if self.iter >= iter_limit:
            print('Warning: not converged')
            converged = 'unconverged value of tau'

        if verbose:
            print(f'{converged}: {self.tau} after: {self.iter-1} iterations in: {np.around(timer() - start, 4)}s ({np.around((timer() - start)/(self.iter-1), 4)} s/iter)')
            if self.device.type == 'cuda':
                print(f"GPU-RAM currently allocated {torch.cuda.memory_allocated(device=self.device) / 1e6:.2f} MB ({torch.cuda.memory_reserved(device=self.device) / 1e6:.2f} MB reserved)")
                print(f"GPU-RAM maximally allocated {torch.cuda.max_memory_allocated(device=self.device) / 1e6:.2f} MB ({torch.cuda.max_memory_reserved(device=self.device) / 1e6:.2f} MB reserved)")
            elif self.device.type == 'cpu':
                memory_info = psutil.virtual_memory()
                print(f"CPU total memory: {memory_info.total / 1e6:.2f} MB")
                print(f"CPU available memory: {memory_info.available / 1e6:.2f} MB")
                print(f"CPU used memory: {memory_info.used / 1e6:.2f} MB")


class Solver(BaseSolver):
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

    def __init__(self, img, bc=(-0.5, 0.5), D_0=1, device='cuda'):
        super().__init__(img, bc, device)
        self.D_0 = D_0
        self.D_mean = None
        self.VF = np.mean(self.cpu_img, axis=(1,2,3))

        if len(np.unique(img).shape) > 2 or np.unique(img).max() not in [0, 1] or np.unique(img).min() not in [0, 1]:
            raise ValueError(
                f'Input image must only contain 0s and 1s. Your image must be segmented to use this tool. If your image has been segmented, ensure your labels are 0 for non-conductive and 1 for conductive phase. Your image has the following labels: {np.unique(img)}. If you have more than one conductive phase, use the multi-phase solver.')

    def init_conc(self, img):
        """Sets an initial linear field across the volume"""
        bs, x, y, z = img.shape
        sh = 1 / (x * 2)
        vec = torch.linspace(self.top_bc + sh, self.bot_bc -
                             sh, x, dtype=self.precision, device=self.device)
        for i in range(2):
            vec = torch.unsqueeze(vec, -1)
        vec = torch.unsqueeze(vec, 0)
        vec = vec.repeat(bs, 1, y, z, )
        return self.pad(img * vec, [self.top_bc * 2, self.bot_bc * 2]).to(self.device)

    def init_nn(self, img):
        """Saves the number of conductive neighbours for flux calculation"""
        img2 = self.pad(self.pad(img, [2, 2]))
        nn = torch.zeros_like(img2, dtype=self.precision)
        # iterate through shifts in the spatial dimensions
        for dim in range(1, 4):
            for dr in [1, -1]:
                nn += torch.roll(img2, dr, dim)
        # remove the two paddings
        nn = self.crop(nn, 2)
        # avoid div 0 errors
        nn[img == 0] = torch.inf
        nn[nn == 0] = torch.inf
        return nn.to(self.device)

    def apply_boundary_conditions(self):
        pass

    def solve(self, iter_limit=5000, verbose=True, conv_crit=2*10**-2, plot_interval=10):
        """
        run a solve simulation

        :param iter_limit: max iterations before aborting, will attemtorch double for the same no. iterations
        if initialised as singles
        :param verbose: Whether to print tau. Can be set to 'per_iter' for more feedback
        :param conv_crit: convergence criteria, minimum percent difference between
        max and min flux through a given layer
        :return: tau
        """
        if (verbose) and (self.device.type == 'cuda'):
            torch.cuda.reset_peak_memory_stats(device=self.device)

        with torch.no_grad():
            start = timer()
            while not self.converged and self.iter <= iter_limit:
                self.apply_boundary_conditions()
                # find sum of all nearest neighbours
                out = self.conc[:, 2:, 1:-1, 1:-1] + \
                    self.conc[:, :-2, 1:-1, 1:-1] + \
                    self.conc[:, 1:-1, 2:, 1:-1] + \
                    self.conc[:, 1:-1, :-2, 1:-1] + \
                    self.conc[:, 1:-1, 1:-1, 2:] + \
                    self.conc[:, 1:-1, 1:-1, :-2]
                # divide by n conductive nearest neighbours to give flux
                out /= self.nn
                # check convergence using criteria
                if self.iter % 100 == 0:
                    self.converged = self.check_convergence(verbose, conv_crit, plot_interval)
                # efficient way of adding flux to old conc with overrelaxation
                out -= self.crop(self.conc, 1)
                out *= self.cb[self.iter % 2]
                self.conc[:, 1:-1, 1:-1, 1:-1] += out
                self.iter += 1
            self.D_mean = self.D_0
            self.D_eff = self.D_mean*self.D_rel
            self._end_simulation(iter_limit, verbose, start)
            return self.tau

    def check_convergence(self, verbose, conv_crit, plot_interval):
        flags, self.new_fl, err, slice_fluxes = self.check_vertical_flux(conv_crit)
        self.D_rel = np.zeros(self.batch_size)
        self.tau = np.zeros(self.batch_size)
        for b in range(self.batch_size):
            if flags[b] == "zero_flux":
                self.D_rel[b] = 0
                self.tau[b] = np.inf
                flags[b] = "converged"
            else:
                self.D_rel[b] = (self.new_fl[b].cpu().numpy()) * self.L_A \
                                 / abs(self.top_bc - self.bot_bc)
                self.tau[b] = self.VF[b] / self.D_rel[b]

        if verbose == 'per_iter':
            if self.batch_size > 1:
                print('Warning: Verbose per_iter will only output the first batch element.')
            print(f'Iter: {self.iter}, conv error: {abs((err[0]).item())}, tau: {self.tau[0].item()}')
            
        if (verbose == 'plot') and (self.iter % (100*plot_interval) == 0):
            clear_output(wait=True)
            print(f'Iter: {self.iter}, conv error: {abs(err[0].item())}, tau: {self.tau[0].item()} (batch element 0)')
            rel_fluxes = ((slice_fluxes - self.new_fl.unsqueeze(1))/self.new_fl.unsqueeze(1)).cpu().numpy()
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

        overall_converged = all(flag == "converged" for flag in flags)
        if overall_converged:
            self.converged = self._check_rolling_mean(conv_crit=1e-3)

            if not self.converged:
                self.old_fl = self.new_fl
                return False
            else:
                return True
        else:
            self.old_fl = self.new_fl
            return False

    def calc_vertical_flux(self):
        '''Calculates the vertical flux through the volume'''
        # Indexing removes boundary layers (1 layer at every boundary)
        vert_flux = self.conc[:, 2:-1, 1:-1, 1:-1] - \
            self.conc[:, 1:-2, 1:-1, 1:-1]
        vert_flux[self.conc[:, 1:-2, 1:-1, 1:-1] == 0] = 0
        vert_flux[self.conc[:, 2:-1, 1:-1, 1:-1] == 0] = 0
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

    def __init__(self, img, spacing, bc=(-0.5, 0.5), D_0=1, device='cuda'):
        if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
            raise ValueError("spacing must be a list or tuple with three elements (dx, dy, dz)")
        if not all(isinstance(x, (int, float)) for x in spacing):
            raise ValueError("All elements in spacing must be integers or floats")
        if (np.max(spacing)/np.min(spacing) > 10):
            warnings.warn("This computation is very questionable for largely different spacings e.g. dz >> dx.")
        dx, dy, dz = spacing
        self.Ky = (dx/dy)**2
        self.Kz = (dx/dz)**2
        super().__init__(img, bc, D_0, device)

    def init_nn(self, img):
        """Saves the number of conductive neighbours for flux calculation"""
        img2 = self.pad(self.pad(img, [2, 2]))
        nn = torch.zeros_like(img2, dtype=self.precision)
        # iterate through shifts in the spatial dimensions
        factor = [1.0, self.Ky, self.Kz]
        for dim in range(1, 4):
            for dr in [1, -1]:
                nn += torch.roll(img2, dr, dim)*factor[dim-1]
        # remove the two paddings
        nn = self.crop(nn, 2)
        # avoid div 0 errors
        nn[img == 0] = torch.inf
        nn[nn == 0] = torch.inf
        return nn.to(self.device)

    def solve(self, iter_limit=5000, verbose=True, conv_crit=2*10**-2, plot_interval=10):
        """
        run a solve simulation

        :param iter_limit: max iterations before aborting, will attemtorch double for the same no. iterations
        if initialised as singles
        :param verbose: Whether to print tau. Can be set to 'per_iter' for more feedback
        :param conv_crit: convergence criteria, minimum percent difference between
        max and min flux through a given layer
        :return: tau
        """
        if (verbose) and (self.device.type == 'cuda'):
            torch.cuda.reset_peak_memory_stats(device=self.device)

        with torch.no_grad():
            start = timer()
            while not self.converged and self.iter <= iter_limit:
                # find sum of all nearest neighbours
                out = self.conc[:, 2:, 1:-1, 1:-1] + self.conc[:, :-2, 1:-1, 1:-1] + \
                      self.Ky*(self.conc[:, 1:-1, 2:, 1:-1] + self.conc[:, 1:-1, :-2, 1:-1]) + \
                      self.Kz*(self.conc[:, 1:-1, 1:-1, 2:] + self.conc[:, 1:-1, 1:-1, :-2])
                # divide by n conductive nearest neighbours to give flux
                out /= self.nn
                # check convergence using criteria
                if self.iter % 100 == 0:
                    self.converged = self.check_convergence(verbose, conv_crit, plot_interval)
                # efficient way of adding flux to old conc with overrelaxation
                out -= self.crop(self.conc, 1)
                out *= self.cb[self.iter % 2]
                self.conc[:, 1:-1, 1:-1, 1:-1] += out
                self.iter += 1
            self.D_mean = self.D_0
            self.D_eff = self.D_mean*self.D_rel
            self._end_simulation(iter_limit, verbose, start)
            return self.tau


class PeriodicSolver(Solver):
    """Two-phase SOR solver with periodic Y/Z boundaries.

    Uses periodic wrapping for neighbor evaluation in Y and Z and
    reapplies periodic boundary conditions to the field each iteration.
    X remains the flux/open direction.

    Notes:
        Overrides ``init_nn`` and ``apply_boundary_conditions`` from
        :class:`Solver`.
    """

    def init_nn(self, img):
        img2 = self.pad(img, [2, 2])[:, :, 1:-1, 1:-1]
        nn = torch.zeros_like(img2)
        # iterate through shifts in the spatial dimensions
        for dim in range(1, 4):
            for dr in [1, -1]:
                nn += torch.roll(img2, dr, dim)
        # avoid div 0 errors
        nn = nn[:, 1:-1]
        nn[img == 0] = torch.inf
        nn[nn == 0] = torch.inf
        return nn.to(self.device)

    def apply_boundary_conditions(self):
        self.conc[:,:,0,:] = self.conc[:,:,-2,:]
        self.conc[:,:,-1,:] = self.conc[:,:,1,:]
        self.conc[:,:,:,0] = self.conc[:,:,:,-2]
        self.conc[:,:,:,-1] = self.conc[:,:,:,1]


class MultiPhaseSolver(BaseSolver):
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

    def __init__(self, img, cond={1: 1}, bc=(-0.5, 0.5), device='cuda'):
        if (0 in cond.values()):
            raise ValueError(
                '0 conductivity phase: non-conductive phase should be labelled 0 in the input image and ommitted from the cond argument')
        if (0 in cond.keys()):
            raise ValueError(
                '0 cannot be used as a conductive phase label, please use a positive integer and leave 0 for non-conductive phase')

        self.cond = {ph: 0.5 / c for ph, c in cond.items()}

        # Results
        super().__init__(img, bc, device)
        if self.batch_size > 1:
            raise TypeError('Error: The MultiPhaseSolver is only implemented for batch_size=1!')
        self.pre_factors = self.nn[1:]
        self.nn = self.nn[0]

        self.VF = {p: np.mean(img == p)
                   for p in np.unique(img)}

        if len(np.array([self.VF[z] for z in self.VF.keys() if z != 0])) > 0:
            self.D_mean = np.sum(
                np.array([self.VF[z]*(1/(2*self.cond[z])) for z in self.VF.keys() if z != 0]))
        else:
            self.D_mean = 0

    def init_conc(self, img):
        bs, x, y, z = img.shape
        sh = 1 / (x + 1)
        vec = torch.linspace(self.top_bc + sh, self.bot_bc - sh, x)
        for i in range(2):
            vec = torch.unsqueeze(vec, -1)
        vec = torch.unsqueeze(vec, 0)
        vec = vec.repeat(bs, 1, y, z)
        vec = vec.to(self.device)
        # vec = vec.astype(self.precision)
        img1 = img.clone().to(self.device)
        img1[img1 > 1] = 1
        return self.pad(img1 * vec, [self.top_bc, self.bot_bc])

    def init_nn(self, img):
        # conductivity map
        img2 = torch.zeros_like(img)
        for ph in self.cond:
            c = self.cond[ph]
            img2[img == ph] = c
        img2 = self.pad(self.pad(img2))
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
                nn_list.append(self.crop(sum, 1).to(self.device))
        # remove the two paddings
        nn = self.crop(nn, 2)
        # avoid div 0 errors
        nn[img == 0] = torch.inf
        nn[nn == 0] = torch.inf
        nn_list.insert(0, nn.to(self.device))
        return nn_list

    def solve(self, iter_limit=5000, verbose=True, conv_crit=2*10**-2):
        """
        run a solve simulation

        :param iter_limit: max iterations before aborting, will attemtorch double for the same no. iterations
        if initialised as singles
        :param verbose: Whether to print tau. Can be set to 'per_iter' for more feedback
        :param conv_crit: convergence criteria, minimum percent difference between
        max and min flux through a given layer
        :return: tau
        """
        if (verbose) and (self.device.type == 'cuda'):
            torch.cuda.reset_peak_memory_stats(device=self.device)

        start = timer()
        while not self.converged and self.iter <= iter_limit:
            out = self.conc[:, 2:, 1:-1, 1:-1] * self.pre_factors[0][:, 2:, 1:-1, 1:-1] + \
                self.conc[:, :-2, 1:-1, 1:-1] * self.pre_factors[1][:, :-2, 1:-1, 1:-1] + \
                self.conc[:, 1:-1, 2:, 1:-1] * self.pre_factors[2][:, 1:-1, 2:, 1:-1] + \
                self.conc[:, 1:-1, :-2, 1:-1] * self.pre_factors[3][:, 1:-1, :-2, 1:-1] + \
                self.conc[:, 1:-1, 1:-1, 2:] * self.pre_factors[4][:, 1:-1, 1:-1, 2:] + \
                self.conc[:, 1:-1, 1:-1, :-2] * \
                self.pre_factors[5][:, 1:-1, 1:-1, :-2]
            out /= self.nn
            if self.iter % 100 == 0:
                self.converged = self.check_convergence(verbose, conv_crit)
            out -= self.crop(self.conc, 1)
            out *= self.cb[self.iter % 2]
            self.conc[:, 1:-1, 1:-1, 1:-1] += out
            self.iter += 1

        self._end_simulation(iter_limit, verbose, start)
        return self.tau

    def check_convergence(self, verbose, conv_crit):
        # print progress
        semi_converged, self.new_fl, err, _ = self.check_vertical_flux(conv_crit)
        _, x, y, z = self.cpu_img.shape
        self.D_eff = (self.new_fl[0]*(x+1)/(y*z)).cpu()
        self.tau = self.D_mean / \
            self.D_eff if self.D_eff != 0 else torch.tensor(torch.inf)
        if semi_converged[0] == 'zero_flux':
            return True

        if verbose == 'per_iter':
            print(
                f'Iter: {self.iter}, conv error: {abs(err[0].item())}, tau: {self.tau.item()}')

        if semi_converged[0]:
            self.converged = self._check_rolling_mean(conv_crit=1e-3)

            if not self.converged:
                self.old_fl = self.new_fl[0]
                return False
            else:
                return True
        else:
            self.old_fl = self.new_fl[0]
            return False

        # increase precision to double if currently single
        # if self.iter >= iter_limit:
        #     # if self.precision == cp.single:
        #     #     print('increasing precision to double')
        #     #     self.iter = 0
        #     #     self.conc = cp.array(self.conc, dtype=cp.double)
        #     #     self.nn = cp.array(self.nn, dtype=cp.double)
        #     #     self.precision = cp.double
        #     else:
        #         return True

        return False

    def calc_vertical_flux(self):
        '''Calculates the vertical flux through the volume'''
        vert_flux = (self.conc[:, 2:-1, 1:-1, 1:-1] - self.conc[:,
                     1:-2, 1:-1, 1:-1]) * self.pre_factors[1][:, 1:-2, 1:-1, 1:-1]
        vert_flux[self.nn[:,1:] == torch.inf] = 0
        return vert_flux


class ElectrodeSolver():
    """AC electrode tortuosity solver (migration + capacitive current).

    Solves a complex-valued potential field under sinusoidal excitation
    with a closed (zero-flux) right boundary, using an SOR-like update
    and frequency-dependent prefactors. Reports the electrode
    tortuosity from boundary current.

    Args:
        img (numpy.ndarray): 2D or 3D binary image; internally batched.
        omega (float, optional): Angular frequency of excitation.
            Defaults to ``1e-6``.
        device (str | torch.device, optional): Compute device.
            Defaults to ``'cuda'``.

    Attributes:
        omega (float): Angular excitation frequency.
        res (float): Series resistance coefficient.
        c_DL (float): Double-layer capacitance coefficient.
        A_CC (int): Current-collector interfacial area.
        k_0 (float): Scaling constant for normalization.
        VF (float): Volume fraction of the conductive phase.
        img (torch.Tensor): Working image on device.
        phi (torch.Tensor): Complex potential field (padded).
        phase_map (torch.Tensor): Padded binary phase mask.
        prefactor (torch.Tensor): Complex update prefactors.
        w (float): Over-relaxation factor.
        cb (list[torch.Tensor]): Checkerboard masks.
        converged (bool): Global convergence flag.
        semiconverged (float | bool): Stage convergence tracker.
        iter (int): Iteration counter.
        tau_e (float | torch.Tensor): Electrode tortuosity estimate.
        D_eff (float | None): Placeholder; not central to AC solve.
        D_mean (float | None): Placeholder; not central to AC solve.

    Notes:
        This class is standalone (does not inherit from :class:`BaseSolver`)
        due to its complex-valued field and AC-specific update scheme.
    """

    def __init__(self, img, omega=1e-6, device='cuda'):
        img = np.expand_dims(img, 0)
        self.cpu_img = img
        self.precision = torch.double
        # check device is available
        self.device = torch.device(device)
        if torch.device(device).type.startswith('cuda') and not torch.cuda.is_available():
            self.device = torch.device('cpu')
            warnings.warn(
                "CUDA not available, defaulting device to cpu. To avoid this warning, explicitly set the device when initialising the solver with device=torch.device('cpu')")
        # Define omega, res and c_DL
        self.omega = omega
        self.res = 1
        self.c_DL = 1
        if len(img.shape) == 4:
            self.A_CC = img.shape[2]*img.shape[3]
        else:
            self.A_CC = img.shape[2]
        self.k_0 = 1

        # VF calc
        self.VF = np.mean(img)

        # save original image in cuda
        img = torch.tensor(img, dtype=self.precision).to(self.device)
        self.img = img
        # init phi

        self.phi = self.init_phi(img)

        self.phase_map = self.pad(img, [1, 0])

        # create prefactor map
        self.prefactor = self.init_prefactor(img)

        # checkerboarding
        self.w = 2 - torch.pi / (1.5 * img.shape[1])
        # self.w = 1.8
        # self.w = 0.01
        self.cb = self.init_cb(img)

        # solving params
        self.converged = False
        self.semiconverged = False
        self.old_fl = -1
        self.iter = 1

        # Results
        self.tau_e = 0
        self.D_eff = None
        self.D_mean = None

    def pad(self, img, vals=[0] * 6):
        while len(vals) < 6:
            vals.append(0)
        if len(img.shape) == 4:
            to_pad = [1]*8
            to_pad[-2:] = (0, 0)
        elif len(img.shape) == 3:
            to_pad = [1]*6
            to_pad[-2:] = (0, 0)

        img = torch.nn.functional.pad(img, to_pad, 'constant')
        img[:, 0], img[:, -1] = vals[:2]
        img[:, :, 0], img[:, :, -1] = vals[2:4]

        if len(img.shape) == 4:
            img[:, :, :, 0], img[:, :, :, -1] = vals[4:]
        return img

    def crop(self, img, c=1):
        if len(img.shape) == 4:
            return img[:, c:-c, c:-c, c:-c]
        elif len(img.shape) == 3:
            return img[:, c:-c, c:-c]

    def init_phi(self, img):
        """
        Initialise phi field as zeros

        :param img: input image, with 1s conductive and 0s non-conductive
        :type img: torch.array
        :return: phi
        :rtype: torch.array
        """
        phi = torch.zeros_like(img, dtype=self.precision,
                               device=self.device)+0j
        phi = self.pad(phi, [1, 0])
        return phi.to(self.device)

    def init_cb(self, img):

        if len(img.shape) == 4:
            bs, x, y, z = img.shape
            cb = np.zeros([x, y, z])
            a, b, c = np.meshgrid(range(x), range(y), range(z), indexing='ij')
            cb[(a + b + c) % 2 == 0] = 1*self.w
            return [torch.roll(torch.tensor(cb), sh, 0).to(self.device) for sh in [0, 1]]

        elif len(img.shape) == 3:
            bs, x, y = img.shape
            cb = np.zeros([x, y])
            a, b = np.meshgrid(range(x), range(y), indexing='ij')
            cb[(a + b) % 2 == 0] = 1*self.w
            cb = [torch.roll(torch.tensor(cb).to(self.device), sh, 0)
                  for sh in [0, 1]]
            cb[1][0] = cb[1][2]
            return cb

    def init_prefactor(self, img):
        """
        Initialise prefactors -> (nn_cond+2j*omega*res*c(dims-nn_cond))**-1

        :param img: input image, with 1s conductive and 0s non-conductive
        :type img: cp.array
        :return: prefactor
        :rtype: cp.array
        """
        dims = (len(img.shape)-1)*2
        # find number of conducting nearest neighbours
        img2 = self.pad(img, [1, 0])
        nn_cond = torch.zeros_like(img2, dtype=self.precision)
        # iterate through shifts in the spatial dimensions
        for dim in range(1, len(img.shape)):
            for dr in [1, -1]:
                nn_cond += torch.roll(img2, dr, dim)
        # remove the paddings
        nn_cond = self.crop(nn_cond, 1)
        self.nn = nn_cond
        orc = self.omega*self.res*self.c_DL
        nn_solid = dims - nn_cond
        omegapf = (orc**2 + 1j*orc)/(orc**2+1)
        prefactor = (nn_cond + 2*nn_solid*omegapf)**-1
        # prefactor = (nn_cond+2j*self.omega*self.res*self.c_DL*(dims-nn_cond))**-1
        prefactor[prefactor == torch.inf] = 0
        prefactor[img == 0] = 0
        return prefactor.to(self.device)

    def sum_neighbours(self):
        i = 0
        for dim in range(1, len(self.phi.shape)):
            for dr in [1, -1]:
                if i == 0:
                    out = torch.roll(self.phi, dr, dim)
                else:
                    out += torch.roll(self.phi, dr, dim)
                i += 1

        out = self.crop(out, 1)
        return out

    def check_convergence(self):

        if len(self.tau_es) < 1000:
            return False
        loss = np.std(np.array(self.tau_es[-100:]))
        # print(len(self.tau_es),self.tau_es[-1], loss)
        if self.verbose == 'per_iter':
            print(f'(iter {self.iter} loss {loss}, taue {self.tau_es[-1]}')
        if loss < self.conv_crit:
            if self.semiconverged:
                if self.tau_es[-1] > 1e-5:
                    if abs(self.semiconverged - self.tau_es[-1]) < self.conv_crit_2:
                        self.tau_e = self.tau_es[-1]
                        self._end_simulation()
                        return True
                else:
                    self.phi = self.init_phi(self.img)
            self.semiconverged = self.tau_es[-1]
            self.omega *= 0.1
            print(
                f'Semi-converged to {self.semiconverged}. Reducing omega to {self.omega} to check convergence')

            self.iter = 0
            self.prefactor = self.init_prefactor(self.img)
            self.solve(iter_limit=self.iter_limit,
                       verbose=self.verbose, conv_crit=self.conv_crit)
            return True
        if self.iter_limit == self.iter:
            print(
                'Iteration limit reached. Increase the iteration limit or try starting from a smaller omega')

            return True
        return False

    def tau_e_from_phi(self):
        #  calculate total current on bottom boundary
        n = self.phase_map[0, 1].sum()
        z = self.res / (n-self.phi[0, 1].sum())
        self.z = z
        r_ion = z.real*3
        tau_e = self.VF * r_ion * self.k_0 * self.A_CC / self.phi.shape[1]

        return tau_e.cpu()

    def solve(self, iter_limit=100000, verbose=True, conv_crit=1e-5, conv_crit_2=1e-3):
        """
        run a solve simulation

        :param iter_limit: max iterations before aborting, will attemtorch double for the same no. iterations
        if initialised as singles
        :param verbose: Whether to print tau. Can be set to 'per_iter' for more feedback
        :param conv_crit: convergence criteria - running standard deviation of tau_e
        :param conv_crit_2: convergence criteria - maximum difference between tau_e in consecutive omega solves
        :return: tau
        """
        self.conv_crit = conv_crit
        self.conv_crit_2 = conv_crit_2

        self.iter_limit = iter_limit
        self.verbose = verbose
        dim = len(self.phi.shape)
        self.start = timer()
        self.frames = []
        self.loss = []
        self.tau_es = []

        while not self.converged and self.iter <= iter_limit:
            out = self.sum_neighbours()
            out *= self.prefactor*self.crop(self.phase_map)
            out[self.prefactor == -1] = 0
            self.tau_es.append(self.tau_e_from_phi())
            if self.iter % 100 == 0:
                self.converged = self.check_convergence()
            out -= self.crop(self.phi, 1)
            out *= self.cb[self.iter % 2]

            if dim == 4:
                self.phi[:, 1:-1, 1:-1, 1:-1] += out
            elif dim == 3:
                self.phi[:, 1:-1, 1:-1] += out

            self.iter += 1
        # self.tau_e = self.tau_es[-1]
        # self._end_simulation(iter_limit, verbose, start)

    def _end_simulation(self, ):
        if self.iter == self.iter_limit - 1:
            print('Warning: not converged')
            converged = 'unconverged value of tau'
        converged = 'converged to'
        if self.verbose:
            print(f'{converged}: {self.tau_e} after: {self.iter} iterations in: {np.around(timer() - self.start, 4)} seconds at a rate of {np.around((timer() - self.start)/self.iter, 4)} s/iter')
