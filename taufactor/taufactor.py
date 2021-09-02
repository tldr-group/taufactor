"""Main module."""
import numpy as np
import cupy as cp
from timeit import default_timer as timer
import matplotlib.pyplot as plt


class Solver:
    """
    Default solver for two phase images. Once solve method is
    called, tau, D_eff and D_rel are available as attributes.
    """

    def __init__(self, img, precision=cp.single, bc=(-0.5, 0.5), D_0=1):
        """
        Initialise parameters, conc map and other tools that can be re-used
        for multiple solves.

        :param img: input img with 1s conductive and 0s non-conductive
        :param precision:  cp.single or cp.double
        :param bc: Upper and lower boundary conditions. Leave as default.
        :param D_0: reference material diffusivity
        """
        # add batch dim now for consistency
        self.D_0 = D_0
        self.top_bc, self.bot_bc = bc
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        # Check number of phases and correct labelling
        if len(np.unique(img)) > 2:
            raise AssertionError(
                "Attempting to run binary solver on multi-phase image. Either use MultiPhaseSolver for combined effective tortuosity, or re-label phase of interest to 1 and all other phases to 0.")
        if len(np.unique(img)) == 2:
            if 1 not in np.unique(img):
                raise AssertionError(
                    "Conductive phase must be labelled 1.")
            if 0 not in np.unique(img):
                raise AssertionError(
                    "Non-conductive phase must be labelled 0.")

        self.cpu_img = img
        self.precision = precision
        # VF calc
        self.VF = np.mean(img)
        # save original image in cuda
        img = cp.array(img, dtype=self.precision)
        self.ph_bot = cp.sum(img[:, -1]) * self.bot_bc
        self.ph_top = cp.sum(img[:, 0]) * self.top_bc

        # init conc
        self.conc = self.init_conc(img)

        # create nn map
        self.nn = self.init_nn(img)

        # checkerboarding
        self.w = 2 - cp.pi / (1.5 * img.shape[1])
        self.cb = self.init_cb(img)

        # solving params
        bs, x, y, z = self.cpu_img.shape
        self.L_A = x / (z * y)
        self.converged = False
        self.semi_converged = False
        self.iter = 0
        img = None

        # Results
        self.tau = None
        self.D_eff = None
        self.D_mean = None

    def init_conc(self, img):
        bs, x, y, z = img.shape
        sh = 1 / (x * 2)
        vec = cp.linspace(self.top_bc + sh, self.bot_bc - sh, x)
        for i in range(2):
            vec = cp.expand_dims(vec, -1)
        vec = cp.expand_dims(vec, 0)
        vec = vec.repeat(z, -1)
        vec = vec.repeat(y, -2)
        vec = vec.repeat(bs, 0)
        vec = vec.astype(self.precision)

        return self.pad(img * vec, [self.top_bc * 2, self.bot_bc * 2])

    def init_nn(self, img):
        img2 = self.pad(self.pad(img, [2, 2]))
        nn = cp.zeros_like(img2, dtype=self.precision)
        # iterate through shifts in the spatial dimensions
        for dim in range(1, 4):
            for dr in [1, -1]:
                nn += cp.roll(img2, dr, dim)
        # remove the two paddings
        nn = self.crop(nn, 2)
        # avoid div 0 errors
        nn[img == 0] = cp.inf
        nn[nn == 0] = cp.inf
        return nn

    def init_cb(self, img):
        bs, x, y, z = img.shape
        cb = np.zeros([x, y, z])
        a, b, c = np.meshgrid(range(x), range(y), range(z), indexing='ij')
        cb[(a + b + c) % 2 == 0] = 1
        cb *= self.w
        return [cp.roll(cp.array(cb), sh, 0) for sh in [0, 1]]

    def pad(self, img, vals=[0] * 6):
        while len(vals) < 6:
            vals.append(0)
        to_pad = [1]*8
        to_pad[:2] = (0, 0)
        img = cp.pad(img, to_pad, 'constant')
        img[:, 0], img[:, -1] = vals[:2]
        img[:, :, 0], img[:, :, -1] = vals[2:4]
        img[:, :, :, 0], img[:, :, :, -1] = vals[4:]
        return img

    def crop(self, img, c=1):
        return img[:, c:-c, c:-c, c:-c]

    def solve(self, iter_limit=5000, verbose=True, conv_crit=2*10**-2):
        """
        run a solve simulation

        :param iter_limit: max iterations before aborting, will attempt double for the same no. iterations
        if initialised as singles
        :param verbose: Whether to print tau. Can be set to 'per_iter' for more feedback
        :param conv_crit: convergence criteria, minimum percent difference between
        max and min flux through a given layer
        :return: tau
        """
        start = timer()
        while not self.converged:
            out = self.conc[:, 2:, 1:-1, 1:-1] + \
                self.conc[:, :-2, 1:-1, 1:-1] + \
                self.conc[:, 1:-1, 2:, 1:-1] + \
                self.conc[:, 1:-1, :-2, 1:-1] + \
                self.conc[:, 1:-1, 1:-1, 2:] + \
                self.conc[:, 1:-1, 1:-1, :-2]
            out /= self.nn
            if self.iter % 20 == 0:
                lt = abs(cp.sum(out[:, 0]) - self.ph_top)
                lb = abs(cp.sum(out[:, -1]) - self.ph_bot)
                self.converged, D_rel = self.check_convergence(
                    lt, lb, verbose, conv_crit, start, iter_limit)
            out -= self.crop(self.conc, 1)
            out *= self.cb[self.iter % 2]
            self.conc[:, 1:-1, 1:-1, 1:-1] += out
            self.iter += 1
        self.D_mean = self.D_0
        self.tau = self.VF/D_rel if D_rel != 0 else cp.inf
        self.D_eff = self.D_mean*D_rel
        if verbose:
            print('tau:', self.tau,
                  'after: ', self.iter, 'iterations in: ', np.around(
                      timer() - start, 4),
                  'seconds at a rate of', np.around((timer() - start)/self.iter, 4), 's/iter')
        return self.tau

    def check_convergence(self, lt, lb, verbose, conv_crit, start, iter_limit):
        loss = lt - lb
        # print progress
        if self.iter % 100 == 0:
            if verbose == 'per_iter':
                print(self.iter, abs(loss))

        # check for convergence if loss is good
        if abs(loss) < conv_crit * 0.01:
            if self.semi_converged:
                cvf = self.check_vertical_flux(conv_crit)
                if cvf:
                    if cvf == 'zero_flux':
                        return True, 0
                    return True, ((lt + lb) * self.L_A / abs(self.top_bc - self.bot_bc)).get()
            else:
                self.semi_converged = True
        else:
            self.semi_converged = False

        # increase precision to double if currently single
        if self.iter >= iter_limit:
            if self.precision == cp.single:
                print('increasing precision to double')
                self.iter = 0
                self.conc = cp.array(self.conc, dtype=cp.double)
                self.nn = cp.array(self.nn, dtype=cp.double)
                self.precision = cp.double
            else:
                print('Did not converge in the iteration limit')
                return True, ((lt + lb) * self.L_A / abs(self.top_bc - self.bot_bc)).get()

        return False, False

    def check_vertical_flux(self, conv_crit):
        vert_flux = self.conc[:, 1:-1, 1:-1, 1:-1] - \
            self.conc[:, :-2, 1:-1, 1:-1]
        vert_flux[self.conc[:, :-2, 1:-1, 1:-1] == 0] = 0
        vert_flux[self.conc[:, 1:-1, 1:-1, 1:-1] == 0] = 0
        fl = cp.sum(vert_flux, (0, 2, 3))[1:-1]
        err = (fl.max() - fl.min())*2/(fl.max() + fl.min())
        if err < conv_crit or np.isnan(err).item():
            return True
        if fl.min() == 0:
            return 'zero_flux'
        return False

    def conc_map(self, lay=0):
        """
        Plots a concentration map perpendicular to the direction of flow
        :param lay: depth to plot
        :return: 3D conc map
        """
        img = self.conc[0, 1:-1, 1:-1, 1:-1].get()
        img[self.cpu_img[0, :, :, :] == 0] = -1
        plt.imshow(img[:, :, lay])
        plt.show()
        return img

    def flux_map(self, lay=0):
        """
        Plots a flux map perpendicular to the direction of flow
        :param lay: depth to plot
        :return: 3D flux map
        """
        flux = cp.zeros_like(self.conc)
        ph_map = self.pad(cp.array(self.cpu_img))
        for dim in range(1, 4):
            for dr in [1, -1]:
                flux += abs(cp.roll(self.conc, dr, dim) -
                            self.conc) * cp.roll(ph_map, dr, dim)
        flux = flux[0, 2:-2, 1:-1, 1:-1].get()
        flux[self.cpu_img[0, 1:-1] == 0] = 0
        plt.imshow(flux[:, :, lay])
        return flux


class PeriodicSolver(Solver):
    """
    Periodic Solver (works for non-periodic structures, but has higher RAM requirements)
    Once solve method is called, tau, D_eff and D_rel are available as attributes.
    """

    def __init__(self, img, precision=cp.single, bc=(-0.5, 0.5), D_0=1):
        """
        Initialise parameters, conc map and other tools that can be re-used
        for multiple solves.

        :param img: input img with 1s conductive and 0s non-conductive
        :param precision:  cp.single or cp.double
        :param bc: Upper and lower boundary conditions. Leave as default.
        :param D_0: reference material diffusivity

        """
        super().__init__(img, precision, bc, D_0)
        self.conc = self.pad(self.conc)[:, :, 2:-2, 2:-2]

    def init_nn(self, img):
        img2 = self.pad(self.pad(img, [2, 2]))[:, :, 2:-2, 2:-2]
        nn = cp.zeros_like(img2, dtype=self.precision)
        # iterate through shifts in the spatial dimensions
        for dim in range(1, 4):
            for dr in [1, -1]:
                nn += cp.roll(img2, dr, dim)
        # avoid div 0 errors
        nn = nn[:, 2:-2]
        nn[img == 0] = cp.inf
        nn[nn == 0] = cp.inf
        return nn

    def solve(self, iter_limit=5000, verbose=True, conv_crit=2*10**-2, D_0=1):
        """
        run a solve simulation

        :param iter_limit: max iterations before aborting, will attempt double for the same no. iterations
        if initialised as singles
        :param verbose: Whether to print tau. Can be set to 'per_iter' for more feedback
        :param conv_crit: convergence criteria, minimum percent difference between
        max and min flux through a given layer
        :return: tau
        """
        start = timer()
        while not self.converged:
            out = cp.zeros_like(self.conc)
            for dim in range(1, 4):
                for dr in [1, -1]:
                    out += cp.roll(self.conc, dr, dim)
            out = out[:, 2:-2]
            out /= self.nn
            if self.iter % 50 == 0:
                lt = abs(cp.sum(out[:, 0]) - self.ph_top)
                lb = abs(cp.sum(out[:, -1]) - self.ph_bot)
                self.converged, D_rel = self.check_convergence(
                    lt, lb, verbose, conv_crit, start, iter_limit)
            out -= self.conc[:, 2:-2]
            out *= self.cb[self.iter % 2]
            self.conc[:, 2:-2] += out
            self.iter += 1

        self.D_mean = D_0
        self.tau = self.VF/D_rel if D_rel != 0 else cp.inf
        self.D_eff = D_0*D_rel
        if verbose:
            print('converged to:', self.tau,
                  'after: ', self.iter, 'iterations in: ', np.around(
                      timer() - start, 4),
                  'seconds at a rate of', np.around((timer() - start)/self.iter, 4), 's/iter')
        return self.tau

    def check_vertical_flux(self, conv_crit):
        vert_flux = abs(self.conc - cp.roll(self.conc, 1, 1))
        vert_flux[self.conc == 0] = 0
        vert_flux[cp.roll(self.conc, 1, 1) == 0] = 0
        fl = cp.sum(vert_flux, (0, 2, 3))[3:-2]
        err = (fl.max() - fl.min())*2/(fl.max() + fl.min())
        if err < conv_crit or np.isnan(err):
            return True
        if fl.min() == 0:
            return 'zero_flux'

    def flux_map(self, lay=0):
        """
        Plots a flux map perpendicular to the direction of flow
        :param lay: depth to plot
        :return: 3D flux map
        """
        flux = cp.zeros_like(self.conc)
        ph_map = self.pad(self.pad(cp.array(self.cpu_img)))[:, :, 2:-2, 2:-2]
        for dim in range(1, 4):
            for dr in [1, -1]:
                flux += abs(cp.roll(self.conc, dr, dim) -
                            self.conc) * cp.roll(ph_map, dr, dim)
        flux = flux[0, 2:-2].get()
        flux[self.cpu_img[0] == 0] = 0
        plt.imshow(flux[:, :, lay])
        return flux

    def conc_map(self, lay=0):
        """
        Plots a concentration map perpendicular to the direction of flow
        :param lay: depth to plot
        :return: 3D conc map
        """
        img = self.conc[0, 2:-2].get()
        img[self.cpu_img[0] == 0] = -1
        plt.imshow(img[:, :, lay])
        plt.show()


class MultiPhaseSolver(Solver):
    """
    Multi=phase solver for two phase images. Once solve method is
    called, tau, D_eff and D_rel are available as attributes.
    """

    def __init__(self, img, cond=None, precision=cp.single, bc=(-0.5, 0.5)):
        """
        Initialise parameters, conc map and other tools that can be re-used
        for multiple solves.

        :param img: input img with n conductive phases labelled as integers, and 0s for non-conductive
        :param cond: dict with n phase labels as keys, and their corresponding conductivities as values e.g
        for a 2 phase material, {1:0.543, 2: 0.420}, with 1s and 2s in the input img
        :param precision:  cp.single or cp.double
        :param bc: Upper and lower boundary conditions. Leave as default.
        :param D_0: reference material diffusivity
        """
        if cond is None:
            cond = {p: 1 for p in np.unique(img) if p != 0}
        if 0 in cond.values() or 0 in cond.keys():
            raise ValueError(
                '0 conductivity phase: non-conductive phase should be labelled 0 in the input image and ommitted from the cond argument')

        self.cond = {ph: 0.5 / c for ph, c in cond.items()}
        super().__init__(img, precision, bc)
        self.pre_factors = self.nn[1:]
        self.nn = self.nn[0]
        self.VF = {p: np.mean(img == p) for p in np.unique(img)}

    def init_nn(self, img):
        # conductivity map
        img2 = cp.zeros_like(img)
        for ph in self.cond:
            c = self.cond[ph]
            img2[img == ph] = c
        img2 = self.pad(self.pad(img2))
        img2[:, 1] = img2[:, 2]
        img2[:, -2] = img2[:, -3]
        nn = cp.zeros_like(img2, dtype=self.precision)
        # iterate through shifts in the spatial dimensions
        nn_list = []
        for dim in range(1, 4):
            for dr in [1, -1]:
                shift = cp.roll(img2, dr, dim)
                sum = img2 + shift
                sum[shift == 0] = 0
                sum[img2 == 0] = 0
                sum = 1/sum
                sum[sum == cp.inf] = 0
                nn += sum
                nn_list.append(self.crop(sum, 1))
        # remove the two paddings
        nn = self.crop(nn, 2)
        # avoid div 0 errors
        nn[img == 0] = cp.inf
        nn[nn == 0] = cp.inf
        nn_list.insert(0, nn)
        return nn_list

    def init_conc(self, img):
        bs, x, y, z = img.shape
        sh = 1 / (x + 1)
        vec = cp.linspace(self.top_bc + sh, self.bot_bc - sh, x)
        for i in range(2):
            vec = cp.expand_dims(vec, -1)
        vec = cp.expand_dims(vec, 0)
        vec = vec.repeat(z, -1)
        vec = vec.repeat(y, -2)
        vec = vec.repeat(bs, 0)
        vec = vec.astype(self.precision)
        img1 = cp.array(img)
        img1[img1 > 1] = 1
        return self.pad(img1 * vec, [self.top_bc, self.bot_bc])

    def solve(self, iter_limit=5000, verbose=True, conv_crit=2*10**-2):
        """
        run a solve simulation

        :param iter_limit: max iterations before aborting, will attempt double for the same no. iterations
        if initialised as singles
        :param verbose: Whether to print tau. Can be set to 'per_iter' for more feedback
        :param conv_crit: convergence criteria, minimum percent difference between
        max and min flux through a given layer
        :return: tau
        """

        start = timer()
        while not self.converged:
            self.iter += 1
            out = self.conc[:, 2:, 1:-1, 1:-1] * self.pre_factors[0][:, 2:, 1:-1, 1:-1] + \
                self.conc[:, :-2, 1:-1, 1:-1] * self.pre_factors[1][:, :-2, 1:-1, 1:-1] + \
                self.conc[:, 1:-1, 2:, 1:-1] * self.pre_factors[2][:, 1:-1, 2:, 1:-1] + \
                self.conc[:, 1:-1, :-2, 1:-1] * self.pre_factors[3][:, 1:-1, :-2, 1:-1] + \
                self.conc[:, 1:-1, 1:-1, 2:] * self.pre_factors[4][:, 1:-1, 1:-1, 2:] + \
                self.conc[:, 1:-1, 1:-1, :-2] * \
                self.pre_factors[5][:, 1:-1, 1:-1, :-2]
            out /= self.nn
            if self.iter % 20 == 0:
                self.converged, self.D_eff = self.check_convergence(
                    verbose, conv_crit, start, iter_limit)
            out -= self.crop(self.conc, 1)
            out *= self.cb[self.iter % 2]
            self.conc[:, 1:-1, 1:-1, 1:-1] += out

        if len(np.array([self.VF[z] for z in self.VF.keys() if z != 0])) > 0:
            self.D_mean = np.sum(
                np.array([self.VF[z]*(1/(2*self.cond[z])) for z in self.VF.keys() if z != 0]))
        else:
            self.D_mean = 0
        self.tau = self.D_mean/self.D_eff if self.D_eff != 0 else cp.inf
        if verbose:
            print('tau:', self.tau,
                  'after: ', self.iter, 'iterations in: ', np.around(
                      timer() - start, 4),
                  'seconds at a rate of', np.around((timer() - start)/self.iter, 4), 's/iter')
        return self.tau

    def check_convergence(self, verbose, conv_crit, start, iter_limit):
        # print progress
        if self.iter % 100 == 0:
            loss, flux = self.check_vertical_flux(conv_crit)
            if verbose == 'per_iter':
                print(loss)
            if abs(loss) < conv_crit or np.isnan(loss).item():
                self.converged = True
                b, x, y, z = self.cpu_img.shape
                flux *= (x+1)/(y*z)
                return True, flux.get()

            # increase precision to double if currently single
            if self.iter >= iter_limit:
                if self.precision == cp.single:
                    print('increasing precision to double')
                    self.iter = 0
                    self.conc = cp.array(self.conc, dtype=cp.double)
                    self.nn = cp.array(self.nn, dtype=cp.double)
                    self.precision = cp.double
                else:
                    print('Did not converge in the iteration limit')
                    b, x, y, z = self.cpu_img.shape
                    flux *= (x+1)/(y*z)
                    return True, flux.get()
        return False, False

    def check_vertical_flux(self, conv_crit):
        vert_flux = (self.conc[:, 1:-1, 1:-1, 1:-1] - self.conc[:,
                     :-2, 1:-1, 1:-1]) * self.pre_factors[1][:, :-2, 1:-1, 1:-1]
        vert_flux[self.nn == cp.inf] = 0
        fl = cp.sum(vert_flux, (0, 2, 3))
        err = (fl.max() - fl.min())*2/(fl.max() + fl.min())
        if abs(fl).min() == 0:
            return 0, cp.array([0], dtype=self.precision)
        return err, fl.mean()
