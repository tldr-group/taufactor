"""Main module."""
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
try:
    import torch
except ImportError:
    raise ImportError("Pytorch is required to use this package. Please install pytorch and try again. More information about TauFactor's requirements can be found at https://taufactor.readthedocs.io/en/latest/")
import warnings


class Solver:
    """
    Default solver for two phase images. Once solve method is
    called, tau, D_eff and D_rel are available as attributes.
    """

    def __init__(self, img, bc=(-0.5, 0.5), D_0=1, device=torch.device('cuda')):
        """
        Initialise parameters, conc map and other tools that can be re-used
        for multiple solves.

        :param img: input img with 1s conductive and 0s non-conductive
        :param precision:  cp.single or cp.double
        :param bc: Upper and lower boundary conditions. Leave as default.
        :param D_0: reference material diffusivity
        :param device: pytorch device, can be cuda or cpu 
        """
        # add batch dim now for consistency
        self.D_0 = D_0
        self.top_bc, self.bot_bc = bc
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        self.cpu_img = img
        self.precision = torch.float
        self.device = torch.device(device)
        # check device is available
        if torch.device(device).type.startswith('cuda') and not torch.cuda.is_available():
            self.device = torch.device('cpu')
            warnings.warn(
                "CUDA not available, defaulting device to cpu. To avoid this warning, explicitly set the device when initialising the solver with device=torch.device('cpu')")
        # save original image in cuda
        img = torch.tensor(img, dtype=self.precision, device=self.device)
        self.VF = torch.mean(img)

        if len(torch.unique(img).shape) > 2 or torch.unique(img).max() not in [0, 1] or torch.unique(img).min() not in [0, 1]:
            raise ValueError(
                f'Input image must only contain 0s and 1s. Your image must be segmented to use this tool. If your image has been segmented, ensure your labels are 0 for non-conductive and 1 for conductive phase. Your image has the following labels: {torch.unique(img).numpy()}. If you have more than one conductive phase, use the multi-phase solver.')

        # calculate
        self.ph_bot = torch.sum(img[:, -1]).to(self.device) * self.bot_bc
        self.ph_top = torch.sum(img[:, 0]).to(self.device) * self.top_bc
        # init conc
        self.conc = self.init_conc(img)
        # create nn map
        self.nn = self.init_nn(img)
        # overrelaxation factor
        self.w = 2 - torch.pi / (1.5 * img.shape[1])
        # checkerboarding to ensure stable steps
        self.cb = self.init_cb(img)
        bs, x, y, z = self.cpu_img.shape
        self.L_A = x / (z * y)
        # solving params
        self.converged = False
        self.semi_converged = False
        self.old_fl = -1
        self.iter = 0
        img = None
        # Results
        self.tau = None
        self.D_eff = None
        self.D_mean = None

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

    def init_cb(self, img):
        """Creates a chequerboard to ensure neighbouring pixels dont update,
        which can cause instability"""
        bs, x, y, z = img.shape
        cb = np.zeros([x, y, z])
        a, b, c = np.meshgrid(range(x), range(y), range(z), indexing='ij')
        cb[(a + b + c) % 2 == 0] = 1
        cb *= self.w
        return [torch.roll(torch.tensor(cb, dtype=self.precision).to(self.device), sh, 0) for sh in [0, 1]]

    def pad(self, img, vals=[0] * 6):
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

    def crop(self, img, c=1):
        """removes a layer from the volume edges"""
        return img[:, c:-c, c:-c, c:-c]

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

        with torch.no_grad():
            start = timer()
            while not self.converged:
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
                    self.converged = self.check_convergence(
                        verbose, conv_crit, start, iter_limit)
                # efficient way of adding flux to old conc with overrelaxation
                out -= self.crop(self.conc, 1)
                out *= self.cb[self.iter % 2]
                self.conc[:, 1:-1, 1:-1, 1:-1] += out
                self.iter += 1
            self.D_mean = self.D_0
            self.D_eff = self.D_mean*self.D_rel
            self.end_simulation(iter_limit, verbose, start)
            return self.tau

    def check_convergence(self, verbose, conv_crit, start, iter_limit):
        # print progress
        self.semi_converged, self.new_fl, err = self.check_vertical_flux(
            conv_crit)
        self.D_rel = ((self.new_fl) * self.L_A /
                      abs(self.top_bc - self.bot_bc)).cpu()
        self.tau = self.VF / \
            self.D_rel if self.D_rel != 0 else torch.tensor(torch.inf)
        if self.semi_converged == 'zero_flux':
            return True

        if verbose == 'per_iter':
            print(
                f'Iter: {self.iter}, conv error: {abs(err.item())}, tau: {self.tau.item()}')

        if self.semi_converged:
            self.converged = self.check_rolling_mean(conv_crit=1e-3)

            if not self.converged:
                self.old_fl = self.new_fl
                return False
            else:
                return True
        else:
            self.old_fl = self.new_fl
            return False

    def check_vertical_flux(self, conv_crit):
        vert_flux = self.conc[:, 1:-1, 1:-1, 1:-1] - \
            self.conc[:, :-2, 1:-1, 1:-1]
        vert_flux[self.conc[:, :-2, 1:-1, 1:-1] == 0] = 0
        vert_flux[self.conc[:, 1:-1, 1:-1, 1:-1] == 0] = 0
        fl = torch.sum(vert_flux, (0, 2, 3))[1:-1]
        err = (fl.max() - fl.min())*2/(fl.max() + fl.min())
        if err < conv_crit or torch.isnan(err).item():
            return True, torch.mean(fl), err
        if fl.min() == 0:
            return 'zero_flux', torch.mean(fl), err
        return False, torch.mean(fl), err

    def check_rolling_mean(self, conv_crit):
        err = (self.new_fl - self.old_fl) / (self.new_fl + self.old_fl)
        if err < conv_crit:
            return True
        else:
            return False

    def end_simulation(self, iter_limit, verbose, start):
        converged = 'converged to'
        if self.iter >= iter_limit - 1:
            print('Warning: not converged')
            converged = 'unconverged value of tau'

        if verbose:
            print(f'{converged}: {self.tau} \
                  after: {self.iter} iterations in: {np.around(timer() - start, 4)}  \
                  seconds at a rate of {np.around((timer() - start)/self.iter, 4)} s/iter')


class PeriodicSolver(Solver):
    """
    Periodic Solver (works for non-periodic structures, but has higher RAM requirements)
    Once solve method is called, tau, D_eff and D_rel are available as attributes.
    """

    def __init__(self, img, bc=(-0.5, 0.5), D_0=1, device=torch.device('cuda:0')):
        """
        Initialise parameters, conc map and other tools that can be re-used
        for multiple solves.

        :param img: input img with 1s conductive and 0s non-conductive
        :param precision:  cp.single or cp.double
        :param bc: Upper and lower boundary conditions. Leave as default.
        :param D_0: reference material diffusivity

        """
        super().__init__(img, bc, D_0, device)
        self.conc = self.pad(self.conc)[:, :, 2:-2, 2:-2].to(self.device)

    def init_nn(self, img):
        img2 = self.pad(self.pad(img, [2, 2]))[:, :, 2:-2, 2:-2]
        nn = torch.zeros_like(img2)
        # iterate through shifts in the spatial dimensions
        for dim in range(1, 4):
            for dr in [1, -1]:
                nn += torch.roll(img2, dr, dim)
        # avoid div 0 errors
        nn = nn[:, 2:-2]
        nn[img == 0] = torch.inf
        nn[nn == 0] = torch.inf
        return nn.to(self.device)

    def solve(self, iter_limit=5000, verbose=True, conv_crit=2*10**-2, D_0=1):
        """
        run a solve simulation

        :param iter_limit: max iterations before aborting, will attemtorch double for the same no. iterations
        if initialised as singles
        :param verbose: Whether to print tau. Can be set to 'per_iter' for more feedback
        :param conv_crit: convergence criteria, minimum percent difference between
        max and min flux through a given layer
        :return: tau
        """
        start = timer()
        while not self.converged:
            out = torch.zeros_like(self.conc)
            for dim in range(1, 4):
                for dr in [1, -1]:
                    out += torch.roll(self.conc, dr, dim)
            out = out[:, 2:-2]
            out /= self.nn
            if self.iter % 50 == 0:
                self.converged = self.check_convergence(
                    verbose, conv_crit, start, iter_limit)
            out -= self.conc[:, 2:-2]
            out *= self.cb[self.iter % 2]
            self.conc[:, 2:-2] += out
            self.iter += 1

        self.D_mean = D_0
        self.D_eff = D_0*self.D_rel
        self.end_simulation(iter_limit, verbose, start)
        return self.tau

    def check_vertical_flux(self, conv_crit):
        vert_flux = abs(self.conc - torch.roll(self.conc, 1, 1))
        vert_flux[self.conc == 0] = 0
        vert_flux[torch.roll(self.conc, 1, 1) == 0] = 0
        fl = torch.sum(vert_flux, (0, 2, 3))[3:-2]
        err = (fl.max() - fl.min())*2/(fl.max() + fl.min())
        if err < conv_crit or torch.isnan(err).item():
            return True, torch.mean(fl), err
        if fl.min() == 0:
            return 'zero_flux', torch.mean(fl), err
        return False, torch.mean(fl), err


class MultiPhaseSolver(Solver):
    """
    Multi=phase solver for two phase images. Once solve method is
    called, tau, D_eff and D_rel are available as attributes.
    """

    def __init__(self, img, cond={1: 1}, bc=(-0.5, 0.5), device=torch.device('cuda:0')):
        """
        Initialise parameters, conc map and other tools that can be re-used
        for multiple solves.

        :param img: input img with n conductive phases labelled as integers, and 0s for non-conductive
        :param cond: dict with n phase labels as keys, and their corresponding conductivities as values e.g
        for a 2 phase material, {1:0.543, 2: 0.420}, with 1s and 2s in the input img
        :param precision:  cp.single or cp.double
        :param bc: Upper and lower boundary conditions. Leave as default.
        """

        if (0 in cond.values()):
            raise ValueError(
                '0 conductivity phase: non-conductive phase should be labelled 0 in the input image and ommitted from the cond argument')
        if (0 in cond.keys()):
            raise ValueError(
                '0 cannot be used as a conductive phase label, please use a positive integer and leave 0 for non-conductive phase')

        self.cond = {ph: 0.5 / c for ph, c in cond.items()}
        self.top_bc, self.bot_bc = bc
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        self.cpu_img = img
        self.precision = torch.float

        self.device = torch.device(device)
        # check device is available
        if torch.device(device).type.startswith('cuda') and not torch.cuda.is_available():
            self.device = torch.device('cpu')
            warnings.warn(
                "CUDA not available, defaulting device to cpu. To avoid this warning, explicitly set the device when initialising the solver with device=torch.device('cpu')")
        # save original image in cuda
        img = torch.tensor(img, dtype=self.precision, device=self.device)

        # calculate
        self.ph_bot = torch.sum(img[:, -1]).to(self.device) * self.bot_bc
        self.ph_top = torch.sum(img[:, 0]).to(self.device) * self.top_bc
        # init conc
        self.conc = self.init_conc(img)
        # create nn map
        self.nn = self.init_nn(img)
        # overrelaxation factor
        self.w = 2 - torch.pi / (1.5 * img.shape[1])
        # checkerboarding to ensure stable steps
        self.cb = self.init_cb(img)
        bs, x, y, z = self.cpu_img.shape
        self.L_A = x / (z * y)
        # solving params
        self.converged = False
        self.iter = 0
        # Results
        self.tau = None
        self.D_eff = None
        self.pre_factors = self.nn[1:]
        self.nn = self.nn[0]
        self.semi_converged = False
        self.old_fl = -1
        self.VF = {p: np.mean(img.cpu().numpy() == p)
                   for p in np.unique(img.cpu().numpy())}

        if len(np.array([self.VF[z] for z in self.VF.keys() if z != 0])) > 0:
            self.D_mean = np.sum(
                np.array([self.VF[z]*(1/(2*self.cond[z])) for z in self.VF.keys() if z != 0]))
        else:
            self.D_mean = 0

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
                self.converged = self.check_convergence(
                    verbose, conv_crit, start, iter_limit)
            out -= self.crop(self.conc, 1)
            out *= self.cb[self.iter % 2]
            self.conc[:, 1:-1, 1:-1, 1:-1] += out

        self.end_simulation(iter_limit, verbose, start)
        return self.tau

    def check_convergence(self, verbose, conv_crit, start, iter_limit):
        # print progress
        if self.iter % 100 == 0:
            self.semi_converged, self.new_fl, err = self.check_vertical_flux(
                conv_crit)
            b, x, y, z = self.cpu_img.shape
            self.D_eff = (self.new_fl*(x+1)/(y*z)).cpu()
            self.tau = self.D_mean / \
                self.D_eff if self.D_eff != 0 else torch.tensor(torch.inf)
            if self.semi_converged == 'zero_flux':
                return True

            if verbose == 'per_iter':
                print(
                    f'Iter: {self.iter}, conv error: {abs(err.item())}, tau: {self.tau.item()}')

            if self.semi_converged:
                self.converged = self.check_rolling_mean(conv_crit=1e-3)

                if not self.converged:
                    self.old_fl = self.new_fl
                    return False
                else:
                    return True
            else:
                self.old_fl = self.new_fl
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

    def check_vertical_flux(self, conv_crit):
        vert_flux = (self.conc[:, 1:-1, 1:-1, 1:-1] - self.conc[:,
                     :-2, 1:-1, 1:-1]) * self.pre_factors[1][:, :-2, 1:-1, 1:-1]
        vert_flux[self.nn == torch.inf] = 0
        fl = torch.sum(vert_flux, (0, 2, 3))[2:-2]
        print(fl.argmin(), fl.argmax())
        err = (fl.max() - fl.min())*2/(fl.max() + fl.min())
        if err < conv_crit or torch.isnan(err).item():
            return True, torch.mean(fl), err
        if fl.min() == 0:
            return 'zero_flux', torch.mean(fl), err
        return False, torch.mean(fl), err


class ElectrodeSolver():
    """
    Electrode Solver - solves the electrode tortuosity factor system (migration and capacitive current between current collector and solid/electrolyte interface)
    Once solve method is called, tau, D_eff and D_rel are available as attributes.
    """

    def __init__(self, img, omega=1e-6, device=torch.device('cuda')):

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
        self.iter = 0

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
                        self.end_simulation()
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

        while not self.converged:
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
        # self.end_simulation(iter_limit, verbose, start)

    def end_simulation(self, ):
        if self.iter == self.iter_limit - 1:
            print('Warning: not converged')
            converged = 'unconverged value of tau'
        converged = 'converged to'
        if self.verbose:
            print(f'{converged}: {self.tau_e} after: {self.iter} iterations in: {np.around(timer() - self.start, 4)} seconds at a rate of {np.around((timer() - self.start)/self.iter, 4)} s/iter')
