"""Main module."""
import numpy as np
import cupy as cp
from timeit import default_timer as timer

class Solver:
    def __init__(self, img, iter_limit=-1):
        # add batch dim now for consistency
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)

        # VF calc
        self.VF = np.mean(img)
        self.iter_limit = iter_limit
        # save original image in cuda
        img = cp.array(img, dtype=cp.single)
        self.ph_bot = cp.sum(img[:, -1])

        # init conc
        self.conc = self.init_conc(img)

        # create nn map
        self.nn = self.init_nn(img)

        #checkerboarding
        self.w = 2 - cp.pi / (1.5 * img.shape[1])
        self.cb = self.init_cb(img)

        # solving params
        bs, x, y, z = img.shape
        self.L_A = x / (z * y)
        self.converged = False
        self.semi_converged = False
        img = None

    def init_conc(self, img):
        bs, x, y, z = img.shape
        sh = 1 / (x * 2)
        vec = cp.linspace(0 + sh, 1 - sh, x)
        for i in range(2):
            vec = cp.expand_dims(vec, -1)
        vec = cp.expand_dims(vec, 0)
        vec = vec.repeat(z, -1)
        vec = vec.repeat(y, -2)
        vec = vec.repeat(bs, 0)
        vec = vec.astype(cp.single)
        return self.pad(img * vec, [0, 2])


    def init_nn(self, img):
        img2 = self.pad(self.pad(img, [2, 2]))
        nn = cp.zeros_like(img2, dtype=cp.single)
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
        """
        adds padding
        :param img: image to pad
        :param vals: list of values to pad given as [x+, x-, y+, y-, z+, z-]
        :return: padded image
        """
        while len(vals) < 6:
            vals.append(0)
        to_pad = [1]*8
        to_pad[:2] = (0,0)
        img = cp.pad(img, to_pad, 'constant')
        img[:, 0], img[:, -1] = vals[:2]
        img[:, :, 0], img[:, :, -1] = vals[2:4]
        img[:, :, :, 0], img[:, :, :, -1] = vals[4:]
        return img

    def crop(self, img, c = 1):
        return img[:, c:-c, c:-c, c:-c]

    def solve(self):
        """
        iteratively converge the vol
        :param iters: number of steps
        :return:
        """
        start = timer()
        iter = 0
        while not self.converged:
            out = self.conc[:, 2:, 1:-1, 1:-1] + \
                  self.conc[:, :-2, 1:-1, 1:-1] + \
                  self.conc[:, 1:-1, 2:, 1:-1] + \
                  self.conc[:, 1:-1, :-2, 1:-1] + \
                  self.conc[:, 1:-1, 1:-1, 2:] + \
                  self.conc[:, 1:-1, 1:-1, :-2]
            out /= self.nn
            if iter % 10 == 0:
                lt = abs(cp.sum(out[:, 0]))
                lb = abs(cp.sum(out[:, -1]) - self.ph_bot)
                self.check_convergence(lt, lb)
            out -= self.crop(self.conc, 1)
            out *= self.cb[iter%2]
            self.conc[:, 1:-1, 1:-1, 1:-1] += out
            iter += 1
            if iter == self.iter_limit:
                print('Did not converge in the iteration limit')
                return (lt * self.L_A * 2).get()
        iter = int(iter/2)
        print('converged to:', cp.around(lt * self.L_A * 2, 6),
              'after: ', iter, 'iterations in: ', timer() - start,
              'seconds at a rate of', (timer() - start)/iter, 'iters per second')
        return (lt * self.L_A * 2).get()

    def check_convergence(self, lt, lb):
        loss = (lt - lb) / ((lt + lb) / 2)
        if abs(loss) < 7 * 10**-4:
            if self.semi_converged:
                self.converged = True
            else:
                self.semi_converged = True
        else:
            self.semi_converged = False

class PeriodicSolver(Solver):
    def __init__(self, img, iter_limit):
        super().__init__(img, iter_limit)
        self.conc = self.pad(self.conc)[:, :, 2:-2, 2:-2]

    def init_nn(self, img):
        img2 = self.pad(self.pad(img, [2, 2]))[:, :, 2:-2, 2:-2]
        nn = cp.zeros_like(img2, dtype=cp.single)
        # iterate through shifts in the spatial dimensions
        for dim in range(1, 4):
            for dr in [1, -1]:
                nn += cp.roll(img2, dr, dim)
        # avoid div 0 errors
        nn = nn[:, 2:-2]
        nn[img == 0] = cp.inf
        nn[nn == 0] = cp.inf
        return nn

    def solve(self):
        """
        iteratively converge the vol
        :param iters: number of steps
        :return:
        """
        start = timer()
        iter = 0
        while not self.converged:
            out = cp.zeros_like(self.conc)
            for dim in range(1, 4):
                for dr in [1, -1]:
                    out += cp.roll(self.conc, dr, dim)
            out = out[:, 2:-2]
            out /= self.nn
            if iter % 10 == 0:
                lt = abs(cp.sum(out[:, 0]))
                lb = abs(cp.sum(out[:, -1]) - self.ph_bot)
                self.check_convergence(lt, lb)
            out -= self.conc[:, 2:-2]
            out *= self.cb[iter % 2]
            self.conc[:, 2:-2] += out
            iter += 1
            if iter == self.iter_limit:
                print('Did not converge in the iteration limit')
                return (lt * self.L_A * 2).get()
        iter = int(iter/2)
        print('converged to:', cp.around(lt * self.L_A * 2, 6),
              'after: ', iter, 'iterations in: ', timer() - start,
              'seconds at a rate of', (timer() - start)/iter, 's/iter')
        return (lt * self.L_A * 2).get()

