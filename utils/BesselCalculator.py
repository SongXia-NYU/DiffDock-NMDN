import math
import os
import torch
import torch.nn

from utils.basis_utils import bessel_basis, real_sph_harm
from utils.utils_functions import get_device


def _cutoff_fn(D, cutoff):
    """
    Private function called by rbf_expansion(D, K=64, cutoff=10)
    """
    x = D / cutoff
    x3 = x ** 3
    x4 = x3 * x
    x5 = x4 * x

    result = 1 - 6 * x5 + 15 * x4 - 10 * x3
    return result


def rbf_expansion_phynet(D, centers, widths, cutoff):
    """
    The rbf expansion of a distance
    Input D: matrix that contains the distance between to atoms
          K: Number of generated distance features
    Output: A matrix containing rbf expanded distances
    """

    rbf = _cutoff_fn(D, cutoff) * torch.exp(-widths * (torch.exp(-D) - centers) ** 2)
    return rbf


def bessel_expansion_raw(dist, numbers, cutoff):
    """
    Bessel expansion function WITHOUT continuous cutoff
    :param dist:
    :param numbers:
    :param cutoff:
    :return:
    """
    n_rbf = torch.arange(1, numbers + 1).view(1, -1).type(dist.type()).to(get_device())
    return math.sqrt(2.0 / cutoff) * torch.sin(n_rbf * dist * math.pi / cutoff) / dist


def bessel_expansion_continuous(dist, numbers, cutoff, p=6):
    cutoff = torch.Tensor([cutoff]).type(dist.type())
    continuous_cutoff = _cutoff_fn_bessel(dist / cutoff, cutoff, p)
    return bessel_expansion_raw(dist, numbers, cutoff) * continuous_cutoff


def _cutoff_fn_bessel(d_expanded, cutoff, p):
    p = torch.Tensor([p]).type(d_expanded.type())
    return 1 - (p + 1) * d_expanded.pow(p) + p * d_expanded.pow(p + 1.0)


class BesselCalculator:
    def __init__(self, n_srbf, n_shbf):
        """

        :param n_srbf: number of radius expansion for Bessel function: used in a_ijk
        :param n_shbf: number of degree expansion for Bessel function: used in a_ijk
        """
        import sympy as sym

        self.n_srbf = n_srbf
        self.n_shbf = n_shbf

        # retrieve formulas
        self._bessel_formulas = bessel_basis(n_shbf, n_srbf)
        self._sph_harm_formulas = real_sph_harm(n_shbf)
        self._funcs = []

        relative_path = '../data'
        abs_path = os.path.dirname(__file__)
        fn_path = os.path.join(abs_path, relative_path, self._get_file_name())

        # convert to numpy functions
        if os.path.exists(fn_path):
            self._funcs = torch.load(fn_path)
        else:
            x = sym.symbols('x')
            theta = sym.symbols('theta')
            for i in range(n_shbf + 1):
                for j in range(n_srbf):
                    self._funcs.append(sym.lambdify(
                        [x, theta], self._sph_harm_formulas[i][0] * self._bessel_formulas[i][j], modules=torch))
            # Not working right now
            # torch.save(self._funcs, fn_path)

    def _get_file_name(self):
        return 'bessel_fns_{}_{}.npy'.format(self.n_srbf, self.n_shbf)

    def cal_sbf(self, dist, angle, feature_dist):
        # t0 = time.time()

        sbf = [f(dist/feature_dist, angle).view(-1, 1) for f in self._funcs]

        # t0 = record_data('sbf.bessel_fns', t0)

        sbf = torch.cat(sbf, dim=-1)

        # t0 = record_data('sbf.cat', t0)
        return sbf

    @staticmethod
    def cal_rbf(dist, feature_dist, n_rbf):
        return bessel_expansion_raw(dist, n_rbf, feature_dist)


def _plot_n_save():
    """
    internal use only
    :return:
    """
    import matplotlib.pyplot as plt

    x = torch.arange(0.01, 10., 0.01).view(-1, 1).cuda()
    _y_raw = bessel_expansion_raw(x, 5, 10.)
    _y_cont = bessel_expansion_continuous(x, 5, 10., 6)

    plt.figure(figsize=(15, 10))
    for i in range(5):
        plt.plot(x.cpu(), _y_cont[:, i].cpu(), label='N={}'.format(i))
    plt.xlabel('distance, A')
    plt.ylabel('expansion function')
    plt.title('Bessel expansion, continuous cutoff at p=6')
    plt.legend()
    plt.savefig('../figures/bessel_cont.png')
    plt.show()


if __name__ == '__main__':
    _plot_n_save()
    print('Finished')
