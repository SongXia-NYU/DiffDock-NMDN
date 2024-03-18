import torch

from utils.BesselCalculator import bessel_expansion_raw, bessel_expansion_continuous
from utils.basis_utils import Jn_zeros, Jn, spherical_bessel_formulas, real_sph_harm, Y_l_fast
from utils.utils_functions import floating_type, get_device


class BesselCalculator:
    """
    A faster implementation of bessel calculator
    """
    def __init__(self, n_srbf, n_shbf, envelop_p, cos_theta=True):
        """

        :param n_srbf:
        :param n_shbf:
        :param cos_theta: if True, sbf angle part input will be cos_theta instead of theta
        """
        import sympy as sym

        self.envelop_p = envelop_p
        self.n_srbf = n_srbf
        self.n_shbf = n_shbf
        self.dim_sbf = n_srbf*(n_shbf+1)

        self.z_ln = torch.as_tensor(Jn_zeros(n_shbf, n_srbf)).type(floating_type)
        self.normalizer_tensor = self.get_normal()
        x = sym.symbols('x')
        j_l = spherical_bessel_formulas(n_shbf)
        self.j_l = [sym.lambdify([x], f, modules=torch) for f in j_l]
        if cos_theta:
            Y_l = Y_l_fast(n_shbf)
            angle_input = sym.symbols('z')
        else:
            Y_l = real_sph_harm(n_shbf)
            angle_input = sym.symbols('theta')
        self.Y_l = [sym.lambdify([angle_input], f[0], modules=torch) for f in Y_l]
        self.Y_l[0] = lambda _theta: torch.zeros_like(_theta).fill_(float(Y_l[0][0]))
        self.to(get_device())

    def cal_sbf(self, dist, angle, feature_dist):
        scaled_dist = (dist/feature_dist).view(-1, 1, 1)
        expanded_dist = scaled_dist*self.z_ln
        radius_part = torch.cat([f(expanded_dist[:, [l], :]) for l, f in enumerate(self.j_l)], dim=1)
        angle_part = torch.cat([f(angle).view(-1, 1) for f in self.Y_l], dim=-1)
        result = self.normalizer_tensor.unsqueeze(0) * radius_part * angle_part.unsqueeze(-1)
        return result.view(-1, self.dim_sbf)

    def cal_rbf(self, dist, feature_dist, n_rbf):
        if self.envelop_p > 0:
            return bessel_expansion_continuous(dist, n_rbf, feature_dist, self.envelop_p)
        else:
            return bessel_expansion_raw(dist, n_rbf, feature_dist)

    def get_normal(self):
        normal = torch.zeros_like(self.z_ln)
        for l in range(self.n_shbf+1):
            for n in range(self.n_srbf):
                normal[l][n] = torch.sqrt(2/(Jn(self.z_ln[l][n], l+1))**2)
        return normal

    def to(self, _device):
        self.z_ln = self.z_ln.to(_device)
        self.normalizer_tensor = self.normalizer_tensor.to(_device)


if __name__ == '__main__':
    print('Finished')
