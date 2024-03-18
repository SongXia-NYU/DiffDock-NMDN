import os
import numpy as np
import torch
import torch_geometric
from torch_scatter import scatter

from utils.utils_functions import floating_type

"""
Pytorch implementation of Grimme's D3 method (only Becke-Johnson damping is implemented)
Grimme, Stefan, et al. "A consistent and accurate ab initio parametrization of density functional dispersion correction (DFT-D) for the 94 elements H-Pu." The Journal of chemical physics 132.15 (2010): 154104.
"""
# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# relative filepath to package folder
package_directory = os.path.dirname(os.path.abspath(__file__))

# conversion factors used in grimme d3 code
d3_autoang = 0.52917726  # for converting distance from bohr to angstrom
d3_autoev = 27.21138505  # for converting a.u. to eV

# global parameters (the values here are the standard for HF)
d3_s6 = 1.0000
d3_s8 = 0.9171
d3_a1 = 0.3385
d3_a2 = 2.8830
d3_k1 = 16.000
d3_k2 = 4 / 3
d3_k3 = -4.000

# tables with reference values
_d3_c6ab = np.load(os.path.join(package_directory, "tables", "c6ab.npy"))
d3_c6ab = torch.Tensor(_d3_c6ab).type(floating_type).to(device)
_d3_r0ab = np.load(os.path.join(package_directory, "tables", "r0ab.npy"))
d3_r0ab = torch.Tensor(_d3_r0ab).type(floating_type).to(device)
_d3_rcov = np.load(os.path.join(package_directory, "tables", "rcov.npy"))
d3_rcov = torch.Tensor(_d3_rcov).type(floating_type).to(device)
_d3_r2r4 = np.load(os.path.join(package_directory, "tables", "r2r4.npy"))
d3_r2r4 = torch.Tensor(_d3_r2r4).type(floating_type).to(device)
d3_maxc = 5  # maximum number of coordination complexes


def _smootherstep(r, cutoff):
    """
    computes a smooth step from 1 to 0 starting at 1 bohr
    before the cutoff
    """
    cuton = cutoff - 1
    x = (cutoff - r) / (cutoff - cuton)
    x2 = x ** 2
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    return torch.where(r <= cuton, torch.ones_like(x),
                       torch.where(r >= cutoff, torch.zeros_like(x), 6 * x5 - 15 * x4 + 10 * x3))


def _ncoord(Zi, Zj, r, idx_i, cutoff=None, k1=d3_k1, rcov=d3_rcov):
    """
    compute coordination numbers by adding an inverse damping function
    """
    rco = rcov[Zi] + rcov[Zj]
    rr = rco.view(-1, 1) / r
    damp = 1.0 / (1.0 + torch.exp(-k1 * (rr - 1.0)))
    if cutoff is not None:
        damp *= _smootherstep(r, cutoff)
    return scatter(reduce='add', src=damp, index=idx_i, dim=0)
    # return torch_geometric.utils.scatter_('add', damp, idx_i)


def _getc6(ZiZj, nci, ncj, c6ab=d3_c6ab, k3=d3_k3):
    """
    interpolate c6
    """
    # gather the relevant entries from the table
    c6ab_ = c6ab[[ZiZj[0, :], ZiZj[1, :]]]

    # for i in range(d3_maxc):
    #     for j in range(d3_maxc):
    #         cn0 = c6ab_[:, i, j, 0].view(-1, 1)
    #         cn1 = c6ab_[:, i, j, 1].view(-1, 1)
    #         cn2 = c6ab_[:, i, j, 2].view(-1, 1)
    #         r = (cn1 - nci) ** 2 + (cn2 - ncj) ** 2
    #         r_save = torch.where(r < r_save, r, r_save)
    #         c6mem = torch.where(r < r_save, cn0, c6mem)
    #         tmp1 = torch.exp(k3 * r)
    #         rsum += torch.where(cn0 > 0.0, tmp1, torch.zeros_like(tmp1))
    #         csum += torch.where(cn0 > 0.0, tmp1 * cn0, torch.zeros_like(tmp1))

    cn0 = c6ab_[:, :, :, 0].view(-1, d3_maxc, d3_maxc, 1)
    cn1 = c6ab_[:, :, :, 1].view(-1, d3_maxc, d3_maxc, 1)
    cn2 = c6ab_[:, :, :, 2].view(-1, d3_maxc, d3_maxc, 1)
    # calculate c6 coefficients
    c6mem = -1.0e99 * torch.ones_like(nci)
    c6mem_ = -1.0e99 * torch.ones_like(cn0)
    r_save_ = 1.0e99 * torch.ones_like(nci)
    r_save = 1.0e99 * torch.ones_like(cn0)
    r = (cn1 - nci.view(-1, 1, 1, 1)) ** 2 + (cn2 - ncj.view(-1, 1, 1, 1)) ** 2
    r_save = torch.where(r < r_save, r, r_save)
    c6mem_ = torch.where(r < r_save, cn0, c6mem_)
    for i in range(d3_maxc):
        for j in range(d3_maxc):
            c6mem = torch.where(c6mem_[:, i, j, :] < r_save_, c6mem_[:, i, j, :], c6mem)
    tmp1 = torch.exp(k3 * r)
    rsum = torch.where(cn0 > 0.0, tmp1, torch.zeros_like(tmp1))
    rsum = torch.sum(rsum, dim=(2, 1))
    csum = torch.where(cn0 > 0.0, tmp1 * cn0, torch.zeros_like(tmp1))
    csum = torch.sum(csum, dim=(2, 1))
    c6 = torch.where(rsum > 0.0, csum / rsum, c6mem)

    if torch.abs(c6).max() > 1e4:
        # print('something')
        c6 = None

    return c6


def edisp(Z, r, idx_i, idx_j, cutoff=None, r2=None,
          r6=None, r8=None, s6=d3_s6, s8=d3_s8, a1=d3_a1, a2=d3_a2, k1=d3_k1, k2=d3_k2,
          k3=d3_k3, c6ab=d3_c6ab, r0ab=d3_r0ab, rcov=d3_rcov, r2r4=d3_r2r4):
    """
    compute d3 dispersion energy in Hartree
    r: distance in bohr!
    """
    # compute all necessary quantities
    Zi = Z.take(idx_i)
    Zj = Z.take(idx_j)
    ZiZj = torch.stack([Zi, Zj], dim=0)  # necessary for gatherin
    nc = _ncoord(Zi, Zj, r, idx_i, cutoff=cutoff, rcov=rcov)  # coordination numbers
    nci = nc.take(idx_i).view(-1, 1)
    ncj = nc.take(idx_j).view(-1, 1)
    c6 = _getc6(ZiZj, nci, ncj, c6ab=c6ab, k3=k3)  # c6 coefficients
    if c6 is None:
        print('WARNING: D3 dispersion error, use 0. instead')
        return torch.as_tensor(0.).type(floating_type).to(device)
    c8 = 3 * c6 * (r2r4.take(Zi).view(-1, 1)) * (r2r4.take(Zj).view(-1, 1))  # c8 coefficient

    # compute all necessary powers of the distance
    if r2 is None:
        r2 = r ** 2  # square of distances
    if r6 is None:
        r6 = r2 ** 3
    if r8 is None:
        r8 = r6 * r2

    # Becke-Johnson damping, zero-damping introduces spurious repulsion
    # and is therefore not supported/implemented
    tmp = a1 * torch.sqrt(c8 / c6) + a2
    tmp2 = tmp ** 2
    tmp6 = tmp2 ** 3
    tmp8 = tmp6 * tmp2
    if cutoff is None:
        e6 = 1 / (r6 + tmp6)
        e8 = 1 / (r8 + tmp8)
    else:  # apply cutoff
        cut2 = cutoff ** 2
        cut6 = cut2 ** 3
        cut8 = cut6 * cut2
        cut6tmp6 = cut6 + tmp6
        cut8tmp8 = cut8 + tmp8
        e6 = 1 / (r6 + tmp6) - 1 / cut6tmp6 + 6 * cut6 / cut6tmp6 ** 2 * (r / cutoff - 1)
        e8 = 1 / (r8 + tmp8) - 1 / cut8tmp8 + 8 * cut8 / cut8tmp8 ** 2 * (r / cutoff - 1)
        e6 = torch.where(r < cutoff, e6, torch.zeros_like(e6))
        e8 = torch.where(r < cutoff, e8, torch.zeros_like(e8))
    e6 = -0.5 * s6 * c6 * e6
    e8 = -0.5 * s8 * c8 * e8
    e_d3 = torch.where(r > 0, e6 + e8, torch.zeros_like(e6 + e8))
    result = scatter(reduce='add', src=e_d3, index=idx_i, dim=0)
    # result = torch_geometric.utils.scatter_('add', e_d3, idx_i)
    return result


def cal_d3_dispersion(Z, batch, edge_dist, edge_index, s6, s8, a1, a2):
    """
    calculate Grimme D3 dispersion energy
    :param Dist_matrix: in angstrom -> will be converted to Bohr
    :return: Energy in eV
    """

    E_atom_d3 = edisp(Z, edge_dist / d3_autoang, idx_i=edge_index[0, :],
                      idx_j=edge_index[1, :], s6=s6, s8=s8, a1=a1, a2=a2)

    return scatter(reduce='add', src=E_atom_d3.view(-1), index=batch, dim=0)
    # return torch_geometric.utils.scatter_('add', E_atom_d3.view(-1), batch)
