import torch
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os.path as osp
import tqdm
from utils.BesselCalculator import bessel_expansion_raw

from utils.utils_functions import _cutoff_fn, gaussian_rbf, softplus_inverse


class RBFVisulizer:
    def __init__(self, n_rbf=64, cutoff=10., save_root="./figures", dens_min=0., dens_max=None, coe=1., linear=False, bessel=False):
        self.n_rbf = n_rbf
        self.save_root = save_root
        self.cutoff = torch.as_tensor(cutoff)
        self.dens_min = dens_min
        self.dens_max = dens_max if dens_max is not None else cutoff
        self.coe = coe
        self.linear = linear
        self.bessel = bessel

        self._name = None

    @property
    def name(self):
        if self._name is None:
            name = "rbf"
            if self.bessel:
                name = "bessel"
            if self.linear:
                name += "_linear"
            self._name = f"{name}_{self.n_rbf}_{self.cutoff.item()}_{self.dens_min}_{self.dens_max}_{self.coe}"
        return self._name

    def run(self):
        centers = softplus_inverse(torch.linspace(math.exp(-self.dens_min), math.exp(-self.dens_max*self.coe), self.n_rbf))
        centers = torch.nn.functional.softplus(centers)

        widths = [softplus_inverse((0.5 / ((1.0 - torch.exp(-self.cutoff)) / self.n_rbf)) ** 2)] * self.n_rbf
        widths = torch.as_tensor(widths)
        widths = torch.nn.functional.softplus(widths)

        distances = torch.arange(0, self.cutoff, 0.01).view(-1, 1)
        effect_dist = torch.exp(-distances*self.coe)
        if self.linear:
            effect_dist = distances / self.cutoff
        rbf = _cutoff_fn(distances, self.cutoff) * torch.exp(-widths * (effect_dist- centers) ** 2)

        if self.bessel:
            rbf = bessel_expansion_raw(distances, self.n_rbf, self.cutoff)
        
        plt.figure()
        for i in tqdm.tqdm(range(self.n_rbf), total=self.n_rbf):
            sns.lineplot(x=distances.view(-1).numpy(), y=rbf[:, i], color="black")
        plt.xlabel("Distance (A)")
        plt.xlim([0, self.cutoff])
        plt.ylim([0, 1])
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])
        plt.tight_layout()
        plt.savefig(osp.join(self.save_root, f"{self.name}.png"), transparent=True)
        plt.close()

        active_rbfs = torch.sum((rbf<0.95) & (rbf > 0.05), dim=-1)
        plt.figure()
        sns.lineplot(x=distances.view(-1).numpy(), y=active_rbfs)
        plt.title("Number of active RBFs (0.05, 0.95)")
        plt.xlabel("Distance (A)")
        plt.tight_layout()
        plt.savefig(osp.join(self.save_root, f"{self.name}_active.png"))
        plt.close()


if __name__ == "__main__":
    rbf = RBFVisulizer(n_rbf=10)
    rbf.run()
