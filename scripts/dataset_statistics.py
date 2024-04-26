from collections import Counter

import ase
import tqdm
import torch
import numpy as np

from utils.data.DummyIMDataset import DummyIMDataset
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os.path as osp

from utils.data.LargeDataset import LargeDataset


class DatasetVisualizer:
    """
    This visualizer is used for the processed InMemoryDataset
    """
    def __init__(self, ds_name=None, data_root="../../dataProviders/data", common_ele=None, locator=None):
        self.ds_name = ds_name
        self.data_root = data_root
        self.common_ele = common_ele
        self.locator = locator

        self._ds_name_base = None
        self._ds = None
        self._n_heavy_list = None
        self._source = None
        self._sample_id = None

    def bond_distribution(self, cutoff, n_select=None, edge_name="BN_edge_index"):
        bins = np.arange(0, cutoff, 0.05)
        hist, bin_edges = None, None
        if n_select is not None:
            selected_i = np.random.permutation(len(self.ds))[:n_select]
        else: 
            selected_i = np.arange(len(self.ds))
        for i in tqdm.tqdm(selected_i, total=len(selected_i)):
            try:
                this_d = self.ds[i]
            except FileNotFoundError:
                continue
            r1 = this_d.R[getattr(this_d, edge_name)[0, :], :]
            r2 = this_d.R[getattr(this_d, edge_name)[1, :], :]
            dist = torch.sqrt(((r1 - r2) ** 2).sum(dim=-1))
            this_hist, this_bin_edges = np.histogram(dist, bins=bins)
            if hist is None:
                hist = this_hist
            else:
                hist += this_hist
            if bin_edges is None:
                bin_edges = this_bin_edges

        plt.figure()
        ax = sns.histplot(x=bins[:-1]+0.025, weights=hist, color="#7995c4", bins=bins)
        ax.set(xlabel="Distance (A)")
        plt.title(f"Distribution on {self.ds_name_base}")
        plt.tight_layout()
        extra = ""
        if n_select is not None:
            extra += f"_select{n_select}"
        if edge_name != "BN_edge_index":
            extra += f"_edge-{edge_name.split('_edge_index')[0]}"
        plt.savefig(osp.join("figures", f"{self.ds_name_base}{extra}_bond_dist.png"))
        plt.close()

    def save_n_heavy_dist(self, save_root=None, xlabel=None):
        xlabel = "Number of Heavy Atoms" if xlabel is None else xlabel
        info = {xlabel: self.n_heavy_list, "Source": self.source}
        info = pd.DataFrame(info)
        if max(self.n_heavy_list) < 100:
            bins = list(range(max(self.n_heavy_list) + 1))
        else:
            bins = 50
        sns.histplot(data=info, bins=bins, x=xlabel, hue="Source", multiple="stack")
        if save_root is None:
            save_root = self.data_root
        plt.savefig(osp.join(save_root, "figures", f"{self.ds_name_base}_dist.png"))
        plt.close()

    def save_element_dict(self, save_root=None):
        if save_root is None:
            save_root = self.data_root
        element_counter = Counter()
        rare_dfs = []
        for d in tqdm.tqdm(self.ds):
            elements = set(self.d2elements(d).numpy().tolist())
            element_counter.update(elements)
            if self.common_ele is not None:
                diff = elements.difference(self.common_ele)
                if len(diff) > 0:
                    # TODO: need to change
                    this_df = pd.DataFrame({"protein_file": osp.basename(d.protein_file[0]),
                                            "ligand_file": osp.basename(d.ligand_file[0]), 
                                            "rare_elements": str(diff)}, index=[0])
                    rare_dfs.append(this_df)
        data = {"atomic_num": [], "Atom Symbol": [], "#Occurence": []}
        for key in element_counter.keys():
            data["atomic_num"].append(key)
            data["#Occurence"].append(element_counter[key])
            atom_sym = ase.Atom(key).symbol
            data["Atom Symbol"].append(atom_sym)
        data = pd.DataFrame(data).sort_values(by="atomic_num", axis=0)
        sns.barplot(data=data, x="Atom Symbol", y="#Occurence", color="salmon", saturation=.5, log=True, edgecolor=".2")
        plt.savefig(osp.join(save_root, "figures", f"{self.ds_name_base}_element.png"))
        plt.close()

        if len(rare_dfs) > 0:
            rare_df = pd.concat(rare_dfs)
            rare_df.to_csv(osp.join(save_root, "figures", f"{self.ds_name_base}_rare.csv"))

    def d2elements(self, d):
        return d.Z

    @property
    def ds_name_base(self):
        if self._ds_name_base is None:
            if self.ds_name is not None:
                result = ".".join(osp.basename(self.ds_name).split(".")[:-1])
            else:
                result = ".".join(osp.basename(self.locator).split(".")[:-2])
            self._ds_name_base = result
        return self._ds_name_base

    @property
    def source(self):
        if self._source is None:
            if self.ds_name == "frag20-ultimate-sol-mmff-04182022.pyg":
                self._source = self.ds.data.dataset_name
            else:
                assert self.ds_name == "freesolv_openchem_aux_05022022.pyg"
                result = []
                for i in range(self.ds.data.mask.shape[0]):
                    this_mask = self.ds.data.mask[i, :]
                    if this_mask.sum() == 6:
                        result.append("Both")
                    elif this_mask[3]:
                        result.append("FreeSolv")
                    else:
                        assert this_mask[5]
                        result.append("Openchem-logP")
                self._source = result
        return self._source

    @property
    def sample_id(self):
        if self._sample_id is None:
            self._sample_id = self.ds.data.sample_id
        return self._sample_id

    @property
    def n_heavy_list(self):
        if self._n_heavy_list is None:
            result = []
            for data in tqdm.tqdm(self.ds):
                z = self.d2elements(data)
                this_n_heavy = (z != 1).sum().item()
                result.append(this_n_heavy)
            self._n_heavy_list = result
        return self._n_heavy_list

    @property
    def ds(self):
        if self._ds is None:
            if self.ds_name is not None:
                self._ds = DummyIMDataset(data_root=self.data_root, dataset_name=self.ds_name, cfg=None)
            else:
                self._ds = LargeDataset(data_root=self.data_root, file_locator=self.locator)
        return self._ds


class PLDatasetVisualizer(DatasetVisualizer):
    def __init__(self, ds_name, mol_type, **kwargs):
        super().__init__(ds_name, **kwargs)
        self.mol_type = mol_type
    
    @property
    def source(self):
        return [self.mol_type] * len(self.n_heavy_list)

    def d2elements(self, d):
        mol_id = 0 if self.mol_type == "protein" else 1
        return d.Z[d.mol_type==mol_id]

    @property
    def ds_name_base(self):
        return super().ds_name_base + f"_{self.mol_type}"


def statistics():
    d_name = "PDBbind_v2020_dry_09022022.pyg"
    visualizer = PLDatasetVisualizer(d_name, "protein",
         common_ele=set([1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]))
    visualizer.save_n_heavy_dist()
    visualizer.save_element_dict()


def statistics1():
    loc = "PL_train-polarH_10$10$6.loc.pth"
    vis = DatasetVisualizer(data_root="/scratch/sx801/data/", locator=loc)
    vis.bond_distribution(10, edge_name="PL_edge_index")
    vis.bond_distribution(10, edge_name="LIGAND_edge_index")

def statistics2():
    vis = PLDatasetVisualizer(data_root="/scratch/sx801/data/im_datasets/", ds_name="casf-scoring-dry-prot.polar-lig.polar.pyg", mol_type="protein")
    vis.save_element_dict("/scratch/sx801/scripts/physnet-dimenet1/MartiniDock/scripts")
    


if __name__ == '__main__':
    statistics2()
