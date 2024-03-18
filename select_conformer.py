import argparse
import copy
import logging
import os
import os.path as osp
import warnings

import pandas as pd
import rdkit.Chem
import torch
import torch_geometric
import tqdm

from glob import glob
from utils.DataPrepareUtils import my_pre_transform
from utils.data.DummyIMDataset import DummyIMDataset
from utils.train.trainer import remove_extra_keys
from utils.eval.trained_folder import read_folder_config
from utils.utils_functions import args2loss_fn, get_device, init_model_test, remove_handler
from torch_geometric.loader import DataLoader


class ConformerSelector:
    def __init__(self, save_folder, pretrained_model, conf_sdf=None, mol_list=None, lowest_i=1, silent=False, pyg_name=None, **kwargs):
        self.silent = silent
        self.PRETRAINED_MODEL = pretrained_model
        self.COMPARE_ID = 0  # 0 for gas energy, 1 for water energy, 2 for octanol energy

        self._mol_list = mol_list
        self.save_folder = save_folder
        self.lowest_i = lowest_i

        if pyg_name is not None:
            assert conf_sdf is None
            self.explicit_pyg = True
        else:
            assert conf_sdf is not None
            self.explicit_pyg = False

        self.conf_sdf = conf_sdf
        self.pyg_name = pyg_name

        os.makedirs(self.save_folder, exist_ok=True)

        if not silent:
            log_tmp = logging.getLogger()
            remove_handler(log_tmp)
            logging.basicConfig(filename=osp.join(save_folder, "select.log"), format='%(asctime)s %(message)s', filemode='w')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        if not self.explicit_pyg:
            self.logger.info(f"Total number of conformers: {len(self.mol_list)}")

        self._dataset = None
        self._data_loader = None
        self._model = None
        self._loss_fn = None
        self._model_config = None
        self._extra = None

    def run(self):
        pred_results = self.pred_results()
        torch.save(pred_results, osp.join(self.save_folder, "test_result.pth"))

        predicted_energy = pred_results[:, self.COMPARE_ID].view(-1)
        sort_id = torch.argsort(predicted_energy, descending=False)
        if sort_id.shape[0] > self.lowest_i:
            best_ids = sort_id[:self.lowest_i]
        else:
            best_ids = sort_id

        self.logger.info(f"The {best_ids.numpy().tolist()}th molecule(s) have the lowest predicted energy (eV):"
                         f" {predicted_energy[best_ids].numpy().tolist()}")
        self.logger.info(f"Highest conformer energy(eV): {predicted_energy[sort_id[-1]].item()}")

        selection_info = {"file_name": osp.basename(self.conf_sdf)}
        for i, idx in enumerate(best_ids):
            this_mol = self.mol_list[idx]
            if not self.silent:
                writer = rdkit.Chem.SDWriter(osp.join(self.save_folder, f"lowest_{i}.sdf"))
                writer.write(this_mol)
            selection_info[f"cluster_no(lowest_{i})"] = this_mol.GetProp("cluster_no")
            selection_info[f"initial_conformation_id(lowest_{i})"] = this_mol.GetProp("initial_conformation_id")
        return pd.DataFrame(selection_info, index=[0])

    def get_pred_df(self):
        pred_results = self.pred_results()
        pred_info = copy.deepcopy(self.extra)
        for i, name in enumerate(["gasEnergy(eV)", "watEnergy(eV)", "octEnergy(eV)"]):
            pred_info[name] = pred_results[:, i].numpy()
        pred_df = pd.DataFrame(pred_info)
        return pred_df

    def pred_results(self):
        pred_results = []
        for val_data in self.data_loader:
            val_data = val_data.to(get_device())
            model_out = self.model(val_data)["mol_prop"].detach().cpu()
            pred_results.append(model_out)
        pred_results = torch.concat(pred_results, dim=0)
        return pred_results

    def pre_load_model(self, model):
        assert self._model is None
        self._model = model

    @property
    def model_config(self):
        if self._model_config is None:
            self._model_config = read_folder_config(self.PRETRAINED_MODEL)[0]
        return self._model_config

    @property
    def loss_fn(self):
        warnings.warn("I will remove loss_fn!")
        if self._loss_fn is None:
            config_dict = self.model_config
            self._loss_fn = args2loss_fn(config_dict)
        return self._loss_fn

    @property
    def model(self):
        if self._model is None:
            model_data = torch.load(osp.join(self.PRETRAINED_MODEL, 'best_model.pt'), map_location=get_device())
            net = init_model_test(self.model_config, model_data)
            self._model = net
        return self._model

    @property
    def data_loader(self):
        if self._data_loader is None:
            import torch.utils.data
            self._data_loader = DataLoader(self.pyg_dataset, batch_size=16, pin_memory=True, num_workers=0)
        return self._data_loader

    @property
    def pyg_dataset(self):
        if self._dataset is None:
            if self.explicit_pyg:
                _dataset = DummyIMDataset(root="../dataProviders/data", dataset_name=self.pyg_name)
                _dataset, _extra = remove_extra_keys(_dataset, return_extra=True)
                self._dataset = _dataset
                self._extra = _extra
            else:
                if not osp.exists(osp.join(self.save_folder, "data", "processed", "dataset.pyg.pth")):
                    raw_dataset = self.raw_dataset_from_mol_list(self.mol_list)
                    os.makedirs(osp.join(self.save_folder, "data", "processed"), exist_ok=True)
                    torch.save(raw_dataset, osp.join(self.save_folder, "data", "processed", "dataset.pyg.pth"))
                self._dataset = DummyIMDataset(root=osp.join(self.save_folder, "data"), dataset_name="dataset.pyg.pth")
        return self._dataset

    @property
    def extra(self):
        if self._extra is None:
            # it is inited in self.pyg_dataset
            __ = self.pyg_dataset
        return self._extra

    @property
    def mol_list(self):
        if self._mol_list is None:
            assert self.conf_sdf is not None
            with rdkit.Chem.SDMolSupplier(self.conf_sdf, removeHs=False) as suppl:
                self._mol_list = [mol for mol in suppl]
        return self._mol_list

    @staticmethod
    def raw_dataset_from_mol_list(mol_list):
        data_list = []
        for mol in mol_list:
            coordinate = mol.GetConformer().GetPositions()
            coordinate = torch.as_tensor(coordinate)
            elements = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            elements = torch.as_tensor(elements).long()
            info_dict = {
                "R": coordinate,
                "Z": elements,
                "N": torch.as_tensor([len(elements)]).long()
            }
            from torch_geometric.data import Data
            this_data = Data(**info_dict)

            this_data = my_pre_transform(this_data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                         cutoff=10.0, boundary_factor=100., use_center=True, mol=None,
                                         cal_3body_term=False, bond_atom_sep=False, record_long_range=True)

            data_list.append(this_data)

        data_concat = torch_geometric.data.InMemoryDataset.collate(data_list)
        return data_concat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformers", default=None)
    parser.add_argument("--pyg_name", default=None)
    parser.add_argument("--save_folder", default="test")
    parser.add_argument("--lowest_i", default=1, type=int)
    parser.add_argument("--tqdm", action="store_true")
    parser.add_argument("--pretrained_model")
    args = parser.parse_args()
    args = vars(args)

    if args["conformers"] is not None:
        warnings.warn("Use pyg_name for faster calculation and better GPU util.")
        assert args["pyg_name"], f"Only one is None: {args['pyg_name']}"

        model = None
        confs = glob(args["conformers"])
        if args["tqdm"]:
            confs = tqdm.tqdm(confs)
        select_summary = []
        for conf_sdf in confs:
            try:
                save_folder = osp.join(args["save_folder"], osp.basename(conf_sdf))
                selector = ConformerSelector(save_folder=save_folder, conf_sdf=conf_sdf, lowest_i=args["lowest_i"],
                                             silent=True)
                # only load model once
                if model is None:
                    model = selector.model
                else:
                    selector.pre_load_model(model)
                select_summary.append(selector.run())
            except Exception as e:
                print(f"Error processing: {conf_sdf}: ", e)
        select_summary = pd.concat(select_summary)
        select_summary.to_csv(osp.join(args["save_folder"], "summary.csv"), index=False)
    else:
        assert args["pyg_name"] is not None
        pygs = glob(args["pyg_name"])
        if args["tqdm"]:
            pygs = tqdm.tqdm(pygs)

        modified_args = copy.deepcopy(args)
        all_df = []
        model = None
        for pyg in pygs:
            modified_args["pyg_name"] = pyg
            selector = ConformerSelector(**modified_args)

            # only load model once
            if model is None:
                model = selector.model
            else:
                selector.pre_load_model(model)

            try:
                df = selector.get_pred_df()
                all_df.append(df)
            except Exception as e:
                print(f"Error processing {pyg}: {e}")
        all_df = pd.concat(all_df, axis=0)
        all_df.to_csv(osp.join(args["save_folder"], "dd_nmr_all_sPhysNet.csv"), index=False)


if __name__ == '__main__':
    main()
