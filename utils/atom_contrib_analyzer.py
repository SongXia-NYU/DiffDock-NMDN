import glob
import os
import os.path as osp
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from torch_geometric.loader import DataLoader

from utils.eval.trained_folder import TrainedFolder, ds_from_args, read_folder_config
from utils.utils_functions import get_device

kcal2ev = 1 / 23.06035
# R in kcal/(mol.K)
R = 1.98720425864083e-3
logP_to_watOct = 2.302585093 * R * 298.15


class AtomContribAnalyzer(TrainedFolder):
    def __init__(self, folder_name, sdf_folder, task="water_sol"):
        super().__init__(folder_name)

        self.task = task
        self.sdf_folder = sdf_folder
        self._dataloader_test = None
        self._intersection_df = None
        self._freesolv_ref_df = None

        if task == "water_sol":
            self.activity_name = "CalcSol(kcal/mol)"
            self.smiles_name = "cano_smiles"
            self.sample_id_name = "index_in_ds"
            self.unit = "kcal/mol"
        else:
            self.activity_name = "CalcLogP"
            self.smiles_name = "gas_smiles"
            self.sample_id_name = "sample_id"
            self.unit = "logP unit"

    def run(self):
        try:
            self._run_atom_contrib()
        except AssertionError as e:
            print(e)
        self._draw_atom_contrib("CalcSol(kcal/mol)", "CalcSol")
        self._draw_atom_contrib("CalcLogP", "CalcLogP")

    def _draw_atom_contrib(self, activity_name, save_name):
        import rdkit
        from rdkit.Chem.Draw import rdMolDraw2D, MolsToGridImage
        from rdkit.Chem.AllChem import RemoveAllHs, MolToPDBFile
        mols = []
        legends = []
        folders = glob.glob(osp.join(self.save_root, "mol_results", "*"))
        folders.sort()
        for mol_root in folders:
            # read sdf as rdkit mol object
            sample_id = osp.basename(mol_root)
            with rdkit.Chem.SDMolSupplier(osp.join(mol_root, f"{sample_id}.mmff.sdf"), removeHs=False) as suppl:
                mol = suppl[0]

            # annotate atoms with atom contribution
            contrib_df = pd.read_csv(osp.join(mol_root, "pred_info.csv"))
            h_contrib = []
            for i in range(contrib_df.shape[0]):
                this_contrib = contrib_df.iloc[i][activity_name]
                atom = mol.GetAtomWithIdx(i)
                atom.SetProp("atomNote", "{:.2f}".format(this_contrib))
                if atom.GetAtomicNum() == 1:
                    h_contrib.append(this_contrib)

            # load extra info
            mol_prop_df = pd.read_csv(osp.join(mol_root, "mol_prop.csv"))
            activity = mol_prop_df[activity_name].item()
            pred = np.sum(contrib_df[activity_name].values).item()

            # draw molecules as 2D
            mol = RemoveAllHs(mol)
            h_contrib = np.asarray(h_contrib)
            h_mean = np.mean(h_contrib)
            h_std = np.std(h_contrib)
            legend = f"sample_id: {sample_id} \n" + \
                     "Hydrogen Contribution: {:.2f} +- {:.2f} \n".format(h_mean, h_std) + \
                     "Predicted: {:.2f} {} \n".format(pred, self.unit) + \
                     "Experimental: {:.2f} {}".format(activity, self.unit)
            d = rdMolDraw2D.MolDraw2DCairo(1000, 800)  # or MolDraw2DSVG to get SVGs
            d.drawOptions().legendFontSize = 40  # It did not work lol
            d.DrawMolecule(mol, legend=legend)
            d.FinishDrawing()
            d.WriteDrawingText(osp.join(mol_root, f"atom_contrib.{save_name}.png"))

            # update mols list
            mols.append(mol)
            legends.append(sample_id)

            # save as PDB file
            pdb_file = osp.join(mol_root, f"{sample_id}.pdb")
            MolToPDBFile(mol, pdb_file)
            # infuse atom contribution as B factor for better visualization
            # Use the command in Chimera
            # label all atoms attribute name; color bfactor atom palette -4,red:0,white:+1,blue transparency 0 key true;
            # key fontSize 60
            from prody import parsePDB, writePDB
            atoms = parsePDB(pdb_file)
            atoms._data["beta"] = contrib_df[activity_name].values
            writePDB(osp.join(mol_root, f"{sample_id}.{save_name}.contrib_infuse.pdb"), atoms)

        if len(mols) > 100:
            mols = mols[:100]
        img = MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 300), legends=legends)
        img.save(osp.join(self.save_root, f"al(most)_tested_mols.{save_name}.png"))

    def _run_atom_contrib(self):
        ds_tqdm = tqdm.tqdm(self.dl_test)
        for batch in ds_tqdm:
            batch = batch.to(get_device())
            i = getattr(batch, self.sample_id_name)[0].item()
            if hasattr(batch, self.smiles_name):
                smiles = getattr(batch, self.smiles_name)[0]
            else:
                smiles = None
            mol_root = osp.join(self.save_root, "mol_results", f"{i}")
            os.makedirs(mol_root, exist_ok=True)
            if hasattr(batch, "activity"):
                mol_prop = {"smiles": smiles, "CalcSol(kcal/mol)": self.freesolv_ref_df.loc[i]["activity"]}
                if i in self.intersection_df.index:
                    mol_prop["CalcLogP"] = self.intersection_df.loc[i]["activity_2"]
                else:
                    mol_prop["CalcLogP"] = 9999
            else:
                mol_prop = {"smiles": smiles, "CalcSol(kcal/mol)": batch.CalcSol.item(),
                            "CalcLogP": batch.CalcSol.item() / logP_to_watOct}
            mol_prop_df = pd.DataFrame(mol_prop, index=[0])
            mol_prop_df.to_csv(osp.join(mol_root, "mol_prop.csv"), index=False)

            model_out = self.predict(batch)
            atom_prop = model_out["atom_prop"].detach().cpu().numpy()
            atomic_number = batch.Z.cpu().numpy().tolist()
            pred_info = pd.DataFrame({"atomic_number": atomic_number, "gasEnergy(eV)": atom_prop[:, 0],
                                      "watEnergy(eV)": atom_prop[:, 1], "octEnergy(eV)": atom_prop[:, 2]})
            pred_info["CalcSol(kcal/mol)"] = (pred_info["watEnergy(eV)"] - pred_info["gasEnergy(eV)"]) / kcal2ev
            pred_info["CalcLogP"] = ((pred_info["watEnergy(eV)"] - pred_info["octEnergy(eV)"])/kcal2ev)/logP_to_watOct

            sdf_f = glob.glob(osp.join(self.sdf_folder, f"{i}.*sdf"))
            assert len(sdf_f) == 1, f"File not found or multiple file found: {sdf_f}"
            sdf_f = sdf_f[0]
            shutil.copy(sdf_f, mol_root)

            pred_info.to_csv(osp.join(mol_root, "pred_info.csv"), index=False)

    def predict(self, batch):
        return self.model(batch)

    @property
    def intersection_df(self):
        if self._intersection_df is None:
            root = "/home/carrot_of_rivia/Documents/PycharmProjects/Mol3DGenerator/data/freesolv_openchem"
            self._intersection_df = pd.read_csv(osp.join(root, "intersection.csv")).set_index("sample_id_1")
        return self._intersection_df

    @property
    def freesolv_ref_df(self):
        if self._freesolv_ref_df is None:
            root = "/home/carrot_of_rivia/Documents/PycharmProjects/Mol3DGenerator/data/freesolv_sol"
            self._freesolv_ref_df = pd.read_csv(osp.join(root, "freesolv_paper_fl.csv")).set_index("sample_id")
        return self._freesolv_ref_df

    @property
    def save_root(self):
        if self._test_dir is None:
            test_prefix = self.cfg["folder_prefix"] + '_atom_contrib_'
            current_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            tmp = test_prefix + current_time
            self._test_dir = osp.join(self.folder_name, osp.basename(tmp))
            os.mkdir(self._test_dir)
        return self._test_dir

    @property
    def ds(self):
        if self._data_provider is None:
            _data_provider = ds_from_args(self.args_raw, rm_keys=False)
            setattr(_data_provider.data, "index_in_ds", torch.arange(len(_data_provider)))
            _data_provider.slices["index_in_ds"] = _data_provider.slices["N"]
            self._data_provider = _data_provider
            self._data_provider_test = _data_provider[_data_provider.test_index]
        return self._data_provider

    @property
    def dl_test(self):
        if self._dataloader_test is None:
            self._dataloader_test = DataLoader(self.ds_test, batch_size=1)
        return self._dataloader_test


class AtomContribAnalyzerEns(AtomContribAnalyzer):
    def __init__(self, folder_name, sdf_folder, task="water_sol", config_folder=None):
        super().__init__(folder_name, sdf_folder, task)

        self.config_folder = config_folder
        self.ens_folders = glob.glob(osp.join(folder_name, "exp_*_cycle_-1_run_*"))
        self.ens_folders.sort()

        self._model_list = None

    @property
    def model_list(self):
        if self._model_list is None:
            tmp_folders = [TrainedFolder(f) for f in self.ens_folders]
            model_list = [f.model for f in tmp_folders]
            self._model_list = model_list
        return self._model_list

    def predict(self, batch):
        model_out_list = None
        for model in self.model_list:
            model_out = model(batch)
            if model_out_list is None:
                model_out_list = {
                    key: [model_out[key]] for key in model_out.keys()
                }
            else:
                for key in model_out:
                    model_out_list[key].append(model_out[key])
        ens_out = {key: sum(model_out_list[key]) / len(model_out_list[key]) for key in model_out_list}
        return ens_out

    @property
    def args_raw(self):
        if self._args_raw is None:
            if self.config_folder is not None:
                _args_raw, _config_name = read_folder_config(self.config_folder)
            else:
                _args_raw, _config_name = read_folder_config(self.ens_folders[0])
            self._args_raw = _args_raw
            self._config_name = _config_name
        return self._args_raw

    @property
    def ds(self):
        if self._data_provider is None:
            _data_provider = ds_from_args(self.args_raw, rm_keys=False)
            setattr(_data_provider.data, "index_in_ds", torch.arange(len(_data_provider)))
            _data_provider.slices["index_in_ds"] = _data_provider.slices["N"]
            self._data_provider = _data_provider
            self._data_provider_test = _data_provider
        return self._data_provider

