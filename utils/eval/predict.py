from utils.utils_functions import torchdrug_imports
torchdrug_imports()

import argparse
from collections import defaultdict
from typing import List
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from glob import glob
import numpy as np
import os
import os.path as osp
import tqdm

from utils.DataPrepareUtils import my_pre_transform
from utils.data.MolFileDataset import MolFromPLDataset, SDFDataset, StackedSDFileDataset
from utils.eval.trained_folder import TrainedFolder
from utils.utils_functions import get_device, kcal2ev, lazy_property
from utils.LossFn import logP_to_watOct

from geometry_processors.pl_dataset.pdb2020_ds_reader import PDB2020DSReader

display_unit_mapper = {
    "E_gas": "eV", "E_water": "eV", "E_oct": "eV", "E_hydration": "kcal/mol", "LogP": ""
}

class SinglePredictor(TrainedFolder):
    def __init__(self, trained_model_folder, sdf_files: List[str] = None, 
                 pyg_name: str = None, pyg_root: str = "/scratch/sx801/data/im_datasets/",
                 init_model = None):
        super().__init__(trained_model_folder, None)
        self.sdf_files = sdf_files
        self.pyg_name = pyg_name
        self.pyg_root = pyg_root

        self._data_loader = None
        self._model = init_model
        self.model.module.requires_atom_prop = True

    def predict(self, data: Data, to_numpy: bool = False) -> dict:
        self.model.eval()
        model_out = self.model(data.to(get_device()))
        if to_numpy:
            for key in model_out: model_out[key] = model_out[key].detach().cpu().numpy()
        model_out["data_batch"] = data
        return model_out
    
    @torch.no_grad()
    def predict_nograd(self, data: Data, to_numpy: bool = False) -> dict:
        return self.predict(data, to_numpy)

    @lazy_property
    def ds(self):
        """
        preprocessing into pytorch data
        """
        if self.pyg_name is not None:
            return MolFromPLDataset(data_root=self.pyg_root, dataset_name=self.pyg_name)
        
        assert self.sdf_files is not None
        if "/pose_diffdock/" in self.sdf_files[0] or "fep-benchmark" in self.sdf_files[0]:
            return StackedSDFileDataset(self.sdf_files)

        return SDFDataset(self.sdf_files)
    
    @lazy_property
    def data_loader(self):
        return DataLoader(self.ds, batch_size=1024, shuffle=False, num_workers=0)

    def iter_predictions(self):
        for this_d in self.data_loader:
            model_out = self.predict_nograd(this_d, True)
            yield model_out

class EnsPredictor:
    def __init__(self, trained_model_folders: str, sdf_files: List[str] = None, 
                 pyg_name: str = None, pyg_root: str = "/scratch/sx801/data/im_datasets/",
                 init_models=None) -> None:
        self.trained_folders = glob(trained_model_folders)
        self.sdf_files = sdf_files
        self.pyg_name = pyg_name

        if init_models is None: init_models = [None] * len(self.trained_folders)
        self.single_predictors = [SinglePredictor(f, sdf_files, pyg_name, pyg_root, init_model) 
                                  for f, init_model in zip(self.trained_folders, init_models)]
        # inject data set to avoid re-loading data set
        # for single_predictor in self.single_predictors[1:]:
        #     single_predictor._lazy__ds = self.single_predictors[0].ds

    def predict(self, data: Data, to_numpy: bool = False) -> dict:
        model_out =defaultdict(lambda: 0.)
        for predictor in self.single_predictors:
            pred_dict: dict = predictor.predict(data, to_numpy)
            for key in pred_dict.keys():
                if not isinstance(pred_dict[key], torch.Tensor):
                    model_out[key] = pred_dict[key]
                    continue
                model_out[key] = pred_dict[key] + model_out[key]
        for key in model_out.keys():
            if not isinstance(model_out[key], torch.Tensor): continue
            model_out[key] = model_out[key] / len(self.single_predictors)
        return model_out
    
    @torch.no_grad()
    def predict_nograd(self, data: Data, to_numpy: bool = False) -> dict:
        return self.predict(data, to_numpy)
    
    def iter_predictions(self):
        # avoid recalculation of the data sets
        # for predictor in self.single_predictors[1:]:
        #     predictor._lazy__data_provider = self.single_predictors[0].ds

        for predictions in zip(*(predictor.iter_predictions() for predictor in self.single_predictors)):
            if len(predictions) == 0:
                raise ValueError("Trained model not successfully downloaded. Please make sure 'bash bash_scripts/download_models_and_extract.bash' has finished successfully.")

            ens_prediciton = defaultdict(lambda: 0.)
            ens_std = defaultdict(lambda: [])
            for pred in predictions:
                for key in pred.keys():
                    if key == "data_batch":
                        continue
                    ens_prediciton[key] = ens_prediciton[key] + pred[key]
                    ens_std[key].append(pred[key][None, ...])
            for key in ens_prediciton.keys():
                ens_prediciton[key] /= len(predictions)
            for key in ens_std.keys():
                ens_std[key] = np.std(np.stack(ens_std[key], axis=0), axis=0)
            ens_prediciton["data_batch"] = predictions[-1]["data_batch"]
            yield ens_prediciton, ens_std

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", default="/scratch/sx801/scripts/sPhysNet-MT/data/aspirin.mmff.sdf")
    parser.add_argument("--pyg_name", default=None)
    parser.add_argument("--model", help="cal_single | cal_ens5 | exp_ens5", default="exp_ens5")
    args = parser.parse_args()

    if args.model == "cal_single":
        predictor = SinglePredictor("./pretrained/exp_frag20sol_012_run_2022-04-20_143627", [args.sdf], args.pyg_name)
        for prediction in predictor.iter_predictions():
            print(prediction)
    elif args.model == "cal_ens5":
        predictor = EnsPredictor("./pretrained/exp_frag20sol_012_active_ALL_2022-05-01_112820/exp_*_cycle_-1_*", [args.sdf], args.pyg_name)
        predictor.display_prediction(["E_gas", "E_water", "E_oct"])
    elif args.model == "exp_ens5":
        predictor = EnsPredictor("./pretrained/exp_ultimate_freeSolv_13_active_ALL_2022-05-20_100309/exp_*_cycle_-1_*", [args.sdf], args.pyg_name)
        for ens, std in predictor.iter_predictions():
            mol_prop = ens["mol_prop"]
            e_gas = mol_prop[:, 0]
            e_water = mol_prop[:, 1]
            e_oct = mol_prop[:, 2]

            e_hydration = (e_water - e_gas) / kcal2ev
            e_wat_oct = (e_water - e_oct) / kcal2ev
            logp = e_wat_oct / logP_to_watOct
            print("Hydration: ", e_hydration)
            print("logP", logp)
    else:
        raise ValueError("Model must be one of the following: cal_single | cal_ens5 | exp_ens5")

def save_pred_pyg():
    # model_path = "./pretrained/exp_frag20sol_012_active_ALL_2022-05-01_112820/exp_*_cycle_-1_*"
    model_path = "./pretrained/exp_frag20sol_012_active_ALL_2022-05-01_112820/exp_frag20sol*"
    pyg_name = "PBind2020OG.polar.polar.implicit.min_dist.linf9.pyg"
    save_name = "PDBind_v2020OG_LinF9_ligatomprop_frag20sol_012_ens.pyg"

    predictor = EnsPredictor(model_path, None, pyg_name)
    save_pyg_path = osp.join("/scratch/sx801/data/im_datasets/processed", save_name)

    data_list = []
    for prediction in tqdm.tqdm(predictor.iter_predictions(), 
                                total=len(predictor.single_predictors[0].data_loader)):
        pred_dict = prediction[0]
        atom_prop = pred_dict["atom_prop"]
        mol_prop = pred_dict["mol_prop"]
        atom_mol_batch = pred_dict["data_batch"].atom_mol_batch.cpu().numpy()
        pdb_list = [osp.basename(f[0]).split(".")[0] for f in pred_dict["data_batch"].protein_file]
        for i, pdb in enumerate(pdb_list):
            this_mask = (atom_mol_batch == i)
            this_atom_prop = atom_prop[this_mask, :]
            this_mol_prop = mol_prop[i, :]
            ligand_file = pred_dict["data_batch"].ligand_file[i]

            this_data = Data(atom_prop=torch.as_tensor(this_atom_prop), pdb=pdb, 
                             ligand_file=ligand_file, mol_prop=this_mol_prop)
            data_list.append(this_data)
    torch.save(InMemoryDataset.collate(data_list), save_pyg_path)

def save_pred_pyg_allh():
    # model_path = "./pretrained/exp_frag20sol_012_active_ALL_2022-05-01_112820/exp_*_cycle_-1_*"
    model_path = "./pretrained/exp_frag20sol_012_active_ALL_2022-05-01_112820/exp_frag20sol*"
    save_name = "PDBind_v2020OG_LinF9_ligatomprop_frag20sol_012_ens.allh.pyg"
    src_ds_pdbs = torch.load("/scratch/sx801/data/im_datasets/processed/PBind2020OG.polar.polar.implicit.min_dist.linf9.pyg")[0].pdb
    reader = PDB2020DSReader("/PDBBind2020_OG")
    allh_lig_files = [reader.pdb2neutral_lig(pdb) for pdb in src_ds_pdbs]

    predictor = EnsPredictor(model_path, allh_lig_files, None)
    save_pyg_path = osp.join("/scratch/sx801/data/im_datasets/processed", save_name)

    data_list = []
    for pred_dict, __ in tqdm.tqdm(predictor.iter_predictions(), 
                                total=len(predictor.single_predictors[0].data_loader)):
        atom_prop = pred_dict["atom_prop"]
        mol_prop = pred_dict["mol_prop"]
        atom_mol_batch = pred_dict["data_batch"].atom_mol_batch.cpu().numpy()
        pdb_list = [osp.basename(f[0]).split(".")[0].split("_")[0] for f in pred_dict["data_batch"].mol_file]
        for i, pdb in enumerate(pdb_list):
            this_mask = (atom_mol_batch == i)
            this_atom_prop = atom_prop[this_mask, :]
            this_mol_prop = mol_prop[i, :]
            ligand_file = pred_dict["data_batch"].mol_file[i]

            this_data = Data(atom_prop=torch.as_tensor(this_atom_prop), pdb=pdb,
                             ligand_file=ligand_file, mol_prop=torch.as_tensor(this_mol_prop))
            data_list.append(this_data)
    torch.save(InMemoryDataset.collate(data_list), save_pyg_path)

def save_pred_casf_docking():
    model_path = "./pretrained/exp_frag20sol_012_active_ALL_2022-05-01_112820/exp_frag20sol*"
    save_name = "CASF2016_docking_ligmolprop_frag20sol_012_ens.allh.pyg"
    src_pyg = "/scratch/sx801/data/im_datasets/processed/casf-docking.allh.allh.implicit.min_dist.pyg"
    predictor = EnsPredictor(model_path, None, src_pyg)
    save_pyg_path = osp.join("/scratch/sx801/data/im_datasets/processed", save_name)

    ds = predictor2im_ds(predictor)
    torch.save(ds, save_pyg_path)

def save_pred_casf_screening():
    model_path = "./pretrained/exp_frag20sol_012_active_ALL_2022-05-01_112820/exp_frag20sol*"
    save_name = "CASF2016_screening_ligmolprop_frag20sol_012_ens.allh"
    save_dir = osp.join("/scratch/sx801/data/im_datasets/processed", save_name)
    os.makedirs(save_dir, exist_ok=True)
    src_pygs = "/scratch/sx801/data/im_datasets/processed/casf-screening.allh.allh.implicit.min_dist/????.pyg"
    for src_pyg in glob(src_pygs):
        tgt_pdb = osp.basename(src_pyg).split(".")[0]
        save_pyg_path = osp.join(save_dir, f"{tgt_pdb}.pyg")
        if osp.exists(save_pyg_path): continue
        predictor = EnsPredictor(model_path, None, src_pyg)

        ds = predictor2im_ds(predictor, collate=True)
        torch.save(ds, save_pyg_path)

def predictor2im_ds(predictor: EnsPredictor, collate=True):
    data_list = []
    for pred_dict, __ in tqdm.tqdm(predictor.iter_predictions(), 
                                total=len(predictor.single_predictors[0].data_loader)):
        atom_prop = pred_dict["atom_prop"]
        mol_prop = pred_dict["mol_prop"]
        atom_mol_batch = pred_dict["data_batch"].atom_mol_batch.cpu().numpy()
        pdb_list = [osp.basename(f).split(".")[0].split("_")[0] for f in pred_dict["data_batch"].protein_file]
        for i, pdb in enumerate(pdb_list):
            this_mol_prop = mol_prop[i, :]
            ligand_file = pred_dict["data_batch"].ligand_file[i]

            this_data = Data(pdb=pdb, ligand_file=ligand_file, mol_prop=torch.as_tensor(this_mol_prop))
            data_list.append(this_data)
    if not collate:
        return data_list
    return InMemoryDataset.collate(data_list)

def compare_and_debug():
    # Error found: I did not average the ensemble predictions...
    raise ValueError
    runtime_vars = torch.load("temp.th")
    print(runtime_vars["pdb_list"])
    print(runtime_vars["phys_mol_prop"])
    from geometry_processors.pl_dataset.casf2016_reader import CASF2016Reader
    reader = CASF2016Reader("/CASF-2016-cyang")
    sdf_list = [reader.pdb2lig_core_sdf(pdb) for pdb in runtime_vars["pdb_list"]]
    model_path = "./pretrained/exp_frag20sol_012_active_ALL_2022-05-01_112820/exp_frag20sol*"
    predictor = EnsPredictor(model_path, sdf_list, None)
    for pred_dict, __ in tqdm.tqdm(predictor.iter_predictions(), 
                                total=len(predictor.single_predictors[0].data_loader)):
        atom_prop = pred_dict["atom_prop"]
        mol_prop = pred_dict["mol_prop"]
    
    print(runtime_vars["phys_mol_prop"][:, :3])
    print("*"*10)
    print(mol_prop)

if __name__ == "__main__":
    save_pred_casf_screening()
