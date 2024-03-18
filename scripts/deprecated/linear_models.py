import glob
import os
import os.path as osp
import time
from datetime import datetime
from math import ceil

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVR
from torch.optim import Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from Networks.SharedLayers.ActivationFns import activation_getter
from utils.Optimizers import EmaAmsGrad
from utils.utils_functions import get_lr

dataset = "lipop_embed160_exp339_qm_diff_dataset.pt"
csv = "lipop_sol.csv"
_tgt_name = "exp_dft_diff"
ss = False
coe = 23.06035


def svr_getter(param):
    return SVR(**param)


def rf_getter(param):
    return RandomForestRegressor(**param)


def xgb_getter(param):
    from xgboost import XGBRegressor
    return XGBRegressor(**param)


def linear_getter(param):
    return LinearRegression(**param)


class DNNLayer(nn.Module):
    def __init__(self, feature_dim, n_layers, act, init_dim=160, final_dim=1, dim_decay=False, bn=False, affine=True,
                 momentum=0.1):
        super().__init__()
        self.linear_layers = nn.ModuleList()
        if bn:
            self.linear_layers.append(nn.BatchNorm1d(init_dim, affine=affine, momentum=momentum))
        if n_layers >= 2:
            self.linear_layers.append(torch.nn.Linear(init_dim, feature_dim))
            if bn:
                self.linear_layers.append(nn.BatchNorm1d(feature_dim, affine=affine, momentum=momentum))
            last_dim = feature_dim
            for i in range(n_layers - 2):
                if dim_decay:
                    this_dim = ceil(last_dim / 2)
                    read_out_i = torch.nn.Linear(last_dim, this_dim)
                    last_dim = this_dim
                else:
                    read_out_i = torch.nn.Linear(last_dim, last_dim)
                    this_dim = last_dim
                self.linear_layers.append(read_out_i)
                if bn:
                    self.linear_layers.append(nn.BatchNorm1d(this_dim, affine=affine, momentum=momentum))
        else:
            last_dim = init_dim
        self.lin_last = nn.Linear(last_dim, final_dim)
        self.activation = activation_getter(act)

    def forward(self, data):
        out = data
        for layer in self.linear_layers:
            out = layer(out)
            if isinstance(layer, nn.Linear):
                out = self.activation(out)
        out = self.lin_last(out)
        return out,


class DNNModel:
    def __init__(self, data_ready, optimizer="EmaAmsGrad", chk=None, **model_kwargs):
        current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.run_dir = f"dnn-exp_run_{current_time}"
        os.makedirs(self.run_dir)
        self.meta_f = osp.join(self.run_dir, "meta.txt")
        self.log = osp.join(self.run_dir, "training.log")
        self.best_model_pt = osp.join(self.run_dir, "best_model.pt")
        self.loss_csv = osp.join(self.run_dir, "loss_data.csv")
        self.optimizer = optimizer
        self.train_model = DNNLayer(**model_kwargs).cuda().double()
        self.val_model = DNNLayer(**model_kwargs).cuda().double()
        self.val_model.load_state_dict(self.train_model.state_dict())

        if chk is not None:
            state_dict = torch.load(chk)
            self.val_model.load_state_dict(state_dict)
            self.train_model.load_state_dict(state_dict)

        self.mse_loss = torch.nn.MSELoss()
        for key in data_ready:
            data_ready[key] = torch.as_tensor(data_ready[key]).cuda().double()
        self.train_dataset = TensorDataset(data_ready["train_X"], data_ready["train_y"])
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.valid_dataset = TensorDataset(data_ready["valid_X"], data_ready["valid_y"])
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=32)
        self.test_dataset = TensorDataset(data_ready["test_X"], data_ready["test_y"])
        self.test_loader = DataLoader(self.test_dataset, batch_size=32)

        self_info = vars(self)
        for key in self_info.keys():
            self.log_into(self.meta_f, f"{key}: {self_info[key]}\n")
        self.log_into(self.meta_f, f"{model_kwargs=} \n")
        self.log_into(self.log, f"{model_kwargs=} \n")

    def fit(self):
        lr = 1e-1
        ema = 0.
        if self.optimizer == "EmaAmsGrad":
            optimizer = EmaAmsGrad(self.train_model.parameters(), self.val_model, lr=lr, ema=ema)
        else:
            optimizer = Adagrad(self.val_model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

        self.log_into(self.loss_csv, "Epoch,train_MSE,val_MSE,val_RMSE,test_RMSE,LR\n")

        best_loss = np.inf
        best_model = self.val_model

        for epoch in (range(1000)):
            train_loss = 0.
            n_total = 0
            self.train_model.train()
            self.val_model.train()
            for X, y in self.train_loader:
                optimizer.zero_grad()
                pred = self.predict_train(torch.as_tensor(X).cuda()).view(-1)
                loss = self.mse_loss(pred, torch.as_tensor(y).cuda())
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X.shape[0]
                n_total += X.shape[0]
            train_loss /= n_total
            val_loss = self.valid()
            test_loss = self.valid(split="test")
            scheduler.step(val_loss)
            self.log_into(self.loss_csv,
                          f"{epoch},{train_loss},{val_loss},{np.sqrt(val_loss)},{np.sqrt(test_loss)},{get_lr(optimizer)}\n")
            if best_loss > val_loss:
                best_model = self.val_model
                best_loss = val_loss
                torch.save(best_model.state_dict(), self.best_model_pt)

        for split in ["train", "valid", "test"]:
            final_loss = self.valid(split=split, model=best_model)
            final_rmse = np.sqrt(final_loss)
            self.log_into(self.log, f"{split} size: {len(getattr(self, f'{split}_dataset'))}\n")
            self.log_into(self.log, f"best model {split} MSE: {final_loss}\n")
            self.log_into(self.log, f"best model {split} RMSE: {final_rmse}\n")
            self.log_into(self.log, "----------------------------------\n")

    def valid(self, split="valid", model=None):
        loader = getattr(self, f"{split}_loader")
        self.train_model.eval()
        self.val_model.eval()
        with torch.no_grad():
            val_loss = 0.
            n_total = 0
            for X, y in loader:
                pred = self.predict(torch.as_tensor(X), model).view(-1)
                val_loss += self.mse_loss(pred, torch.as_tensor(y)).item() * X.shape[0]
                n_total += X.shape[0]
            val_loss /= n_total
        return val_loss

    def predict(self, X, model=None):
        if model is not None:
            return model(X)[0]
        else:
            return self.val_model(X)[0]

    def predict_train(self, X):
        if self.optimizer == "EmaAmsGrad":
            return self.train_model(X)[0]
        else:
            return self.val_model(X)[0]

    @staticmethod
    def log_into(f_in, msg):
        with open(f_in, "a") as f:
            f.write(msg)


class LinSRLayer:
    def __init__(self, data_ready):
        self.lin = torch.nn.Linear(160, 1, bias=True).cuda().double()
        self.lin_train = torch.nn.Linear(160, 1, bias=True).cuda().double()
        self.lin_train.load_state_dict(self.lin.state_dict())
        # self.lin.weight.data = torch.load("../exp325_run_2021-07-26_191533/best_model.pt")["main_module_list.2.output.lin.weight"]
        for key in data_ready:
            data_ready[key] = torch.as_tensor(data_ready[key]).cuda().double()
        self.train_dataset = TensorDataset(data_ready["train_X"], data_ready["train_y"])
        self.train_loader = DataLoader(self.train_dataset, batch_size=8, shuffle=True)
        self.valid_dataset = TensorDataset(data_ready["valid_X"], data_ready["valid_y"])
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=32)
        self.test_dataset = TensorDataset(data_ready["test_X"], data_ready["test_y"])
        self.test_loader = DataLoader(self.test_dataset, batch_size=32)

    def fit(self):
        lr = 0.01
        optimizer = EmaAmsGrad(self.lin_train.parameters(), self.lin, lr=lr, ema=0.99)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=20)
        val_loss = self.valid()
        print(f"valid before {val_loss}")
        for epoch in (range(1000)):
            train_loss = 0.
            n_total = 0
            for X, y in self.train_loader:
                optimizer.zero_grad()
                pred = self.predict(torch.as_tensor(X))
                loss = self.loss_fn(pred, torch.as_tensor(y))
                train_loss += loss.item() * X.shape[0]
                n_total += X.shape[0]
                loss.backward()
                optimizer.step()
            train_loss /= n_total
            val_loss = self.valid()
            test_loss = self.valid(test=True)
            scheduler.step(val_loss.sum())
            print(f"Epoch {epoch}, train mae: {train_loss}, val mse: {val_loss}"
                  f", test mse {test_loss}, ")

    def valid(self, test=False):
        with torch.no_grad():
            val_loss = 0.
            n_total = 0
            loader = self.test_loader if test else self.valid_loader
            for X, y in loader:
                pred = self.predict(torch.as_tensor(X))
                val_loss = val_loss + self.mse(pred, torch.as_tensor(y)).detach().cpu() * X.shape[0]
                n_total += X.shape[0]
            val_loss /= n_total
        return val_loss

    def predict(self, X):
        te = self.lin(X)
        wat_sol = coe * (te[:, 1] - te[:, 0])
        oct_sol = coe * (te[:, 2] - te[:, 0])
        return torch.cat([te, wat_sol.view(-1, 1), oct_sol.view(-1, 1)], dim=-1)

    @staticmethod
    def loss_fn(pred, tgt):
        mae = torch.mean((pred - tgt).abs(), dim=0)
        return mae.sum()

    @staticmethod
    def mse(pred, tgt):
        result = torch.mean((pred - tgt) ** 2, dim=0)
        return result


xgb_param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'gamma': [0.001, 0.01, 0.1, 0.7],
    'min_child_weight': range(1, 10, 3),
    'subsample': np.arange(0.1, 1.0, 0.3),
    'colsample_bytree': np.arange(0.1, 1.0, 0.3),
    'max_depth': range(3, 10, 3),
    'n_estimators': range(200, 1000, 200)
}

xgb_param_grid_default = {
}

rf_param_grid = {'n_estimators': [10, 50, 100, 200, 400], 'max_depth': [None, 3, 6, 12],
                 'min_samples_leaf': [1, 3, 5, 10, 50], 'min_impurity_decrease': [0, 0.01],
                 'max_features': ['auto', 'sqrt', 'log2', 0.7, 0.9]}

svr_param_grid = {'C': [0.1, 0.5, 1.0, 5.0, 10.0, 20, 50, 100, 500, 1000, 2000]}

dnn_param_grid = {
    "feature_dim": [160, 320, 480],
    "act": ["ssp", "relu", "swish"],
    "n_layers": [0, 1, 2, 3, 4, 5],
    "dim_decay": [True, False],
    "optimizer": ["AdaGrad"],
    "bn": [True]
}

linear_param_grid = {
    "fit_intercept": [True]
}

tmp = {'act': ['ssp'], 'dim_decay': [False], 'feature_dim': [320], 'n_layers': [3], 'optimizer': ["AdaGrad"],
       "bn": [True], "affine": [True]}


def get_data(pt_path, csv_path, root="../../dataProviders/data/processed/", embed_dim=None, sr_te=False,
             remove_atom_id=None, rd_split=None, tgt_name=_tgt_name):
    data_pt = torch.load(osp.join(root, pt_path))
    data_csv = pd.read_csv(osp.join(root, csv_path))
    # assert len(data_csv) == len(data_pt[tgt_name])

    if remove_atom_id is not None:
        remove_atom_id = set(remove_atom_id)
        atom_type_mask = []
        for smiles in data_csv["cano_smiles"]:
            from rdkit.Chem import MolFromSmiles
            mol = MolFromSmiles(smiles)
            retain_mol = True
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() in remove_atom_id:
                    retain_mol = False
                    break
            atom_type_mask.append(retain_mol)
    else:
        atom_type_mask = [True] * len(data_csv)
    atom_type_mask = torch.as_tensor(atom_type_mask).bool()
    # print(f"difference: {(torch.as_tensor(data_csv[tgt_name]) - data_pt[tgt_name]).abs().sum()}")

    data_ready = {}

    if sr_te:
        gas_e = data_pt["mol_prop"][:, 0].view(-1, 1)
        wat_sol = data_pt[tgt_name].view(-1, 1)
        wat_e = (gas_e + wat_sol / coe).view(-1, 1)
        oct_e = data_pt["mol_prop"][:, 2].view(-1, 1)
        oct_sol = coe * (oct_e - gas_e).view(-1, 1)
        sr_te_tgt = torch.cat([gas_e, wat_e, oct_e, wat_sol, oct_sol], dim=-1)
        data_pt[tgt_name] = sr_te_tgt

    for split in ["train", "valid", "test"]:
        if rd_split is None:
            mask = (torch.as_tensor(data_csv["group"].values == split).bool() & atom_type_mask)
        else:
            mask = torch.zeros_like(atom_type_mask).bool().fill_(False)
            mask[rd_split[f"{split}_index"]] = True
        embedding_name = "embedding"
        if ss:
            embedding_name = embedding_name + "_ss"
        data_ready[f"{split}_X"] = data_pt[embedding_name][mask].numpy()
        if embed_dim is not None:
            data_ready[f"{split}_X"] = data_ready[f"{split}_X"][:, :embed_dim]

        data_ready[f"{split}_y"] = data_pt[tgt_name][mask].numpy()

    return data_ready


def rmse(y_true, y_pred):
    return np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))


def rmse_model(model, data, split):
    return rmse(model.predict(data[f'{split}_X']), data[f'{split}_y'])


def train_model(model_getter, param_grid, remove_atom_id=None, split_files=None):
    if split_files is None:
        rd_splits = [None]
    else:
        rd_splits = glob.glob(split_files)
    current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    log_name = f"log_{current_time}.txt"
    csv_name = f"result_{current_time}.csv"

    losses = {}

    csv_dict = {}

    def _add_to_csv_dict(k: str, v, o: dict):
        if k not in o:
            o[k] = [v]
        else:
            o[k].append(v)

    for rd_split_p in rd_splits:
        rd_split = torch.load(rd_split_p) if rd_split_p is not None else None
        best_losses, best_param = hp_tune(model_getter, param_grid, log_name, remove_atom_id, rd_split)
        for _dict in [best_losses, best_param]:
            for key in _dict:
                _add_to_csv_dict(key, _dict[key], csv_dict)
        _add_to_csv_dict("split", rd_split_p, csv_dict)
        for key in best_losses:
            _add_to_csv_dict(key, best_losses[key], losses)

    for key in losses:
        losses[key] = torch.as_tensor(losses[key])
    with open(log_name, "a") as f:
        for key in losses:
            f.write(f"{key}: {losses[key].mean()} +- {losses[key].std()} \n")

    result_csv = pd.DataFrame(csv_dict)
    result_csv.to_csv(csv_name)

    time.sleep(1)


def hp_tune(model_getter, param_grid, log_name, remove_atom_id=None, rd_split=None):
    with open(log_name, "a") as f:
        f.write(f"Parm grid: {param_grid}\n")
        f.write(f"Model: {model_getter}\n")
        f.write(f"dataset: {dataset}\n")
        f.write(f"training target name: {_tgt_name}\n")
    best_val = np.inf
    best_param = None
    best_model = None
    data = get_data(dataset, csv, embed_dim=None, remove_atom_id=remove_atom_id, rd_split=rd_split)
    for param in tqdm(ParameterGrid(param_grid)):
        model = model_getter(param)
        model.fit(data["train_X"], data["train_y"])
        val_loss = rmse_model(model, data, "valid")
        if val_loss < best_val:
            best_param = param
            best_val = val_loss
            best_model = model
            with open(log_name, "a") as f:
                f.write(f"new best param with val loss {val_loss}:\n")
                f.write(f"{best_param} \n")

    best_losses = dict()

    with open(log_name, "a") as f:
        print_split_size(data, f)
        f.write("*" * 40 + "\n")
        for split in ["train", "valid", "test"]:
            this_rmse = rmse_model(best_model, data, split)
            best_losses[f"{split}_rmse"] = this_rmse
            f.write(f"train RMSE: {this_rmse}\n")
        # test_diff = best_model.predict(data['test_X']) - data['test_y']
        # f.write(f"test mol_lvl_detail: {test_diff}\n")
        # f.write(f"max {np.max(np.abs(test_diff))}\n")
        # f.write(f"rmse after removing: {np.sqrt(np.mean((test_diff[test_diff < 5])**2))}")
        f.write(f"best param: {best_param}\n")
    time.sleep(1)
    return best_losses, best_param


def train_dnn(param_grid, remove_atom_id=None):
    for param in ParameterGrid(param_grid):
        data = get_data(dataset, csv, embed_dim=None, remove_atom_id=remove_atom_id)
        model = DNNModel(data, **param)
        model.fit()


def test_dnn():
    param = {'act': 'relu', 'dim_decay': False, 'feature_dim': 480, 'n_layers': 2}
    data = get_data(dataset, csv, embed_dim=None, remove_atom_id={34, 53, 14})
    model = DNNModel(param, data, optimizer="Adagrad",
                     chk="../../raw_data/freeSolvEmbeddingTrain/EMBED-160-exp331/feature_dim_160/dnn-exp_run_2021-08-06_000641/best_model.pt")
    print(model.valid(split="train"))
    print(model.valid())
    print(model.valid(split="test"))


def train_te_lin():
    data = get_data(dataset, csv, embed_dim=None, sr_te=True)
    model = LinSRLayer(data)
    model.fit()


def print_split_size(data_ready, f):
    f.write("*" * 40 + "\n")
    for split in ["train", "valid", "test"]:
        assert data_ready[f"{split}_X"].shape[0] == data_ready[f"{split}_y"].shape[0]
        f.write(f"{split} size: {data_ready[f'{split}_X'].shape[0]} \n")


if __name__ == '__main__':
    # train_model(linear_getter, linear_param_grid, remove_atom_id={34, 53, 14})
    # train_model(svr_getter, svr_param_grid, remove_atom_id={34, 53, 14})
    train_model(xgb_getter, xgb_param_grid, remove_atom_id={34, 53, 14})

    # train_dnn(tmp, remove_atom_id={34, 53, 14})
    # test_dnn()

    # train_te_lin()
