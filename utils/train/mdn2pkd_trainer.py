import io
import os.path as osp
import pickle
import numpy as np
import yaml

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import torch
from utils.scores.casf_scores import plot_scatter_info

from utils.train.trainer import Trainer


class MDN2PKdTrainer(Trainer):
    def __init__(self, config_args, data_provider=None):
        super().__init__(config_args, data_provider)

        self.model_name = config_args["mdn2pkd_model"]
        self.embed_type = config_args["mdn_embed_type"]
        self.ds_pkl = osp.join(config_args["data_root"], "processed", config_args["dataset_name"])
        
        self._mdn2pkd_model = None
        self._x_features = None
        self._y_labels = None

    def train(self):
        self.log("Gathering training data")
        train_index: np.ndarray = self.ds["train_index"].numpy()
        val_index: np.ndarray = self.ds["val_index"].numpy()
        x_train = self.x_features[train_index, :]
        y_train = self.y_labels[train_index]
        x_val = self.x_features[val_index, :]
        y_val = self.y_labels[val_index]

        self.log("Setting up scaler")
        feat_scaler = StandardScaler()
        x_train_scaled = feat_scaler.fit_transform(x_train)
        scaler_save = osp.join(self.run_directory, "scaler.pkl")
        with open(scaler_save, "wb") as f:
            pickle.dump(feat_scaler, f)

        self.log("Training model")
        self.mdn2pkd_model.fit(x_train_scaled, y_train)
        y_train_pred = self.mdn2pkd_model.predict(x_train_scaled)
        train_mae = np.mean(np.abs(y_train - y_train_pred))
        model_save = osp.join(self.run_directory, "model.pkl")
        with open(model_save, "wb") as f:
            pickle.dump(self.mdn2pkd_model, f)

        self.log("Evaluating...")
        y_val_pred = self.mdn2pkd_model.predict(feat_scaler.transform(x_val))
        val_mae = np.mean(np.abs(y_val - y_val_pred))

        self.log("Visualizing...")
        r_train = plot_scatter_info(y_train, y_train_pred, self.run_directory, "pred_vs_label_train", 
                    "Predicted vs. Experimental pKd on the Training Set", None, None, "Experimental pKd", "Predicted", True)
        r_val = plot_scatter_info(y_val, y_val_pred, self.run_directory, "pred_vs_label_val", 
                    "Predicted vs. Experimental pKd on the Validation Set", None, None, "Experimental pKd", "Predicted", True)
        loss_info = {"MAE_train": train_mae.item(), "MAE_valid": val_mae.item(), 
                     "PearsonR_train": r_train.item(), "PearsonR_valid": r_val.item()}
        with open(osp.join(self.run_directory, "loss_info.yaml"), "w") as f:
            yaml.safe_dump(loss_info, f)

    @property
    def x_features(self):
        if self._x_features is not None:
            return self._x_features
        
        x_features = self.ds[f"{self.embed_type}_embed"]
        x_features = np.concatenate([t.numpy().reshape(1, -1) for t in x_features], axis=0)
        self._x_features = x_features
        return self._x_features
    
    @property
    def y_labels(self):
        if self._y_labels is not None:
            return self._y_labels
        
        y_labels = self.ds["pKd"]
        y_labels = np.asarray([i.item() for i in y_labels])
        self._y_labels = y_labels
        return self._y_labels

    @property
    def mdn2pkd_model(self):
        if self._mdn2pkd_model is not None:
            return self._mdn2pkd_model
        
        model_mapper = {"linear": LinearRegression, "rf": RandomForestRegressor,
                        "xgb": XGBRegressor}
        self._mdn2pkd_model = model_mapper[self.model_name]()
        return self._mdn2pkd_model

    @property
    def ds(self):
        if self._ds is not None:
            return self._ds
        with open(self.ds_pkl, "rb") as f:
            self._ds = CPU_Unpickler(f).load()
        return self._ds

# pickle load tensors that are stored as CUDA tensor
# https://github.com/pytorch/pytorch/issues/16797
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
