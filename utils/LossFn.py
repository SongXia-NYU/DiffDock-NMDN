from copy import deepcopy
import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch.distributions import Normal
from utils.configs import Config
from utils.data.data_utils import get_lig_batch, get_lig_natom, get_lig_z, get_prop, get_sample_id
from utils.eval.nmdn import NMDN_Calculator, calculate_probablity

from utils.tags import tags
from utils.utils_functions import kcal2ev

# R in kcal/(mol.K)
R = 1.98720425864083e-3
logP_to_watOct = 2.302585093 * R * 298.15
pKd2deltaG = -logP_to_watOct


def lossfn_factory(cfg: Config):
    loss_metric = cfg.training.loss_fn.loss_metric
    if loss_metric == "mdn":
        return MDNLossFn(cfg)
    elif loss_metric.startswith("mdn_"):
        return MDNMixLossFn(cfg)
    elif loss_metric == "kl_div":
        return KLDivRegressLossFn(cfg)
    loss_fn = LossFn(cfg=cfg)
    return loss_fn

class BaseLossFn:
    def __init__(self) -> None:
        pass

    # only model_output, data_batch and loss_detail is needed here
    # other parameters are only for compatibility
    def __call__(self, model_output, data_batch, is_training, loss_detail=False, mol_lvl_detail=False):
        raise NotImplementedError

    def inference_mode(self) -> None:
        pass
    
class KLDivRegressLossFn(BaseLossFn):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.target_names = cfg["target_names"]
        self.num_targets = len(self.target_names)
        self.cfg = cfg

    def __call__(self, model_output, data_batch, is_training, loss_detail=False, mol_lvl_detail=False):
        # the mol-level prediction. range: -inf to inf
        # [num_mols, num_targets * 2] -> [num_mols, num_targets, 2]
        mol_pred: torch.Tensor = model_output["mol_prop"]
        assert mol_pred.shape[1] == 2 * self.num_targets, (mol_pred.shape, self.num_targets)
        mol_pred = mol_pred.view(-1, self.num_targets, 2)
        pred_prob = self.log_softmax(mol_pred)
        
        prop_tgt = torch.cat([getattr(data_batch, name).view(-1, 1) for name in self.target_names], dim=-1)
        tgt_prob = torch.stack([1-prop_tgt, prop_tgt], dim=-1)
        kl_loss = self.kl_div(pred_prob, tgt_prob)
        if not loss_detail:
            return kl_loss
        
        # recording details needed during evaluation
        detail = {"kl_loss": kl_loss, "sample_id": data_batch.sample_id.detach().cpu()}
        mol_pred = mol_pred.detach().cpu()
        prop_tgt = prop_tgt.cpu()
        detail["n_units"] = mol_pred.shape[0]

        detail["PRED_RAW"] = mol_pred
        mol_pred = self.softmax(mol_pred)[:, :, -1]
        detail["PROP_PRED"] = mol_pred
        detail["PROP_TGT"] = prop_tgt
        mae_loss = torch.mean(torch.abs(mol_pred - prop_tgt), dim=0, keepdim=True)
        mse_loss = torch.mean((mol_pred - prop_tgt) ** 2, dim=0, keepdim=True)
        for i, name in enumerate(self.target_names):
            detail["MAE_{}".format(name)] = mae_loss[:, i].item()
            detail["MSE_{}".format(name)] = mse_loss[:, i].item()
        return kl_loss, detail


class MDNMixLossFn(BaseLossFn):
    """
    Using both MDN loss and normal regression loss (MAE, MSE, etc..)
    """
    def __init__(self, cfg: Config) -> None:
        super().__init__()

        self.mdn_loss_fn = MDNLossFn(cfg)
        normal_cfg = copy.deepcopy(cfg)
        normal_cfg.training.loss_fn.loss_metric = normal_cfg.training.loss_fn.loss_metric.split("_")[-1]
        self.normal_loss_fn = LossFn(normal_cfg)
        self.mse_loss = nn.MSELoss()
        self.mse_loss_noreduce = nn.MSELoss(reduction="none")

        loss_cfg = cfg.training.loss_fn
        self.w_mdn = loss_cfg["w_mdn"]
        self.w_regression = loss_cfg["w_regression"]

        no_pkd_score: bool = cfg.training.loss_fn.no_pkd_score
        self.compute_pkd_score: bool = (self.w_regression > 0.) and (not no_pkd_score)

    def inference_mode(self) -> None:
        self.mdn_loss_fn.inference_mode()
        self.normal_loss_fn.inference_mode()
        return super().inference_mode()

    def __call__(self, model_output, data_batch, is_training, loss_detail=False, mol_lvl_detail=False):
        # mdn loss
        mdn_out = 0.
        if self.w_mdn > 0.:
            mdn_out = self.mdn_loss_fn(model_output, data_batch, is_training, loss_detail, mol_lvl_detail)

        # regression loss
        reg_out = 0.
        if self.compute_pkd_score:
            reg_out = self.normal_loss_fn(model_output, data_batch, is_training, loss_detail, mol_lvl_detail)

        # mdn_out is a number during training
        if not loss_detail:
            return self.w_mdn*mdn_out + self.w_regression*reg_out

        loss_detail = {}
        total_loss = 0.
        # mdn_out is a tuple during evaluation (loss and a dictionary of loss details)
        if self.w_mdn > 0.:
            mdn_loss, mdn_detail = mdn_out
            mdn_detail["PROP_PRED_MDN"] = mdn_detail["PROP_PRED"]
            loss_detail.update(mdn_detail)
            total_loss = total_loss + self.w_mdn * mdn_loss

        # regression loss
        if self.compute_pkd_score:
            reg_loss, reg_detail = reg_out
            total_loss = total_loss + self.w_regression*reg_loss
            loss_detail.update(reg_detail)
        return total_loss, loss_detail

    @property
    def target_names(self):
        return self.normal_loss_fn.target_names


class MDNLossFn(BaseLossFn):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.target_names = None
        loss_cfg = cfg.training.loss_fn
        mdn_cfg = cfg.model.mdn

        self.mdn_threshold_train = mdn_cfg["mdn_threshold_train"]
        self.mdn_threshold_eval = mdn_cfg["mdn_threshold_eval"]
        # Aux tasks
        self.mdn_w_lig_atom_types = loss_cfg["mdn_w_lig_atom_types"]
        self.mdn_w_prot_atom_types = loss_cfg["mdn_w_prot_atom_types"]
        self.lig_atom_types = loss_cfg["mdn_w_lig_atom_types"] > 0.
        self.prot_atom_types = loss_cfg["mdn_w_prot_atom_types"] > 0.
        self.lig_atom_props = loss_cfg["mdn_w_lig_atom_props"] > 0.
        self.mdn_w_lig_atom_props = loss_cfg["mdn_w_lig_atom_props"]
        self.prot_sasa = loss_cfg["mdn_w_prot_sasa"] > 0.
        self.mdn_w_prot_sasa = loss_cfg["mdn_w_prot_sasa"]
        if self.lig_atom_types or self.prot_atom_types:
            self.ce_loss = torch.nn.CrossEntropyLoss()
        if self.lig_atom_props or self.prot_sasa:
            self.reg_loss_fn = nn.L1Loss(reduction="none")

        self.cfg = cfg
        # ignore auxillary task, only make predictions
        self.only_predict = False
        # compute external MDN scores: normalized scores, max MDN scores, reference at 6.5A, etc...
        self.compute_external_mdn: bool = cfg.model.mdn.compute_external_mdn or cfg.model.mdn.nmdn_eval
        self.nmdn_calculator = NMDN_Calculator(cfg)

    def inference_mode(self) -> None:
        self.only_predict = True
        return super().inference_mode()

    def __call__(self, model_output, data_batch, is_training, loss_detail=False, mol_lvl_detail=False):
        mdn = mdn_loss_fn(model_output["pi"], model_output["sigma"], model_output["mu"], model_output["dist"])
        mdn = mdn[torch.where(model_output["dist"] <= self.mdn_threshold_train)[0]]
        mdn = mdn.mean()
        total_loss = mdn

        detail = {"sample_id": get_sample_id(data_batch).detach().cpu(), "mdn_loss": mdn.detach().cpu().item()}
        if self.lig_atom_types and not self.only_predict:
            lig_atom_ce = self.ce_loss(model_output["lig_atom_types"], model_output["lig_atom_types_label"]).mean()
            total_loss = total_loss + self.mdn_w_lig_atom_types * lig_atom_ce
            detail["CE_lig_atom"] = lig_atom_ce.detach().cpu().item()

        if self.prot_atom_types and not self.only_predict:
            prot_atom_ce = self.ce_loss(model_output["prot_atom_types"], model_output["prot_atom_types_label"]).mean()
            total_loss = total_loss + self.mdn_w_prot_atom_types * prot_atom_ce
            detail["CE_prot_atom"] = prot_atom_ce.detach().cpu().item()

        if self.lig_atom_props and not self.only_predict:
            # element-wise loss
            lig_props_pred = post_calculation(model_output["lig_atom_props"])
            lig_props_label = post_calculation(data_batch.atom_prop)
            lig_atom_props_loss = self.reg_loss_fn(lig_props_pred, lig_props_label)
            # element-wise weight: for those molecules with missing labels, the weight is zero, else the weight is mdn_w_lig_atom_props
            lig_loss_by_props = (data_batch.mdn_w_lig_atom_props.view(-1, 1) * lig_atom_props_loss).mean(dim=0)
            lig_atom_props_loss = lig_loss_by_props.mean()
            total_loss = total_loss + lig_atom_props_loss
            lig_loss_by_props = lig_loss_by_props.detach().cpu()
            detail["MAE_lig_E_gas(eV)"] = lig_loss_by_props[0].item() / self.mdn_w_lig_atom_props
            detail["MAE_lig_E_water(eV)"] = lig_loss_by_props[1].item() / self.mdn_w_lig_atom_props
            detail["MAE_lig_E_oct(eV)"] = lig_loss_by_props[2].item() / self.mdn_w_lig_atom_props
            detail["MAE_lig_gas->water(kcal/mol)"] = lig_loss_by_props[3].item() / self.mdn_w_lig_atom_props
            detail["MAE_lig_gas->oct(kcal/mol)"] = lig_loss_by_props[4].item() / self.mdn_w_lig_atom_props
            detail["MAE_lig_oct->water(kcal/mol)"] = lig_loss_by_props[5].item() / self.mdn_w_lig_atom_props

        if self.prot_sasa and not self.only_predict:
            # element-wise loss
            prot_sasa_loss = self.reg_loss_fn(model_output["prot_sasa"], data_batch.martini_sasa.view(-1, 1))
            # element-wise weight: for those molecules with missing labels, the weight is zero, else the weight is mdn_w_lig_atom_props
            prot_sasa_loss = (data_batch.mdn_w_prot_sasa.view(-1, 1) * prot_sasa_loss).mean()
            total_loss = total_loss + prot_sasa_loss
            detail["MAE_prot_sasa"] = prot_sasa_loss.detach().cpu().item() / self.mdn_w_prot_sasa

        if not loss_detail:
            return total_loss

        detail["n_units"] = get_lig_natom(data_batch).shape[0]
        pair_prob = calculate_probablity(model_output["pi"], model_output["sigma"], model_output["mu"], model_output["dist"])
        if self.compute_external_mdn:
            detail.update(self.nmdn_calculator(pair_prob, model_output, data_batch))
        pair_prob[torch.where(model_output["dist"] > self.mdn_threshold_eval)[0]] = 0.
        probx = scatter_add(pair_prob, model_output["C_batch"].to(get_lig_natom(data_batch).device), dim=0, dim_size=detail["n_units"])
        detail["PROP_PRED"] = probx
        if "mdn_hist" in model_output: 
            detail["mdn_hist"] = model_output["mdn_hist"].detach().cpu()
        return total_loss, detail

def mdn_loss_fn(pi, sigma, mu, y, eps=1e-10):
    normal = Normal(mu, sigma)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    loss = -torch.logsumexp(torch.log(pi + eps) + loglik, dim=1)
    return loss

def post_calculation(raw: torch.Tensor):
    # 0: gas, 1: water, 2: oct
    assert raw.shape[1] == 3
    transfers = [raw[:, 1]-raw[:, 0], raw[:, 2]-raw[:, 0], raw[:, 1]-raw[:, 2]]
    transfers = torch.stack(transfers, dim=1) / kcal2ev
    res = torch.concat([raw, transfers], dim=-1)
    return res


class LossFn(BaseLossFn):
    def __init__(self, cfg: Config, only_predict=False):
        """
        Loss function to deal with the loss calculation logic
        :param w_e: energy weight
        :param w_f: force weight
        :param w_q: charge weight
        :param w_p: dipole weight
        :param action: controls how the loss is calculated
        :param auto_sol: used in sPhysNet-MT, when model predicts gas, water and octanol energy,
        it will calculate transfer energies indirectly
        :param target_names: names of the target labels for loss calculation
        :param config_dict: config dictionary for other controls
        :param only_predict: model only make prediction, no loss is calculated.
        """
        super().__init__()
        self.cfg = cfg
        lossfn_cfg = cfg.training.loss_fn
        self.lossfn_cfg = cfg.training.loss_fn
        self._wat_ref = None
        self.only_predict = only_predict
        self.loss_metric = lossfn_cfg.loss_metric.lower()
        self.loss_metric_upper = self.loss_metric.upper()
        assert self.loss_metric in tags.loss_metrics

        self.target_names = deepcopy(lossfn_cfg.target_names)
        self.action = deepcopy(lossfn_cfg.action)
        self.w_e = 1.
        self.w_f = lossfn_cfg.force_weight
        self.w_q = lossfn_cfg.charge_weight
        self.w_d = lossfn_cfg.dipole_weight

        self._target_name_to_id = None

        if self.loss_metric == "ce":
            assert self.action in ["names", "names_atomic"]
            assert len(self.target_names) == 1, "Currently only support single task classification"
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.num_targets = len(self.target_names)

    def inference_mode(self) -> None:
        self.only_predict = True
        return super().inference_mode()

    def __call__(self, model_output, data_batch, is_training, loss_detail=False, mol_lvl_detail=False):
        if self.only_predict:
            assert loss_detail
            pred = self.get_processed_pred(model_output, data_batch)
            fake_loss = torch.as_tensor(0.)
            detail = {"PROP_PRED": pred.detach().cpu(), "sample_id": get_sample_id(data_batch).detach().cpu(),
                      "n_units": get_lig_natom(data_batch).shape[0]}
            return fake_loss, detail

        if self.loss_metric == "ce":
            return self.multi_classification(model_output, data_batch, is_training, loss_detail, mol_lvl_detail)
        
        if self.loss_metric == "bce":
            return self.binary_classification(model_output, data_batch, is_training, loss_detail, mol_lvl_detail)

        if self.action in ["names", "names_and_QD", "names_atomic"]:
            detail = {}

            prop_tgt, prop_pred = self.get_pred_target(model_output, data_batch)
            coe = 1.

            mae_loss = torch.mean(torch.abs(prop_pred - prop_tgt), dim=0, keepdim=True)
            mse_loss = torch.mean((prop_pred - prop_tgt) ** 2, dim=0, keepdim=True)
            rmse_loss = torch.sqrt(mse_loss)

            if self.loss_metric == "mae":
                total_loss = torch.sum(coe * mae_loss)
            elif self.loss_metric == "mse":
                total_loss = torch.sum(coe * mse_loss)
            elif self.loss_metric == "rmse":
                total_loss = torch.sum(coe * rmse_loss)
            else:
                raise ValueError("Invalid total loss: " + self.loss_metric)

            if loss_detail:
                # record details including MAE, RMSE, Difference, etc..
                # It is required while valid and test step but not required in training
                for i, name in enumerate(self.target_names):
                    detail["MAE_{}".format(name)] = mae_loss[:, i].item()
                    detail["MSE_{}".format(name)] = mse_loss[:, i].item()
                if mol_lvl_detail:
                    detail["PROP_PRED"] = prop_pred.detach().cpu()
                    detail["PROP_TGT"] = prop_tgt.detach().cpu()
            else:
                detail = None

            if self.action == "names_and_QD":
                if self.loss_metric == "mae":
                    q_loss = torch.mean(torch.abs(model_output["Q_pred"] - data_batch.Q))
                    d_loss = torch.mean(torch.abs(model_output["D_pred"] - data_batch.D))
                else:
                    q_loss = torch.mean((model_output["Q_pred"] - data_batch.Q) ** 2)
                    d_loss = torch.mean((model_output["D_pred"] - data_batch.D) ** 2)
                    if self.loss_metric == "rmse":
                        q_loss = torch.sqrt(q_loss)
                        d_loss = torch.sqrt(d_loss)
                total_loss = total_loss + self.w_q * q_loss + self.w_d * d_loss
                if loss_detail:
                    detail["{}_Q".format(self.loss_metric_upper)] = q_loss.item()
                    detail["{}_D".format(self.loss_metric_upper)] = d_loss.item()
                    if mol_lvl_detail:
                        detail["DIFF_Q"] = (model_output["Q_pred"] - data_batch.Q).detach().cpu().view(-1)
                        detail["DIFF_D"] = (model_output["D_pred"] - data_batch.D).detach().cpu().view(-1)

            if loss_detail:
                # n_units: number of molecules
                detail["n_units"] = get_lig_natom(data_batch).shape[0]
                detail["ATOM_MOL_BATCH"] = get_lig_batch(data_batch).detach().cpu()
                detail["ATOM_Z"] = get_lig_z(data_batch).detach().cpu()
                if hasattr(data_batch, "sample_id"):
                    detail["sample_id"] = get_sample_id(data_batch).detach().cpu()
                for key in ["atom_embedding"]:
                    if key in model_output.keys():
                        detail[key] = model_output[key].detach().cpu()
                return total_loss, detail

            return total_loss

        elif self.action == "E":
            return self.legacy_physnet_cal(model_output, data_batch, is_training, loss_detail, mol_lvl_detail)
        else:
            raise ValueError("Invalid action: {}".format(self.action))

    def get_processed_pred(self, model_output: dict, data_batch):
        # multi-task prediction
        if self.action in tags.requires_atomic_prop:
            # TODO predict atom and mol prop at the same time
            prop_name = "atom_prop"
        else:
            prop_name = "mol_prop"
        prop_pred = model_output[prop_name]
        if "mol_prop_pool" in model_output.keys():
            prop_pred = torch.cat([prop_pred, model_output["mol_prop_pool"]], dim=-1)

        if "pKd" in self.target_names:
            assert prop_pred.shape[-1] == 1
            prop_pred = prop_pred / pKd2deltaG
        return prop_pred

    def get_pred_target(self, model_output: dict, data_batch):
        """
        Get energy target from data_batch
        Solvation energy is in kcal/mol but gas/water/octanol energy is in eV
        """
        prop_pred = self.get_processed_pred(model_output, data_batch)
        prop_tgt = torch.cat([get_prop(data_batch, name).view(-1, 1) for name in self.target_names], dim=-1)
        if self.loss_metric not in ["ce"]:
            assert prop_pred.shape[-1] == self.num_targets, f"{prop_pred.shape}, {self.num_targets}"

        if self.mask_atom:
            mask = data_batch.mask.bool()
            prop_tgt = prop_tgt[mask, :]
        return prop_tgt, prop_pred

    def multi_classification(self, model_output, data, is_training, loss_detail=False, mol_lvl_detail=False):
        """
        Used when loss function is CrossEntropyLoss
        :param model_output:
        :param data:
        :param is_training:
        :param loss_detail:
        :param mol_lvl_detail:
        :return:
        """
        prop_tgt, prop_pred = self.get_pred_target(model_output, data)
        assert prop_tgt.shape[-1] == 1, "Currently only support single task classification"
        prop_tgt = prop_tgt.view(-1)
        total_loss = self.ce_loss(prop_pred, prop_tgt)
        if not loss_detail:
            return total_loss

        label_pred = torch.argmax(prop_pred, dim=-1)
        accuracy = (label_pred == prop_tgt).sum() / len(label_pred)

        # n_units: number of molecules
        detail = {"n_units": data.N.shape[0], "accuracy": float(accuracy)}
        if mol_lvl_detail:
            detail["RAW_PRED"] = prop_pred.detach().cpu()
            detail["LABEL"] = prop_tgt.cpu()
            detail["ATOM_MOL_BATCH"] = data.atom_mol_batch.detach().cpu()
        return total_loss, detail
    
    def binary_classification(self, model_output, data, is_training, loss_detail=False, mol_lvl_detail=False):
        """
        """
        prop_tgt, prop_pred = self.get_pred_target(model_output, data)
        assert prop_pred.shape[-1] == 1, "For binary classification, you only predict a number (-inf, inf)"
        confidence_loss = F.binary_cross_entropy_with_logits(prop_pred.reshape(-1), prop_tgt.reshape(-1).type(prop_pred.dtype))
        if not loss_detail:
            return confidence_loss

        label_pred = (prop_pred > 0.).long()
        accuracy = (label_pred == prop_tgt).sum() / len(label_pred)

        # n_units: number of molecules
        detail = {"n_units": data.N.shape[0], "accuracy": float(accuracy)}
        return confidence_loss, detail

    def legacy_physnet_cal(self, model_output, data, is_training, loss_detail=False, mol_lvl_detail=False):
        # default PhysNet setting
        assert self.loss_metric == "mae"
        E_loss, F_loss, Q_loss, D_loss = 0, 0, 0, 0
        E_loss = self.w_e * torch.mean(torch.abs(model_output["mol_prop"] - data.E))

        Q_loss = self.w_q * torch.mean(torch.abs(model_output["Q_pred"] - data.Q))

        D_loss = self.w_d * torch.mean(torch.abs(model_output["D_pred"] - data.D))

        if loss_detail:
            return E_loss + F_loss + Q_loss + D_loss, {"MAE_E": E_loss.item(), "MAE_F": F_loss,
                                                       "MAE_Q": Q_loss.item(), "MAE_D": D_loss.item(),
                                                       "DIFF_E": (model_output[
                                                                      "mol_prop"] - data.E).detach().cpu().view(-1)}
        else:
            return E_loss + F_loss + Q_loss + D_loss

    @property
    def target_name_to_id(self):
        if self._target_name_to_id is None:
            temp = {}
            for name in self.target_names:
                temp[name] = self.target_names.index(name)
            self._target_name_to_id = temp
        return self._target_name_to_id
