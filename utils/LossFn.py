from copy import deepcopy
import copy
import logging
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import numpy as np
from torch_scatter import scatter_add, scatter_max
from torch.distributions import Normal
from utils.data.data_utils import get_lig_batch, get_lig_natom, get_lig_z, get_prop, get_sample_id
from utils.eval.nmdn import NMDN_Calculator, calculate_probablity

from utils.tags import tags
from utils.utils_functions import DistCoeCalculator, kcal2ev, evidential_loss_new
from sklearn.metrics import roc_auc_score

# R in kcal/(mol.K)
R = 1.98720425864083e-3
logP_to_watOct = 2.302585093 * R * 298.15
pKd2deltaG = -logP_to_watOct


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
    def __init__(self, config_dict) -> None:
        super().__init__()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.target_names = config_dict["target_names"]
        self.num_targets = len(self.target_names)
        self.config_dict = config_dict

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
    def __init__(self, cfg: dict) -> None:
        super().__init__()

        self.mdn_loss_fn = MDNLossFn(cfg)
        normal_config = copy.copy(cfg)
        normal_config["loss_metric"] = normal_config["loss_metric"].split("_")[-1]
        w_e, w_f, w_q, w_p = 1, normal_config["force_weight"], normal_config["charge_weight"], normal_config["dipole_weight"]
        self.normal_loss_fn = LossFn(w_e=w_e, w_f=w_f, w_q=w_q, w_p=w_p, config_dict=normal_config, **normal_config)
        self.mse_loss = nn.MSELoss()
        self.mse_loss_noreduce = nn.MSELoss(reduction="none")

        self.w_mdn = cfg["w_mdn"]
        self.w_regression = cfg["w_regression"]
        self.w_cross_mdn_pkd = cfg["w_cross_mdn_pkd"]
        self.cross_mdn_prop_name = cfg["cross_mdn_prop_name"]
        self.cross_mdn_behaviour = cfg["cross_mdn_behaviour"]
        self.cutoff4cross_loss = cfg["mdn_threshold_prop"]

        no_pkd_score: bool = cfg.get("no_pkd_score", False)
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

        cross_mdn_pkd = self.compute_cross_mdn_pkd(model_output, data_batch)

        # mdn_out is a number during training
        if not loss_detail:
            return self.w_mdn*mdn_out + self.w_regression*reg_out + self.w_cross_mdn_pkd*cross_mdn_pkd

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

        if self.w_cross_mdn_pkd > 0.:
            total_loss = total_loss + self.w_cross_mdn_pkd*cross_mdn_pkd
            loss_detail["MSE_cross_mdn_pkd"] = cross_mdn_pkd
        return total_loss, loss_detail
    
    def compute_cross_mdn_pkd(self, model_output, data_batch):
        # cross mdn pkd loss
        assert self.w_cross_mdn_pkd >= 0., self.w_cross_mdn_pkd
        if self.w_cross_mdn_pkd == 0.:
            return 0.

        pair_pkd = model_output["pair_atom_prop"]
        pair_mdn = model_output[self.cross_mdn_prop_name]
        if self.cross_mdn_behaviour == "pair_mean":
            cross_mdn_pkd = self.mse_loss(pair_pkd, pair_mdn)
        else:
            assert self.cross_mdn_behaviour == "mol_sum_mean", self.cross_mdn_behaviour
            pair_loss = self.mse_loss_noreduce(pair_pkd, pair_mdn).view(-1)
            # a batch_index mapping pair-properties to molecule in the batch
            pair_mol_batch = model_output["C_batch"]
            # the pair index should be masked based on distance cutoff
            pl_dist_mask = (model_output["dist"] <= self.cutoff4cross_loss).view(-1)
            pair_mol_batch = pair_mol_batch[pl_dist_mask]

            mol_sum = scatter_add(pair_loss, pair_mol_batch, dim=0)
            cross_mdn_pkd = mol_sum.mean()
        return cross_mdn_pkd

    @property
    def target_names(self):
        return self.normal_loss_fn.target_names


class MDNLossFn(BaseLossFn):
    def __init__(self, config_dict) -> None:
        super().__init__()
        self.target_names = None

        self.mdn_threshold_train = config_dict["mdn_threshold_train"]
        self.mdn_threshold_eval = config_dict["mdn_threshold_eval"]
        # Aux tasks
        self.mdn_w_lig_atom_types = config_dict["mdn_w_lig_atom_types"]
        self.mdn_w_prot_atom_types = config_dict["mdn_w_prot_atom_types"]
        self.lig_atom_types = config_dict["mdn_w_lig_atom_types"] > 0.
        self.prot_atom_types = config_dict["mdn_w_prot_atom_types"] > 0.
        self.lig_atom_props = config_dict["mdn_w_lig_atom_props"] > 0.
        self.mdn_w_lig_atom_props = config_dict["mdn_w_lig_atom_props"]
        self.prot_sasa = config_dict["mdn_w_prot_sasa"] > 0.
        self.mdn_w_prot_sasa = config_dict["mdn_w_prot_sasa"]
        if self.lig_atom_types or self.prot_atom_types:
            self.ce_loss = torch.nn.CrossEntropyLoss()
        if self.lig_atom_props or self.prot_sasa:
            self.reg_loss_fn = nn.L1Loss(reduction="none")

        self.config_dict = config_dict
        # ignore auxillary task, only make predictions
        self.only_predict = False
        # compute external MDN scores: normalized scores, max MDN scores, reference at 6.5A, etc...
        self.compute_external_mdn: bool = config_dict.get("compute_external_mdn", False)
        self.nmdn_calculator = NMDN_Calculator(config_dict)
        # distance coefficient when evaluating
        if config_dict["val_pair_prob_dist_coe"] is not None:
            msg = "The use of 'val_pair_prob_dist_coe' is deprecated. It will no longer take effect."
            print(msg)
            logging.warn(msg)

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
    def __init__(self, w_e=1., w_f=1., w_q=1., w_p=1., action: Union[List[str], str] = "E", auto_sol=False,
                 target_names=None, config_dict=None, only_predict=False, lamda_sol=None, wat_ref_file=None,
                 auto_pl_water_ref=False, **__):
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
        self.auto_pl_water_ref = auto_pl_water_ref
        self.wat_ref_file = wat_ref_file
        self._wat_ref = None
        self.lamda_sol = lamda_sol
        self.only_predict = only_predict
        self.loss_metric = config_dict["loss_metric"].lower()
        self.z_loss_weight = config_dict["z_loss_weight"]
        # keep only solvation energy/logP, only used in transfer learning on exp datasets
        self.keep = config_dict["keep"]
        self.mask_atom = config_dict["mask_atom"]
        self.flex_sol = config_dict["flex_sol"]
        self.loss_metric_upper = self.loss_metric.upper()
        assert self.loss_metric in tags.loss_metrics

        self.target_names = deepcopy(target_names)
        self.action = deepcopy(action)
        self.w_e = w_e
        self.w_f = w_f
        self.w_q = w_q
        self.w_d = w_p
        self.auto_sol = auto_sol
        self.auto_sol_no_conv = config_dict["auto_sol_no_conv"]
        if self.auto_sol:
            if "watEnergy" in self.target_names and "gasEnergy" in self.target_names:
                self.target_names.append("CalcSol")
            if "octEnergy" in self.target_names and "gasEnergy" in self.target_names:
                self.target_names.append("CalcOct")
            if "watEnergy" in self.target_names and "octEnergy" in self.target_names:
                self.target_names.append("watOct")
        if self.auto_pl_water_ref:
            assert not self.auto_sol
            assert self.keep is None
            self.target_names = ["E_wat(eV)", "E_protein(eV)", "DeltaG(kcal/mol)"]
        self._target_name_to_id = None

        if self.loss_metric == "ce":
            assert self.action in ["names", "names_atomic"]
            assert len(self.target_names) == 1, "Currently only support single task classification"
        self.ce_loss = torch.nn.CrossEntropyLoss()
        if self.loss_metric == "evidential":
            self.evi_loss = evidential_loss_new
            self.soft_plus = torch.nn.Softplus()
        self.num_targets = len(self.target_names)

        ds_name_base = config_dict["data_provider"].split('[')[0]
        self.delta_learning_pkd = config_dict["delta_learning_pkd"]
        self.delta_learning = (ds_name_base in ["pl_delta_learning"] or config_dict["delta_learning_pkd"])
        self.ignore_nan = config_dict["regression_ignore_nan"]

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

            evi_cal_dict = {}
            if self.loss_metric == "evidential":
                min_val = 1e-6
                means, log_lambdas, log_alphas, log_betas = torch.split(prop_pred, prop_pred.shape[-1] // 4, dim=-1)
                lambdas = self.soft_plus(log_lambdas) + min_val
                # add 1 for numerical constraints of Gamma function
                alphas = self.soft_plus(log_alphas) + min_val + 1
                betas = self.soft_plus(log_betas) + min_val
                evi_cal_dict["mu"] = means
                evi_cal_dict["v"] = lambdas
                evi_cal_dict["alpha"] = alphas
                evi_cal_dict["beta"] = betas
                prop_pred = means
                evi_cal_dict["targets"] = prop_tgt

            coe = 1.
            if self.flex_sol:
                # backward compatibility when multi-tasking on Frag20-solv-678k
                mask = data_batch.mask
                batch = torch.ones_like(mask).long()
                batch[mask] = 0
                diff = prop_pred - prop_tgt
                mae_loss = torch_scatter.scatter_mean(diff.abs(), batch, dim=0)[[0], :]
                mse_loss = torch_scatter.scatter_mean(diff ** 2, batch, dim=0)[[0], :]
                rmse_loss = torch.sqrt(mse_loss)
                if self.lamda_sol is not None:
                    coe = self.lamda_sol / 23.061
                    coe = torch.as_tensor([1./coe, 1./coe, 1./coe, 1., 1., 1.]).view(-1, 1).to(mae_loss.device)

                flex_units = {}
                for i, key in enumerate(self.target_names):
                    flex_units[key] = (mask[:, i]).sum().item()
                detail["flex_units"] = flex_units
            else:
                if self.ignore_nan:
                    # masking examples with target values, [batch_size, n_targets]
                    val_mask = ~ torch.isnan(prop_tgt)
                    # the number of examples with target values, [n_targets,]
                    val_count = torch.sum(val_mask, dim=0)
                    # normalization: since only part of the examples have the target properties,
                    # the mean value is lower than expected. The norm term is map them back.
                    norm = (val_mask.shape[0]) / val_count
                    # Shape [1, n_targets]
                    norm = torch.where(val_count != 0, norm, 0.0).view(1, -1)
                    prop_pred = torch.where(val_mask, prop_pred, 0.0)
                    prop_tgt = torch.where(val_mask, prop_tgt, 0.0)
                else:
                    norm = 1.
                mae_loss = torch.mean(torch.abs(prop_pred - prop_tgt), dim=0, keepdim=True) * norm
                mse_loss = torch.mean((prop_pred - prop_tgt) ** 2, dim=0, keepdim=True) * norm
                rmse_loss = torch.sqrt(mse_loss)

            if self.loss_metric == "mae":
                total_loss = torch.sum(coe * mae_loss)
            elif self.loss_metric == "mse":
                total_loss = torch.sum(coe * mse_loss)
            elif self.loss_metric == "rmse":
                total_loss = torch.sum(coe * rmse_loss)
            elif self.loss_metric == "evidential":
                total_loss = torch.sum(coe * self.evi_loss(**evi_cal_dict))
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
                    if self.loss_metric == "evidential":
                        detail["UNCERTAINTY"] = (betas / (lambdas * (alphas - 1))).detach().cpu()
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

            if self.z_loss_weight > 0:
                assert "first_layer_vi" in model_output
                z_loss = self.ce_loss(model_output["first_layer_vi"], get_lig_z(data_batch))
                total_loss = total_loss + self.z_loss_weight * z_loss
                if loss_detail:
                    detail["z_loss"] = z_loss.item()
                    detail["Z_PRED"] = torch.argmax(model_output["first_layer_vi"].detach().cpu(), dim=-1)

            # when the whole batch has none example with specified property (e.g. pKd in BioLip), 
            # total_loss is nan. we do not want nan be back-propagated in this case.
            if self.ignore_nan and torch.isnan(total_loss):
                total_loss = torch.zeros_like(total_loss)

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

        # "auto_sol" is used in training the multi-task model for both electronic energy and transfer energy.
        if self.auto_sol:
            if self.auto_sol_no_conv:
                coe = 1.
            else:
                coe = kcal2ev
            total_pred = [prop_pred]
            if "gasEnergy" in self.target_name_to_id.keys():
                target_name_to_id = self.target_name_to_id
            else:
                assert self.keep
                target_name_to_id = {
                    "gasEnergy": 0, "watEnergy": 1, "octEnergy": 2
                }
            for sol_name in ["watEnergy", "octEnergy"]:
                if sol_name in target_name_to_id.keys():
                    gas_id = target_name_to_id["gasEnergy"]
                    sol_id = target_name_to_id[sol_name]
                    # converting it to kcal/mol
                    total_pred.append((prop_pred[:, sol_id] - prop_pred[:, gas_id]).view(-1, 1) / coe)
            if "watEnergy" in target_name_to_id.keys() and "octEnergy" in target_name_to_id.keys():
                wat_id = target_name_to_id["watEnergy"]
                oct_id = target_name_to_id["octEnergy"]
                total_pred.append((prop_pred[:, wat_id] - prop_pred[:, oct_id]).view(-1, 1) / coe)
            prop_pred = torch.cat(total_pred, dim=-1)

        if self.auto_pl_water_ref:
            assert self.keep is None
            deltaG = (prop_pred[:, [1]] - prop_pred[:, [0]]) / kcal2ev
            prop_pred = torch.cat([prop_pred, deltaG], dim=-1)

        if not self.flex_sol:
            # when you want to predict multiple targets but only use one to calculate loss.
            if self.keep == "waterSol":
                prop_pred = prop_pred[:, [3]]
            elif self.keep == "logP":
                prop_pred = prop_pred[:, [5]] / logP_to_watOct
            elif self.keep == "watOct":
                prop_pred = prop_pred[:, [5]]
            elif self.keep == "pKd":
                # bound ligand energy minus energy in water
                prop_pred = prop_pred[:, [3]] - prop_pred[:, [1]]
            elif self.keep == "pKd_ref":
                prop_pred = prop_pred[:, [3]] - self.get_wat_ref(data_batch).to(prop_pred.device)
            else:
                assert self.keep is None, f"Invalid keep arg: {self.keep}"

        if self.mask_atom:
            mask = data_batch.mask.bool()
            prop_pred = prop_pred[mask, :]

        if self.delta_learning:
            prop_pred = prop_pred + data_batch.base_value.view(-1, 1)
            if self.delta_learning_pkd:
                prop_pred = prop_pred / kcal2ev

        if "pKd" in self.target_names:
            assert prop_pred.shape[-1] == 1
            assert not self.auto_pl_water_ref
            prop_pred = prop_pred / pKd2deltaG
        return prop_pred

    def get_pred_target(self, model_output: dict, data_batch):
        """
        Get energy target from data_batch
        Solvation energy is in kcal/mol but gas/water/octanol energy is in eV
        """
        prop_pred = self.get_processed_pred(model_output, data_batch)
        if self.auto_pl_water_ref:
            wat_ref = self.get_wat_ref(data_batch).to(prop_pred.device)
            deltaG = get_prop(data_batch, "pKd").view(-1, 1) * pKd2deltaG
            e_protein = wat_ref + deltaG * kcal2ev
            prop_tgt = torch.cat([wat_ref, e_protein, deltaG], dim=-1)
        else:
            prop_tgt = torch.cat([get_prop(data_batch, name).view(-1, 1) for name in self.target_names], dim=-1)
        if self.loss_metric not in ["ce", "evidential"]:
            assert prop_pred.shape[-1] == self.num_targets, f"{prop_pred.shape}, {self.num_targets}"
        elif self.loss_metric == "evidential":
            # mu, v, alpha, beta
            assert prop_pred.shape[-1] == self.num_targets * 4

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

        # if 'F' in data_batch.keys():
        #     F_loss_batch = torch_geometric.utils.scatter_('mean', torch.abs(F_pred - data_batch['F'].to(device)),
        #                                                   data_batch['atom_to_mol_batch'].to(device))
        #     F_loss = self.w_f * torch.sum(F_loss_batch) / 3

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
    
    def get_wat_ref(self, data_batch):
        if self.wat_ref_file is not None:
            return self.wat_ref[data_batch.sample_id, :]
        
        atom_prop = data_batch.atom_prop[:, [1]]
        mol_batch = get_lig_batch(data_batch)
        return torch_scatter.scatter_add(atom_prop, mol_batch, dim=0)

    @property
    def wat_ref(self):
        if self._wat_ref is None:
            assert self.wat_ref_file is not None
            d = torch.load(self.wat_ref_file)
            wat_ref = d["PROP_PRED"][d["sample_id"], [1]]
            self._wat_ref = wat_ref.view(-1, 1)
        return self._wat_ref
