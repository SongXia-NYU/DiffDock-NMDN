from typing import Tuple, List
import torch
import torch.nn as nn
from Networks.PairedPropLayers.HeteroPairedPropLayer import HeteroPairedPropLayer
from Networks.PairedPropLayers.ProtSingleMDNLayer import ProtSingleMDNLayer
from Networks.PhysLayers.PhysModule import PhysModule
from Networks.PhysLayers.PhysModuleGlobalNode import GlobalNodeReadoutPooling, PhysModuleGlobalNode
from Networks.SharedLayers.PNAConvPoolReadout import PNAConvPoolReadout
from Networks.SharedLayers.PoolingReadoutLayer import PoolingReadout
from Networks.SharedLayers.AtomToEdgeLayer import AtomToEdgeLayer
from Networks.SharedLayers.EdgeToAtomLayer import EdgeToAtomLayer
from Networks.SharedLayers.GatedEmbedLayer import GatedEmbedLayer
from Networks.PairedPropLayers.MDNLayer import MDNLayer
from Networks.SharedLayers.MartiniToAtomPooling import MartiniToAtomPooling
from Networks.PairedPropLayers.PairedPropLayer import MPNNPairedPropLayer, MDNPropLayer, NMDN_AuxPropLayer
from Networks.SharedLayers.ProtEmbedTransformLayer import ProtEmbedTransformLayer
from Networks.PairedPropLayers.NotEnoughMDNLayers import KanoProtMDNLayer, MetalLigMDNLayer, ProtProtMDNLayer, ComENetProtMDNLayer
from Networks.PairedPropLayers.ProtProtIntraMDNLayer import ProtProtIntraMDNLayer
from Networks.SharedLayers.ReadoutPoolingLayer import ReadoutPooling
from Networks.UncertaintyLayers.MCDropout import ConcreteDropout
from torch_geometric.data import Dataset
from Networks.comenet.ComENetAtomEmbed import ComENetAtomEmbed
from Networks.comenet.ComENetMolProp import ComENetMolProp
from Networks.comenet.SphereNet import SphereNetWrapped

from utils.configs import Config
from utils.utils_functions import error_message, option_solver


class MainModuleFactory:
    def __init__(self, config_dict: Config, expansion_info_getter: dict, base_unit_getter: dict, ds: Dataset) -> None:
        self.previous_module = "P"
        self.cfg: Config = config_dict
        self.expansion_info_getter = expansion_info_getter
        self.base_unit_getter = base_unit_getter
        self.ds = ds

    def get_module(self, module_str: str, bonding_str: str, activation: str, n_output: int, energy_scale, energy_shift) -> Tuple[List[nn.Module], List[dict]]:
        cfg: Config = self.cfg
        model_cfg = cfg.model
        n_feature, n_atom_embedding = model_cfg["n_feature"], model_cfg["n_atom_embedding"]
        phys_cfg = cfg.model.physnet
        n_phys_atomic_res, n_phys_interaction_res, n_phys_output_res = phys_cfg["n_phys_atomic_res"], phys_cfg["n_phys_interaction_res"], phys_cfg["n_phys_output_res"]
        dime_cfg = cfg.model.dimenet
        n_dime_before_residual, n_dime_after_residual = dime_cfg["n_dime_before_residual"], dime_cfg["n_dime_after_residual"]
        n_output_dense, n_bi_linear = dime_cfg["n_output_dense"], dime_cfg["n_bi_linear"]
        uncertainty_modify, dropout, batch_norm = model_cfg["uncertainty_modify"], model_cfg["dropout"], model_cfg["batch_norm"]
        loss_metric = cfg.training.loss_fn["loss_metric"]
        last_lin_bias = model_cfg["last_lin_bias"]

        # contents within "[]" will be considered as options
        m_args: dict = option_solver(module_str, type_conversion=True)
        module_str: str = module_str.split("[")[0]
        combination: str = module_str + "_" + bonding_str
        module_info_list: List[dict] = [{"module_str": module_str, "bonding_str": bonding_str, "is_transition": False,
                            "combine_str": "{}_{}".format(module_str, bonding_str)}]
        module_list: List[nn.Module] = []
        if module_str in ["D", "D-noOut"]:
            from Networks.DimeLayers.DimeModule import DimeModule
            n_dime_rbf = self.expansion_info_getter[combination]["n"]
            n_srbf = self.expansion_info_getter[combination]["n_srbf"]
            n_shbf = self.expansion_info_getter[combination]["n_shbf"]
            dim_sbf = n_srbf * (n_shbf + 1)
            this_layer = DimeModule(dim_rbf=n_dime_rbf, dim_sbf=dim_sbf, dim_msg=n_feature, n_output=n_output, n_res_interaction=n_dime_before_residual,
                                            n_res_msg=n_dime_after_residual, n_dense_output=n_output_dense, dim_bi_linear=n_bi_linear, activation=activation,
                                            uncertainty_modify=uncertainty_modify)
            if uncertainty_modify == "concreteDropoutModule":
                this_layer = ConcreteDropout(this_layer, module_type="DimeNet")
            module_list.append(this_layer)
            if self.base_unit_getter[self.previous_module] == "atom":
                module_list.append(AtomToEdgeLayer(n_dime_rbf, n_feature, activation))
                module_info_list.append({"is_transition": True})
        elif module_str in ["P", "P-noOut", "PProt-noOut", "P-PL", "P-PL-noOut", "P-GlobalNode-noOut", "P-PM-noOut"]:
            phys_cls = PhysModuleGlobalNode if module_str == "P-GlobalNode-noOut" else PhysModule
            this_layer = phys_cls(F=n_feature, K=self.expansion_info_getter[combination]["n"], n_output=n_output,
                                    n_res_atomic=n_phys_atomic_res, n_res_interaction=n_phys_interaction_res,
                                    n_res_output=n_phys_output_res, activation=activation, uncertainty_modify=uncertainty_modify,
                                    n_read_out=m_args["n_read_out"] if "n_read_out" in m_args else 0, batch_norm=batch_norm,
                                    dropout=dropout, zero_last_linear=(loss_metric not in ["ce", "bce"]), bias=last_lin_bias, config_dict=cfg, module_str=module_str)
            if uncertainty_modify == "concreteDropoutModule":
                this_layer = ConcreteDropout(this_layer, module_type="PhysNet")
            module_list.append(this_layer)
            if self.base_unit_getter[self.previous_module] == "edge":
                module_list.append(EdgeToAtomLayer())
                module_info_list.append({"is_transition": True})
        elif module_str == "GlobalReadoutPool":
            m_args.update({"n_res_output": n_phys_output_res, "uncertainty_modify": uncertainty_modify})
            module_list.append(GlobalNodeReadoutPooling(activation=activation, n_feature=n_feature, n_output=n_output, **m_args))
            module_info_list[0]["is_transition"] = True
        elif module_str in ["MDN", "MDN-PP", "MDN-PP-intra", "MDN-KANO", "MDN-COMENET", "MDN-Metal-Lig",
                            "MDN-SingleProt"]:
            mdn_cls_mapper = {"MDN": MDNLayer, "MDN-PP": ProtProtMDNLayer, "MDN-SingleProt": ProtSingleMDNLayer,
                              "MDN-PP-intra": ProtProtIntraMDNLayer,  "MDN-Metal-Lig": MetalLigMDNLayer,
                              "MDN-KANO": KanoProtMDNLayer, "MDN-COMENET": ComENetProtMDNLayer}
            mdn_cls = mdn_cls_mapper[module_str]
            mdn_layer = mdn_cls(dropout_rate=0.15, mdn_edge_name=bonding_str,
                                n_atom_types=n_atom_embedding, cfg=cfg, **m_args)
            module_list.append(mdn_layer)
            module_info_list[0]["is_transition"] = True
        elif module_str in ["MDNProp", "NMDN_AuxProp"]:
            cls = {"MDNProp": MDNPropLayer, "NMDN_AuxProp": NMDN_AuxPropLayer}[module_str]
            this_layer = cls(dropout_rate=0.15, mdn_edge_name=bonding_str,
                            n_atom_types=n_atom_embedding, config_dict=cfg)
            module_list.append(this_layer)
            module_info_list[0]["is_transition"] = True
        elif module_str in ["MPNNProp", "MPNNHeteroProp"]:
            cls = MPNNPairedPropLayer
            if module_str == "MPNNHeteroProp": cls = HeteroPairedPropLayer
            this_layer = cls(cfg, bonding_str, activation, **m_args)
            module_list.append(this_layer)
            module_info_list[0]["is_transition"] = True
        elif module_str in ["ProtTF"]:
            this_layer = ProtEmbedTransformLayer(config_dict=cfg, activation=activation, **m_args)
            module_list.append(this_layer)
            module_info_list[0]["is_transition"] = True
        elif module_str.startswith("MTA_"):
            module_info_list[0]["is_transition"] = True
            reduce = module_str.split("MTA_")[-1]
            module_list.append(MartiniToAtomPooling(reduce))
        elif module_str == "GatedEmbed":
            module_info_list[0]["is_transition"] = True
            module_list.append(GatedEmbedLayer())
        elif module_str == "PoolReadout":
            m_args.update({"n_res_output": n_phys_output_res, "uncertainty_modify": uncertainty_modify})
            module_list.append(PoolingReadout(activation=activation, n_feature=n_feature, n_output=n_output, **m_args))
            module_info_list[0]["is_transition"] = True
        elif module_str == "PNAConvPoolReadout":
            m_args.update({"n_res_output": n_phys_output_res, "uncertainty_modify": uncertainty_modify})
            module_list.append(PNAConvPoolReadout(activation=activation, n_feature=n_feature, n_output=n_output, ds=self.ds, **m_args))
            module_info_list[0]["is_transition"] = True
        elif module_str == "ReadoutPool":
            m_args.update({"n_res_output": n_phys_output_res, "uncertainty_modify": uncertainty_modify})
            module_list.append(ReadoutPooling(activation=activation, n_feature=n_feature, n_output=n_output, **m_args))
            module_info_list[0]["is_transition"] = True
        elif module_str == "KANO":
            from Networks.kano.kano4mdn import KanoAtomEmbed
            module_list.append(KanoAtomEmbed(self.cfg))
            module_info_list[0]["is_transition"] = True
        elif module_str == "KANO-Metal":
            from Networks.kano.kano4metal import Kano4Metal
            module_list.append(Kano4Metal(self.cfg))
            module_info_list[0]["is_transition"] = True
        elif module_str == "ComENet":
            if self.cfg["n_output"] == 0:
                module_list.append(ComENetAtomEmbed(self.cfg))
            else:
                module_list.append(ComENetMolProp(self.cfg, energy_scale, energy_shift))
            module_info_list[0]["is_transition"] = True
        elif module_str == "SphereNet":
            module_list.append(SphereNetWrapped(self.cfg))
            module_info_list[0]["is_transition"] = True
        elif module_str == "Comb-P-KANO":
            from Networks.kano.combind_p_kano import CombinedPKano
            module_list.append(CombinedPKano())
            module_info_list[0]["is_transition"] = True
        elif module_str == "EquiformerV2":
            from Networks.SharedLayers.EquiformerV2 import EquiformerV2
            module_list.append(EquiformerV2(cfg))
            module_info_list[0]["is_transition"] = True
        elif module_str == "ESMGearnet":
            from Networks.esm_gearnet.model_wrapper import ESMGearnet
            module_list.append(ESMGearnet())
            module_info_list[0]["is_transition"] = True
        elif module_str in ["C", "D3"]:
            pass
        else:
            error_message(module_str, "module")
        self.previous_module = module_str

        assert len(module_list) == len(module_info_list), f"{len(module_list)} != {len(module_info_list)} :<"

        return module_list, module_info_list
