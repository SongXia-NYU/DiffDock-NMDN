from collections import defaultdict
import logging
import math
import time
from typing import Dict, List, Set, Tuple, Union

import torch
import torch.nn as nn
from torch_scatter import scatter
from Networks.MainModuleFactory import MainModuleFactory

from utils.DataPrepareUtils import cal_msg_edge_index
from Networks.PhysLayers.CoulombLayer import CoulombLayer
from Networks.PhysLayers.D3DispersionLayer import D3DispersionLayer
from Networks.PhysLayers.PhysModule import PhysModule
from Networks.SharedLayers.EmbeddingLayer import EmbeddingLayer
from Networks.SharedLayers.MyMemPooling import MyMemPooling
from utils.BesselCalculator import bessel_expansion_raw
from utils.BesselCalculatorFast import BesselCalculator
from utils.data.data_utils import get_lig_natom, get_lig_z, get_lig_batch, get_lig_coords
from utils.tags import tags
from torch_geometric.data import Dataset, Batch, HeteroData
from utils.time_meta import record_data
from utils.utils_functions import floating_type, dime_edge_expansion, softplus_inverse, \
    gaussian_rbf, info_resolver, expansion_splitter, error_message, option_solver, get_device


class PhysDimeNet(nn.Module):

    def __init__(self, cfg: dict, energy_shift=None, energy_scale=None, ext_atom_dim=0, ds: Dataset=None, **kwargs):
        super().__init__()
        self.cfg: dict = cfg
        self.logger = logging.getLogger()

        # ----------- Regularizations ----------- #
        self.nhlambda: float = cfg["nh_lambda"]
        self.restrain_non_bond_pred: bool = cfg["restrain_non_bond_pred"]
        self.uni_task_ss: bool = cfg["uni_task_ss"]

        # ----------- Normalizations ----------- #
        self.normalize: bool = cfg["normalize"]
        self.shared_normalize_param: bool = cfg["shared_normalize_param"]

        # ----------- Misc ----------- #
        self.coulomb_charge_correct: bool = cfg["coulomb_charge_correct"]
        self.lin_last: bool = cfg["lin_last"]
        if cfg["z_loss_weight"] > 0:
            assert cfg["mask_z"], "If you want to predict Z (atomic number), please mask Z in input"
        self.mask_z = cfg["mask_z"]
        self.z_loss_weight = cfg["z_loss_weight"]
        self.time_debug: bool = cfg["time_debug"]
        self.ligand_only: bool = cfg["ligand_only"]
        self.use_acsf: bool = (cfg["acsf"] is not None)
        if self.use_acsf:
            self.acsf_convert: nn.Module = nn.Linear(cfg["acsf"], cfg["n_feature"])
        self.debug_mode: bool = cfg["debug_mode"]
        self.requires_atom_prop: bool = False
        self.action: str = cfg["action"]
        if self.action in tags.requires_atomic_prop or cfg["w_cross_mdn_pkd"] > 0.:
            self.logger.info("Overwriting self.requires_atom_prop = True because you require atomic props")
            self.requires_atom_prop = True
        self.requires_atom_embedding = cfg["requires_atom_embedding"]
        self.ext_atom_features: str = cfg["ext_atom_features"]
        self.ext_atom_dim: int = ext_atom_dim
        self.trioMPW: bool = cfg["trioMPW"]
        self.cross_mdn_prop_name: str = cfg["cross_mdn_prop_name"]

        self.MDN_MODULE_NAMES = ["MDN", "MDN-PP", "MDN-PP-intra", "MDN-KANO", "MDN-COMENET", "MDN-SingleProt"]

        if cfg["n_output"] == 0:
            # Here we setup a fake n_output to avoid errors in initialization
            # But the sum pooling result will not be used
            self.no_sum_output = True
            self.n_output = 1
        else:
            self.no_sum_output = False
            self.n_output = cfg["n_output"]

        print("------unused keys in model-----")
        for key in kwargs:
            print("{}={}".format(key, kwargs[key]))
        print(">>>>>>>>unused keys in model<<<<<<<<<<")

        self.expansion_fn: Dict[str, str] = expansion_splitter(cfg["expansion_fn"])

        self.activations: List[str] = cfg["activations"].split(" ")
        module_str_list: List[str] = cfg["modules"].split()
        bonding_str_list: List[str] = cfg["bonding_type"].split()
        self.use_mdn: bool = ("MDN" in module_str_list)
        
        # ------------------------------------------------------------------------------------------------------------------#
        # The spaghetti code to parse model, bond and distance expansion information.                                       #
        # main modules, including P (PhysNet), D (DimeNet), etc.                                                            #
        self.main_module_str: List[str] = []                                                                                #                                                                                                                          
        self.main_bonding_str: List[str] = []                                                                               #                                            
        # post modules are either C (Coulomb) or D3 (D3 Dispersion)                                                         #                                                                
        self.post_module_str: List[str] = []                                                                                #                                            
        self.post_bonding_str: List[str] = []                                                                               #                                            
        assert len(module_str_list) == len(bonding_str_list), f"{module_str_list}, {bonding_str_list}"                      #                                                                                                    
        for this_module_str, this_bonding_str in zip(module_str_list, bonding_str_list):                                    #                                                                                        
            '''                                                                                                             #            
            Separating main module and post module                                                                          #                                                
            '''                                                                                                             #            
            this_module_str = this_module_str.split('[')[0]                                                                 #                                                        
            if this_module_str in ['C', 'D3']:                                                                              #                                            
                self.post_module_str.append(this_module_str)                                                                #                                                            
                self.post_bonding_str.append(this_bonding_str)                                                              #                                                            
            else:                                                                                                           #                
                self.main_module_str.append(this_module_str)                                                                #                                                            
                self.main_bonding_str.append(this_bonding_str)                                                              #                                                            
        self.bonding_type_keys = set([s for s in bonding_str_list if s.lower() != "none"])                                  #                                                                                        
                                                                                                                            #
        self.expansion_keys: Set[str] = set()                                                                               #                                            
        self.msg_bond_type: Set[str] = set()                                                                                #                                            
        for module, bond in zip(module_str_list, bonding_str_list):                                                         #                                                                
            module = module.split('[')[0]                                                                                   #                                        
            if module in ["D", "D-noOut"]:                                                                                  #                                        
                self.msg_bond_type.add(bond)                                                                                #                                            
            if bond.lower() == "none" or module in \
                ["MDN", "MDN-PP", "MDNProp", "MPNNProp", "ProtTF", "MDN-KANO", "KANO",                                      #
                "Comb-P-KANO", "MDN-COMENET", "NMDN_AuxProp", "MDN-PP-intra", "MPNNHeteroProp",
                "ESMGearnet", "MDN-SingleProt"]:                                              #                                                                                            
                continue                                                                                                    #                        
            self.expansion_keys.add('{}_{}'.format(module, bond))                                                           #                                                                    
                                                                                                                            #
        # A dictionary which parses an expansion combination into detailed information                                      #                                                                                    
        self.expansion_info_getter: Dict[str, Dict[str, str]] = {                                                           #                                                                
            combination: info_resolver(self.expansion_fn[combination])                                                      #                                                                    
            for combination in self.expansion_keys                                                                          #                                                
        }                                                                                                                   #
                                                                                                                            #
        self.base_unit_getter = defaultdict(lambda: "atom")                                                                 #
        # DimeNet is operating on edge representation, others are operating on atom representation.                         #
        self.base_unit_getter["D"] = "edge"                                                                                 #
        self.base_unit_getter["D-noOut"] = "edge"                                                                           #        
        # ------------------------------------------------------------------------------------------------------------------#

        self.register_expansion_parms()

        self.dist_calculator = nn.PairwiseDistance(keepdim=True)

        self.embedding_layer = EmbeddingLayer(cfg["n_atom_embedding"], cfg["n_feature"] - self.ext_atom_dim)
        
        self.register_main_modules(module_str_list, bonding_str_list, cfg, ds, energy_scale, energy_shift)

        self.register_post_modules()

        self.register_norm_params(energy_shift, energy_scale)

        pool_base, pool_options = option_solver(cfg["pooling"], type_conversion=True, return_base=True)
        if pool_base == "sum":
            self.pooling_module = None
        elif pool_base == "mem_pooling":
            self.pooling_module = MyMemPooling(in_channels=cfg["n_feature"], **pool_options)
        else:
            raise ValueError(f"invalid pool_base: {pool_base}")

        if self.lin_last:
            # extra safety
            assert self.requires_atom_embedding
            assert isinstance(self.main_module_list[-1], PhysModule)
        self.runtime_vars_checked: bool = False

    def register_main_modules(self, module_str_list, bonding_str_list, config_dict, ds, energy_scale, energy_shift):
        '''
        registering main modules.
        '''
        self.main_module_list = nn.ModuleList()
        # stores extra info including module type, bonding type, etc
        self.main_module_info: List[dict] = []
        module_factory = MainModuleFactory(config_dict, self.expansion_info_getter, self.base_unit_getter, ds)
        for i, (module_str, bonding_str) in enumerate(zip(module_str_list, bonding_str_list)):
            activation: str = self.activations[i]
            module_list, module_info_list = module_factory.get_module(module_str, bonding_str, activation, self.n_output, energy_scale, energy_shift)
            # special logics for MDN layer
            if module_str.split("[")[0] in self.MDN_MODULE_NAMES:
                self.mdn_layer = module_list[0]
                module_list = [None]
                for module_str in module_str_list[:i]:
                    if module_str.startswith("MTA_"):
                        self.mdn_layer.martini2aa_pooling = True
                        break
            self.main_module_list.extend(module_list)
            self.main_module_info.extend(module_info_list)

    def register_norm_params(self, energy_shift, energy_scale):
        if not self.normalize:
            return
        
        '''
        Atom-wise shift and scale, used in PhysNet
        '''
        n_atom_embedding = self.cfg["n_atom_embedding"]
        if self.cfg["uni_task_ss"]:
            ss_dim = 1
        else:
            ss_dim = self.n_output
        if self.cfg["shared_normalize_param"]:
            shift_matrix = torch.zeros(n_atom_embedding, ss_dim).type(floating_type)
            scale_matrix = torch.zeros(n_atom_embedding, ss_dim).type(floating_type).fill_(1.0)
            if energy_shift is not None:
                if isinstance(energy_shift, torch.Tensor):
                    shift_matrix[:, :] = energy_shift.view(1, -1)[:, :ss_dim]
                else:
                    shift_matrix[:, 0] = energy_shift
            if energy_scale is not None:
                if isinstance(energy_scale, torch.Tensor):
                    scale_matrix[:, :] = energy_scale.view(1, -1)[:, :ss_dim]
                else:
                    scale_matrix[:, 0] = energy_scale
            shift_matrix = shift_matrix / len(self.bonding_type_keys)
            self.register_parameter('scale', torch.nn.Parameter(scale_matrix, requires_grad=True))
            self.register_parameter('shift', torch.nn.Parameter(shift_matrix, requires_grad=self.cfg["train_shift"]))
        else:
            for key in self.bonding_type_keys:
                shift_matrix = torch.zeros(n_atom_embedding, ss_dim).type(floating_type)
                scale_matrix = torch.zeros(n_atom_embedding, ss_dim).type(floating_type).fill_(1.0)
                if energy_shift is not None:
                    shift_matrix[:, 0] = energy_shift
                if energy_scale is not None:
                    scale_matrix[:, 0] = energy_scale
                shift_matrix = shift_matrix / len(self.bonding_type_keys)
                self.register_parameter('scale{}'.format(key), torch.nn.Parameter(scale_matrix, requires_grad=True))
                self.register_parameter('shift{}'.format(key),
                                        torch.nn.Parameter(shift_matrix, requires_grad=self.cfg["train_shift"]))

    def register_post_modules(self):
        # TODO Post modules to list
        for i, (module_str, bonding_str) in enumerate(zip(self.post_module_str, self.post_bonding_str)):
            if module_str == 'C':
                combination = module_str + "_" + bonding_str
                self.add_module('post_module{}'.format(i),
                                CoulombLayer(cutoff=self.expansion_info_getter[combination]['dist']))
            elif module_str == 'D3':
                self.add_module('post_module{}'.format(i), D3DispersionLayer(s6=0.5, s8=0.2130, a1=0.0, a2=6.0519))
            else:
                error_message(module_str, 'module')

    def register_expansion_parms(self):
        # registering necessary parameters for some expansions if needed
        for combination in self.expansion_fn.keys():
            expansion_fn_info = self.expansion_info_getter[combination]
            if expansion_fn_info['name'] == "gaussian":
                n_rbf = expansion_fn_info['n']
                feature_dist = expansion_fn_info['dist']
                feature_dist = torch.as_tensor(feature_dist).type(floating_type)
                self.register_parameter('cutoff' + combination, torch.nn.Parameter(feature_dist, False))
                expansion_coe = torch.as_tensor(expansion_fn_info["coe"]).type(floating_type)
                self.register_parameter('expansion_coe' + combination, torch.nn.Parameter(expansion_coe, False))
                # Centers are params for Gaussian RBF expansion in PhysNet
                dens_min = expansion_fn_info["dens_min"]
                centers = softplus_inverse(torch.linspace(math.exp(-dens_min), math.exp(-feature_dist * expansion_coe), n_rbf))
                centers = torch.nn.functional.softplus(centers)
                self.register_parameter('centers' + combination, torch.nn.Parameter(centers, True))

                # Widths are params for Gaussian RBF expansion in PhysNet
                widths = [softplus_inverse((0.5 / ((1.0 - torch.exp(-feature_dist)) / n_rbf)) ** 2)] * n_rbf
                widths = torch.as_tensor(widths).type(floating_type)
                widths = torch.nn.functional.softplus(widths)
                self.register_parameter('widths' + combination, torch.nn.Parameter(widths, True))
            elif expansion_fn_info['name'] == 'defaultDime':
                n_srbf = self.expansion_info_getter[combination]['n_srbf']
                n_shbf = self.expansion_info_getter[combination]['n_shbf']
                envelop_p = self.expansion_info_getter[combination]['envelop_p']
                setattr(self, f"bessel_calculator_{n_srbf}_{n_shbf}", BesselCalculator(n_srbf, n_shbf, envelop_p))

    def freeze_prev_layers(self, freeze_extra=False):
        if freeze_extra:
            # Freeze scale, shift and Gaussian RBF parameters
            for param in self.parameters():
                param.requires_grad_(False)
        for i in range(len(self.main_module_str)):
            self.main_module_list[i].freeze_prev_layers()
            self.main_module_list[i].output.freeze_residual_layers()

    def forward(self, data):
        # torch.cuda.synchronize(device=device)
        t0 = time.time()
        data = self.proc_data_dtype(data)

        atom_mol_batch = get_lig_batch(data)
        
        edge_index_getter, edge_type_getter = self.gather_edge_info(data)
        if self.time_debug:
            t0 = record_data("bond_setup", t0)

        msg_edge_index_getter = self.gather_msg_edge_info(data, edge_index_getter)
        if self.time_debug:
            t0 = record_data("msg_bond_setup", t0)

        expansions = self.gather_expansion_info(data, edge_index_getter, msg_edge_index_getter)
        if self.time_debug:
            t0 = record_data("expansion_prepare", t0)

        # runtime_vars stores runtime embeddings as well as auxilliary variables that is used by modules
        # it is updated by the modules every run. 
        runtime_vars: dict = self.gather_init_embeddings(data)
        if self.time_debug:
            t0 = record_data("embedding_prepare", t0)

        output, runtime_vars, pred_last_by_bond, pred_sum_by_bond, nh_loss = \
            self.run_main_modules(runtime_vars, edge_index_getter, msg_edge_index_getter, expansions, edge_type_getter)
        if self.time_debug:
            t0 = record_data("main_modules", t0)

        pred_sum_by_bond = self.atom_lvl_norm(pred_sum_by_bond, data)
        if self.time_debug:
            t0 = record_data("normalization", t0)

        # predicted property at atomic level
        atom_prop = 0.
        for key in self.bonding_type_keys:
            atom_prop = atom_prop + pred_sum_by_bond[key]

        atom_prop = self.run_post_modules(data, edge_index_getter, expansions, atom_prop)
        if self.time_debug:
            t0 = record_data("post_modules", t0)

        if self.restrain_non_bond_pred and ('N' in self.bonding_type_keys):
            # Bonding energy should be larger than non-bonding energy
            atom_prop2 = atom_prop ** 2
            non_bond_prop2 = pred_sum_by_bond['N'] ** 2
            nh_loss = nh_loss + torch.mean(non_bond_prop2 / (atom_prop2 + 1e-7)) * self.nhlambda
        output["nh_loss"] = nh_loss

        # only sum over ligand atoms
        # 0: protein; 1: ligand
        if self.ligand_only:
            atom_prop = torch.where(data.mol_type.view(-1, 1) == 1, atom_prop,
                                    torch.as_tensor([0.]).type(floating_type).to(get_device()))
        
        if isinstance(atom_prop, torch.Tensor):
            # Total prediction is the summation of bond and non-bond prediction
            mol_prop = scatter(reduce='sum', src=atom_prop, index=atom_mol_batch, dim=0)
        else:
            N = get_lig_natom(data)
            # set it to zero and worry about it later
            mol_prop = torch.zeros_like(N)

        # only runs when you want to predict charge and dipole
        if self.action in ["E", "names_and_QD"]:
            output, mol_prop = self.qd_modification(data, output, mol_prop, atom_prop)

        if self.pooling_module is not None:
            output["mol_prop_pool"] = self.pooling_module(atom_mol_batch=atom_mol_batch, vi=runtime_vars["vi"])

        if self.requires_atom_prop:
            output["atom_prop"] = atom_prop
            output["atom_mol_batch"] = atom_mol_batch

        for prop_name in [self.cross_mdn_prop_name, "pair_atom_prop"]:
            if prop_name in runtime_vars:
                output[prop_name] = runtime_vars[prop_name]

        if self.requires_atom_embedding:
            output = self.infuse_atom_embed(data, output, runtime_vars)

        if not self.runtime_vars_checked:
            self.check_runtime_vars(runtime_vars, atom_prop)
            self.runtime_vars_checked = True
        output = self.record_mol_prop(data, output, mol_prop, runtime_vars)
        if self.time_debug:
            t0 = record_data("scatter_pool_others", t0)

        return output
    
    def atom_lvl_norm(self, pred_sum_by_bond, data_batch):
        """
        Atom-level scale and shift for regression tasks
        """
        if self.lin_last or not self.normalize:
            return pred_sum_by_bond

        Z = get_lig_z(data_batch)
        if self.mask_z:
            Z = torch.zeros_like(Z)

        if self.shared_normalize_param:
            for key in self.bonding_type_keys:
                pred_sum_by_bond[key] = self.scale[Z, :] * pred_sum_by_bond[key] + self.shift[Z, :]
        else:
            for key in self.bonding_type_keys:
                pred_sum_by_bond[key] = getattr(self, 'scale{}'.format(key))[Z, :] * pred_sum_by_bond[key] + \
                                            getattr(self, 'shift{}'.format(key))[Z, :]
        return pred_sum_by_bond
    
    def run_main_modules(self, runtime_vars: dict, edge_index_getter, msg_edge_index_getter, expansions, edge_type_getter):
        '''
        Going through main modules
        '''
        # output gathers all information needed for evaluation
        output: dict = {}
        # the latest prediction of the model, separated by bond type
        pred_last_by_bond: Dict[str, torch.Tensor] = {key: None for key in self.bonding_type_keys}
        # the summation of model prediction of the model, separated by bond type
        pred_sum_by_bond: Dict[str, torch.Tensor] = {key: 0. for key in self.bonding_type_keys}

        # non-hierachical loss
        nh_loss: torch.Tensor = torch.zeros(1).type(floating_type).to(get_device())
        for i, (info, this_module) in enumerate(zip(self.main_module_info, self.main_module_list)):
            if info["module_str"] in self.MDN_MODULE_NAMES:
                mdn_out = self.mdn_layer(runtime_vars)
                output.update(mdn_out)
                continue

            runtime_vars["info"] = info
            # grab edge_index and edge_attr information
            if not info["is_transition"]:
                runtime_vars["edge_index"] = edge_index_getter[info["bonding_str"]]
                runtime_vars["edge_attr"] = expansions[info["combine_str"]]
                if self.trioMPW:
                    runtime_vars["edge_type"] = edge_type_getter[info["bonding_str"]]
            # grab edge-edge edges (edge-level edges) for DimeNet
            if info["module_str"].split("-")[0] == "D":
                runtime_vars["msg_edge_index"] = msg_edge_index_getter[info["bonding_str"]]

            runtime_vars = this_module(runtime_vars)

            # clearing edge_index and edge_attr from runtime_vars
            runtime_vars["edge_index"] = None
            runtime_vars["edge_attr"] = None

            # first layer embedding will be used to predict atomic number (Z)
            if self.z_loss_weight > 0 and i == 0:
                output["first_layer_vi"] = runtime_vars["vi"]

            # due to some bugs in previous code, I added "legacy_exps" for reproducibility of legacy experiments
            if (info["is_transition"] or info["module_str"].split("-")[-1] == 'noOut') and \
                    not self.cfg["legacy_exps"]:
                continue

            # calculate non-hierachical loss
            nh_loss = nh_loss + runtime_vars["regularization"]
            # record output by bond-type
            if pred_last_by_bond[info["bonding_str"]] is not None:
                # Calculating non-hierarchical penalty
                out2 = runtime_vars["out"] ** 2
                last_out2 = pred_last_by_bond[info["bonding_str"]] ** 2
                nh_loss = nh_loss + torch.mean(out2 / (out2 + last_out2 + 1e-7)) * self.nhlambda
            pred_last_by_bond[info["bonding_str"]] = runtime_vars["out"]
            pred_sum_by_bond[info["bonding_str"]] = pred_sum_by_bond[info["bonding_str"]] + runtime_vars["out"]
        
        return output, runtime_vars, pred_last_by_bond, pred_sum_by_bond, nh_loss
    
    def gather_init_embeddings(self, data_batch):
        """
        Get the initial embeddings as the model input
        """
        mji = None
        '''
        mji: edge mol_lvl_detail
        vi:  node mol_lvl_detail
        '''
        Z = get_lig_z(data_batch)
        if self.mask_z:
            Z = torch.zeros_like(Z)
        if self.use_acsf:
            vi_init = data_batch.acsf_features.type(floating_type)
            vi_init = self.acsf_convert(vi_init)
        else:
            vi_init = self.embedding_layer(Z)
            if self.ext_atom_features is not None:
                vi_init = torch.cat([vi_init, getattr(data_batch, self.ext_atom_features)], dim=-1)
        runtime_vars = {"vi": vi_init, "mji": mji, "data_batch": data_batch}
        if self.cfg["prot_embedding_root"] is not None:
            runtime_vars["prot_embed"] = data_batch.prot_embed
            runtime_vars["prot_embed_chain1"] = getattr(data_batch, "prot_embed_chain1", None)
            runtime_vars["prot_embed_chain2"] = getattr(data_batch, "prot_embed_chain2", None)
        return runtime_vars

    def gather_edge_info(self, data_batch: Union[Batch, dict]):
        """
        Given a data_batch, gather all required edge_index infomation for the model to train.

        bonding_str is a string representing the bond type. I use B for non-bonding, N for bonding, BN for combined non-bonding and bonding. 
            P for protein-protein, L for ligand-ligand, PL for protein-ligand
        """
        edge_index_getter: Dict[str, torch.LongTensor] = {}
        if isinstance(data_batch, dict):
            # data_batch is a dict when using ESM-GearNet
            if "BN" in self.bonding_type_keys:
                edge_index_getter["BN"] = data_batch["ligand"][("ligand", "interaction", "ligand")].edge_index
            return edge_index_getter, {}
        
        # heterogenous data
        if isinstance(data_batch.get_example(0), HeteroData):
            for key in self.bonding_type_keys:
                if key == "BN":
                    edge_index_getter["BN"] = data_batch[("ligand", "interaction", "ligand")].edge_index
                    continue
                if key == "PL_min_dist_sep_oneway":
                    edge_index_getter["PL_min_dist_sep_oneway"] = data_batch[("ligand", "interaction", "protein")].edge_index
                    continue
                assert key.startswith("("), key
                parsed_key = parse_hetero_edge(key)
                edge_index_getter[key] = data_batch[parsed_key].min_dist_edge_index
                # protein-protein interaction: to save disk, only one-way interaction is saved
                # for proper message-passing, the edge_index is converted to two-way edge here
                if parsed_key == ("protein", "interaction", "protein"):
                    edge_index_getter[key] = torch.concat([edge_index_getter[key], edge_index_getter[key][[1, 0], :]], dim=-1)
            return edge_index_getter, {}

         # edge_type is used in protein-ligand systems, 0 means protein-protein, 1: protein-ligand, 2: ligand-ligand
        edge_type_getter = {}
        pl_type_mapper = {"PROTEIN": 0, "PL": 1, "LIGAND": 2}

        # we now support bonding type separated by '+' to combine multiple edges
        for bonding_str in self.bonding_type_keys:
            edge_index = getattr(data_batch, bonding_str + '_edge_index', False)
            if edge_index is not False:
                edge_index_getter[bonding_str] = edge_index
            else:
                edge_indices = [data_batch[t + '_edge_index'] for t in bonding_str.split("+")]
                if self.trioMPW:
                    this_edge_types = []
                    for this_edge_index, edge_name in zip(edge_indices, bonding_str.split("+")):
                        this_edge_type = torch.zeros_like(this_edge_index[[0], :]).fill_(pl_type_mapper[edge_name]).type(floating_type)
                        this_edge_types.append(this_edge_type)
                    edge_type_getter[bonding_str] = torch.cat(this_edge_types, dim=-1).view(-1, 1)
                edge_index_getter[bonding_str] = torch.cat(edge_indices, dim=-1)
            if bonding_str == "PROTEIN_Voronoi1":
                edge_index_getter[bonding_str] = edge_index_getter[bonding_str].long()
        return edge_index_getter, edge_type_getter
    
    def gather_msg_edge_info(self, data_batch, edge_index_getter):
        """
        Gather the 'msg_edge' (edge of edges) information. It is required by DimeNet models.
        """
        msg_edge_index_getter = {}
        for bonding_str in self.msg_bond_type:
            # prepare msg edge index
            this_msg_edge_index = getattr(data_batch, bonding_str + '_msg_edge_index', False)
            if this_msg_edge_index is not False:
                msg_edge_index_getter[bonding_str] = this_msg_edge_index
            else:
                msg_edge_index_getter[bonding_str] = cal_msg_edge_index(edge_index_getter[bonding_str]).to(get_device())
        return msg_edge_index_getter
    
    def run_post_modules(self, data_batch: Union[Batch, dict], edge_index_getter: dict, expansions, atom_prop):
        '''
        Post modules: Coulomb or D3 Dispersion layers
        '''
        Z = get_lig_z(data_batch)
        if self.mask_z:
            Z = torch.zeros_like(Z)
        for i, (module_str, bonding_str) in enumerate(zip(self.post_module_str, self.post_bonding_str)):
            this_edge_index = edge_index_getter[bonding_str]
            this_expansion = expansions["{}_{}".format(module_str, bonding_str)]
            if module_str == 'C':
                if self.coulomb_charge_correct:
                    Q = data_batch.Q
                    N = get_lig_natom(data_batch)
                    coulomb_correction = self._modules["post_module{}".format(i)](atom_prop[:, -1],
                                                                                  this_expansion["pair_dist"],
                                                                                  this_edge_index, q_ref=Q, N=N,
                                                                                  atom_mol_batch=data_batch.atom_mol_batch)
                else:
                    # print("one of the variables needed for gradient computation has been modified by an inplace"
                    #       " operation: need to be fixed here, probably in function cal_coulomb_E")
                    coulomb_correction = self._modules["post_module{}".format(i)](atom_prop[:, -1],
                                                                                  this_expansion['pair_dist'],
                                                                                  this_edge_index)
                atom_prop[:, 0] = atom_prop[:, 0] + coulomb_correction
            elif module_str == 'D3':
                d3_correction = self._modules["post_module{}".format(i)](Z, this_expansion, this_edge_index)
                atom_prop[:, 0] = atom_prop[:, 0] + d3_correction
            else:
                error_message(module_str, 'module')
        return atom_prop
    
    def gather_expansion_info(self, data_batch: Union[Batch, dict], edge_index_getter: dict, msg_edge_index_getter: dict):
        def compute_dist(data_batch: Union[Batch, dict], bond_type: str, edge_index_getter: dict):
            if isinstance(data_batch, dict):
                data_batch = data_batch["ligand"]
            d0 = data_batch.get_example(0)
            if isinstance(d0, HeteroData) and bond_type != "BN":
                parsed_key = parse_hetero_edge(bond_type)
                dist: torch.Tensor = data_batch[parsed_key].min_dist
                if parsed_key == ("protein", "interaction", "protein"):
                    dist = torch.concat([dist, dist], dim=0)
                return dist
            if hasattr(data_batch, f"{bond_type}_dist"):
                return getattr(data_batch, f"{bond_type}_dist").view(-1, 1)
            this_edge_index = edge_index_getter[bond_type]
            pair_dist = self.dist_calculator(lig_coords[this_edge_index[0, :], :], lig_coords[this_edge_index[1, :], :])
            return pair_dist
        
        lig_coords = get_lig_coords(data_batch)
        expansions = {}
        '''
        calculating expansion
        '''
        for combination in self.expansion_keys:
            module_str = combination.split('_')[0]
            bond_type = "_".join(combination.split('_')[1:])
            expansion_info = self.expansion_info_getter[combination]
            this_expansion = expansion_info['name']
            if (module_str == 'C') or (module_str == 'D3'):
                '''
                In this situation, we only need to calculate pair-wise distance.
                '''
                expansions[combination] = {"pair_dist": compute_dist(data_batch, bond_type, edge_index_getter)}
                continue

            # DimeNet, calculate sbf and rbf
            if this_expansion == "defaultDime":
                n_srbf = expansion_info['n_srbf']
                n_shbf = expansion_info['n_shbf']
                expansions[combination] = dime_edge_expansion(lig_coords, edge_index_getter[bond_type],
                                                                msg_edge_index_getter[bond_type],
                                                                expansion_info['n'],
                                                                self.dist_calculator,
                                                                getattr(self, f"bessel_calculator_{n_srbf}_{n_shbf}"),
                                                                expansion_info['dist'],
                                                                return_dict=True)
            # PhysNet, calculate rbf
            elif this_expansion == 'bessel':
                dist_atom = compute_dist(data_batch, bond_type, edge_index_getter)
                rbf = bessel_expansion_raw(dist_atom, expansion_info['n'],
                                            expansion_info['dist'])
                expansions[combination] = {"rbf": rbf}
            elif this_expansion == 'gaussian':
                pair_dist = compute_dist(data_batch, bond_type, edge_index_getter)
                expansions[combination] = gaussian_rbf(pair_dist, getattr(self, 'centers' + combination),
                                                        getattr(self, 'widths' + combination),
                                                        getattr(self, 'cutoff' + combination),
                                                        getattr(self, 'expansion_coe' + combination),
                                                        return_dict=True, linear=expansion_info["linear"])
        return expansions
    
    def qd_modification(self, data_batch, output, mol_prop, atom_prop):        
        assert self.n_output > 1
        # the last dimension is considered as atomic charge prediction
        Q_pred = mol_prop[:, -1]
        Q_atom = atom_prop[:, -1]
        atom_prop = atom_prop[:, :-1]
        D_atom = Q_atom.view(-1, 1) * get_lig_coords(data_batch)
        D_pred = scatter(reduce='sum', src=D_atom, index=get_lig_batch(data_batch), dim=0)
        output["Q_pred"] = Q_pred
        output["D_pred"] = D_pred
        if self.requires_atom_prop:
            output["Q_atom"] = Q_atom
        mol_prop = mol_prop[:, :-1]
        return output, mol_prop
    
    def infuse_atom_embed(self, data_batch, output, runtime_embed_dynamic):
        if self.normalize:
            Z = get_lig_z(data_batch)
            if self.mask_z:
                Z = torch.zeros_like(Z)
            dim_ss = self.scale.shape[-1]
            output["atom_embedding"] = torch.cat([runtime_embed_dynamic["embed_b4_ss"] * self.scale[Z, i].reshape(-1, 1)
                                                    for i in range(dim_ss)], dim=-1)
            output["atom_embedding_ss"] = torch.cat([runtime_embed_dynamic["embed_b4_ss"] * self.scale[Z, i].reshape(-1, 1)
                                                        + self.shift[Z, i].reshape(-1, 1)
                                                        for i in range(dim_ss)], dim=-1)
        else:
            output["atom_embedding"] = runtime_embed_dynamic["embed_b4_ss"]
            output["atom_embedding_ss"] = runtime_embed_dynamic["embed_b4_ss"]
        return output
    
    def check_runtime_vars(self, runtime_vars: dict, atom_prop: Union[float, torch.Tensor]):
        """
        Check the keys in runtime_vars to make sure only one "mol_prop" is set.
        "mol_prop" may be explicitly set by modules, for example: MPNNPairedPropLayer, ReadoutPooling and PoolingReadout layer.
        If "mol_prop" is not explicitly set, then atom_prop should be set.
        """
        if hasattr(self, "mdn_layer"):
            return
        
        mol_prop_counter: int = 0
        for key in runtime_vars.keys():
            if "mol_prop" in key:
                mol_prop_counter += 1
        assert mol_prop_counter <= 1, str(list(runtime_vars.keys()))

        if mol_prop_counter == 0:
            assert isinstance(atom_prop, torch.Tensor), atom_prop.__class__
        else:
            assert isinstance(atom_prop, float), atom_prop.__class__
    
    def record_mol_prop(self, data_batch, output, mol_prop, runtime_vars):
        # Record MDNPropLayer output
        # Since the keys are already checked during self.check_runtime_vars, we can safely assume only one "mol_prop" is predicted. 
        for key in runtime_vars.keys():
            if "mol_prop" in key:
                output["mol_prop"] = runtime_vars[key]
                return output

        if self.lin_last:
            raise ValueError("lin_last is no longer used.")
        elif self.no_sum_output:
            output["mol_prop"] = torch.Tensor(torch.Size([mol_prop.shape[0], 0])).to(mol_prop.device)
        else:
            output["mol_prop"] = mol_prop
        return output

    def proc_data_dtype(self, maybe_data_batch: Union[Batch, dict]) -> Union[Batch, dict]:
        # data_batch is a dict when using ESM-GearNet
        if isinstance(maybe_data_batch, dict):
            maybe_data_batch["ligand"] = self.proc_data_dtype(maybe_data_batch["ligand"])
            return maybe_data_batch

        data_batch = maybe_data_batch
        d0 = data_batch.get_example(0)
        def _correct_dtype(t):
            if not isinstance(t, torch.Tensor):
                return t
            if t.dtype == torch.float64:
                return t.type(floating_type)
            return t
        
        if isinstance(d0, HeteroData):
            return data_batch.apply(_correct_dtype)

        for key in data_batch.keys:
            prop = getattr(data_batch, key)
            if not isinstance(prop, torch.Tensor):
                continue
            # temp fix
            if prop.dtype == torch.float64:
                setattr(data_batch, key, prop.type(floating_type))
        return maybe_data_batch

def parse_hetero_edge(edge_name: str) -> Tuple[str, str, str]:
    # eg. "(ion, interaction, protein)" -> ("ion", "interaction", "protein")
    edge_name = edge_name.strip("()")
    src, interaction, dst = edge_name.split(",")
    src, interaction, dst = src.strip(), interaction.strip(), dst.strip()
    return (src, interaction, dst)
