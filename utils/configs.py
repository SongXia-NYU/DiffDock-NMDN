from dataclasses import dataclass, field
from typing import List, Optional, Union
from glob import glob
import os.path as osp

from omegaconf import OmegaConf, MISSING

@dataclass
class ModelConfig:
    # eg: D P D P D P, D for DimeNet and P for PhysNet
    modules: str = MISSING
    # eg: B N B N B N, B for bonding-edge, N for non-bonding edge, L for long-range interaction and 
    # BN for both bonding and non-bonding
    bonding_type: str = MISSING
    activations: str = MISSING
    expansion_fn: str = MISSING
    n_feature: int = MISSING
    cutoffs: str = MISSING
    n_atom_embedding: int = 95
    # none | concreteDropoutModule | concreteDropoutOutput | swag_${start}_${freq}
    uncertainty_modify: str = "none"
    # calculate charge correction when calculation Coulomb interaction
    coulomb_charge_correct: bool = False
    # sum | mem_pooling[heads=?,num_clusters=?,tau=?,n_output=?]
    pooling: str = "sum"
    n_output: int = 1
    use_trained_model: Optional[str] = None
    # Use shadow model (best model) to initialize both training and shadow model 
    # This is helpful when you want to freeze some parameters without messing up the weights by SWA
    ft_discard_training_model: bool = False
    reset_optimizer: bool = True
    reset_output_layers: bool = False
    reset_scale_shift: bool = False
    reset_ptn: List[str] = field(default_factory=lambda: [])
    batch_norm: bool = False
    dropout: bool = False
    requires_atom_embedding: bool = False
    last_lin_bias: bool = False

    @dataclass
    class PhysNetConfig:
        n_phys_atomic_res: int = 1
        n_phys_interaction_res: int = 1
        n_phys_output_res: int = 1
    physnet: PhysNetConfig = field(default_factory=PhysNetConfig)

    @dataclass 
    class MDNConfig:
        n_mdn_hidden: Optional[int] = None
        n_mdn_lig_metal_hidden: Optional[int] = None
        n_mdnprop_hidden: Optional[int] = None
        # Number of MLP layer in the MDN layer
        n_mdn_layers: int = 1
        n_mdnprop_layers: int = 1
        mdn_threshold_train: Optional[float] = None
        mdn_threshold_eval: Optional[float] = None
        mdn_threshold_prop: Optional[float] = None
        mdn_voronoi_edge: bool = False
        # the distance expansion function for MDN paired properties prediction
        mdn_dist_expansion: Optional[str] = None
        pair_prop_dist_coe: Optional[str] = None
        n_mdn_gauss: int = 10
        pkd_phys_terms: Optional[str] = None
        pkd_phys_concat: bool = False
        pkd_phys_norm: Optional[float] = None
        auxprop_nmdn_name: str = "MDN_LOGSUM"
        auxprop_nmdn_compute_ref: bool = False
        protprot_exclude_edge: Optional[int] = None
        n_paired_mdn_readout: int = 1
        n_paired_mdn_readout_hidden: Optional[int] = None
        metal_atom_embed_path: Optional[str] = None
        metal_atom_embed_slice: Optional[int] = None
        w_lig_metal: float = 1.0
        mdn_freeze_bn: bool = False
        # regularize pair MDN prob by pair distance: inverse | inverse_square
        val_pair_prob_dist_coe: Optional[str] = None
        hist_pp_intra_mdn: bool = False
        nmdn_eval: bool = False
        compute_external_mdn: bool = True
    mdn: MDNConfig = field(default_factory=MDNConfig)

    @dataclass
    class DimeNetConfig:
        n_dime_before_residual: int = 1
        n_dime_after_residual: int = 2
        n_output_dense: int = 3
        n_bi_linear: int = 8
    dimenet: DimeNetConfig = field(default_factory=DimeNetConfig)

    @dataclass
    class NormalizationConfig:
        normalize: bool = True
        train_shift: bool = True
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)

    @dataclass
    class KANOConfig:
        kano_ckpt: Optional[str] = None
    kano: KANOConfig = field(default_factory=KANOConfig)

    @dataclass
    class ComENetConfig:
        comenet_cutoff: float = 8.
        comenet_num_layers: int = 4
        comenet_num_radial: int = 3
        comenet_num_spherical: int = 2
        comenet_num_output_layers: int = 3
    comenet: ComENetConfig = field(default_factory=ComENetConfig)

@dataclass
class TrainingConfig:
    batch_size: int = MISSING
    valid_batch_size: int = MISSING
    num_epochs: int = 1000
    # early stopping, set to -1 to disable
    early_stop: int = -1
    stop_low_lr: bool = False
    # emaAms_${ema} | sgd
    optimizer: str = "emaAms_0.999"
    learning_rate: float = 1e-3
    max_norm: float = 1000.
    error_if_nonfinite: bool = False
    swa_use_buffers: bool = False
    swa_start_step: int = 0
    # StepLR | ReduceLROnPlateau
    scheduler: str = "ReduceLROnPlateau[factor=0.3,patience=40]"
    normal_lr_ptn: List[str] = field(default_factory=lambda: [])
    lower_lr_ptn: List[str] = field(default_factory=lambda: [])
    ft_lr_factor: Optional[float] = None
    eval_per_step: Optional[int] = None

    @dataclass
    class LossFnConfig:
        # mae|rmse|mse|ce|bce|evidential|mdn|mdn_mae
        loss_metric: str = "mae"
        w_mdn: float = 1.
        w_regression: float = 1.
        mdn_w_lig_atom_types: float = 0.
        mdn_w_prot_atom_types: float = 0.
        mdn_w_lig_atom_props: float = 0.
        mdn_w_prot_sasa: float = 0.
        keep: Optional[str] = None
        l2lambda: float = 0.
        force_weight: float = 1.
        charge_weight: float = 1.
        dipole_weight: float = 1.
        action: str = "names"
        target_names: List[str] = field(default_factory=lambda: [])
        regression_ignore_nan: bool = False
        auto_sol: bool = False
        auto_sol_no_conv: bool = False
        target_nodes: Union[bool, int] = False
        auto_pl_water_ref: bool = False   
        no_pkd_score: bool = False 
    loss_fn: LossFnConfig = field(default_factory=LossFnConfig)

@dataclass
class DataConfig:
    data_provider: str = MISSING
    over_sample: bool = False
    data_root: str = "../dataProviders/data"
    dataset_name: Union[str, None] = None
    dataset_names: Optional[List[str]] = None
    split: Optional[str] = None
    diffdock_nmdn_result: Optional[List[str]] = None
    diffdock_confidence: bool = False
    valid_size: Optional[int] = None
    split_seed: int = 2333
    proc_in_gpu: bool = True
    cache_bonds: bool = False
    debug_mode_n_train: int = 1000
    debug_mode_n_val: int = 100
    dynamic_batch: bool = False
    dynamic_batch_max_num: Optional[int] = None
    kano_ds: Optional[str] = None

    @dataclass
    class PrecomputedConfig:
        prot_info_ds: Optional[str] = None
        atom_prop_ds: Optional[str] = None
        prot_embedding_root: Union[List[str], None] = None
        lig_identifier_src: str = "ligand_file"
        lig_identifier_dst: str = "ligand_file"
        precomputed_mol_prop: bool = False
        linf9_csv: Optional[str] = None
        rmsd_csv: Optional[str] = None
        rmsd_expansion: Optional[str] = None
    pre_computed: PrecomputedConfig = field(default_factory=PrecomputedConfig)

    test_name: Optional[str] = None
    test_set: Optional[str] = None
    proc_lit_pcba: bool = False

@dataclass
class Config:
    comment: str = MISSING
    folder_prefix: str = MISSING
    debug_mode: bool = False
    no_pkd_score: bool = False

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # test_specific:
    short_name: Optional[str] = None
    record_name: Optional[List[str]] = None

schema: Config = OmegaConf.structured(Config)

def read_folder_config(folder_name: str) -> Config:
    config_file = glob(osp.join(folder_name, 'config-*.yaml'))[0]
    args = read_config_file(config_file)
    return args, config_file

def read_config_file(config_file: str) -> Config:
    config = OmegaConf.load(config_file)
    config: Config = OmegaConf.merge(schema, config)
    return config
