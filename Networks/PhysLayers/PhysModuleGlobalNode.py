import torch
import torch.nn as nn
from torch_scatter import scatter
from Networks.PhysLayers.PhysModule import PhysModule
from Networks.PairedPropLayers.MDNLayer import GaussExpandLayer
from Networks.SharedLayers.PoolingReadoutLayer import ReadoutLayerFactory


class PhysModuleGlobalNode(PhysModule, GaussExpandLayer):
    """
    Automatically add a master node (global node) to each molecule.
    """
    def __init__(self, *args, **kwargs):
        PhysModule.__init__(self, *args, **kwargs)
        GaussExpandLayer.__init__(self, kwargs["config_dict"])

    def forward(self, runtime_vars: dict):
        self.infuse_global_node(runtime_vars)
        self.modify_edge_index(runtime_vars)
        runtime_vars = PhysModule.forward(self, runtime_vars)
        return runtime_vars
    
    def infuse_global_node(self, runtime_vars: dict) -> None:
        if ("global_node_infused" in runtime_vars) and runtime_vars["global_node_infused"]:
            return
        # Add global nodes (fake nodes) to the end of real nodes (atoms)
        # Global nodes are initialized by KANO embeddings.
        phys_atom_embed: torch.Tensor = runtime_vars["vi"]
        kano_atom_embed: torch.Tensor = runtime_vars["kano_atom_embed"]
        atom_mol_batch: torch.LongTensor = runtime_vars["data_batch"].atom_mol_batch
        mol_embed: torch.Tensor = scatter(kano_atom_embed, atom_mol_batch, dim=0)
        combined_embed: torch.Tensor = torch.concat([phys_atom_embed, mol_embed], dim=0)
        runtime_vars["vi"] = combined_embed
        runtime_vars["global_node_infused"] = True

    def modify_edge_index(self, runtime_vars: dict) -> None:
        # modify edge index and add fake distances
        atom_mol_batch: torch.LongTensor = runtime_vars["data_batch"].atom_mol_batch
        n_mols: int = runtime_vars["data_batch"].N.shape[0]
        combined_embed: torch.Tensor = runtime_vars["vi"]
        n_atoms: int = combined_embed.shape[0] - n_mols

        # modify edge_index
        imaginary_nodes_id: torch.LongTensor = atom_mol_batch + n_atoms
        real_node_id: torch.LongTensor = torch.arange(atom_mol_batch.shape[0]).to(atom_mol_batch.device)
        real2img_edge_index = torch.stack([real_node_id, imaginary_nodes_id], dim=0)
        img2real_edge_index = torch.stack([imaginary_nodes_id, real_node_id], dim=0)
        edge_index = torch.concat([runtime_vars["edge_index"], real2img_edge_index, img2real_edge_index], dim=-1)
        runtime_vars["edge_index"] = edge_index

        if ("fake_rbf_infused" in runtime_vars) and runtime_vars["fake_rbf_infused"]:
            return
        # modify RBF expansion
        # For some reason the edge_attr is stored in a dictionary and therefore modified in-place
        # as a result, it only needs to be modified once.
        edge_attr: torch.Tensor = runtime_vars["edge_attr"]
        fake_dist = torch.zeros_like(real2img_edge_index[0], dtype=edge_attr["rbf"].dtype).fill_(5.0).view(-1, 1)
        fake_dist = torch.concat([fake_dist, fake_dist], dim=0)
        fake_rbf = self.gaussian_dist_infuser(fake_dist)
        edge_attr["rbf"] = torch.concat([edge_attr["rbf"], fake_rbf], dim=0)
        runtime_vars["edge_attr"] = edge_attr
        runtime_vars["fake_rbf_infused"] = True


class GlobalNodeReadoutPooling(nn.Module):
    def __init__(self, activation: str, use_embed: str="mol", pooling_type: str = "mean", readout_type: str = "lin", **readout_kwargs) -> None:
        super().__init__()
        readout_factory = ReadoutLayerFactory()
        self.readout_layer = readout_factory.get_layer(readout_type, activation=activation, **readout_kwargs)
        self.pooling_type: str = pooling_type
        self.use_embed: str = use_embed

    def forward(self, runtime_vars: dict):
        if self.use_embed == "mol":
            return self.use_mol_embed_forward(runtime_vars)
        
        assert self.use_embed == "atom", self.use_embed
        return self.use_atom_embed_forward(runtime_vars)

    def use_atom_embed_forward(self, runtime_vars: dict):
        combined_embed: torch.Tensor = runtime_vars["vi"]
        n_mols: int = runtime_vars["data_batch"].N.shape[0]
        atom_embed = combined_embed[:-n_mols, :]
        atom_prop = self.readout_layer(atom_embed)
        atom_mol_batch: torch.LongTensor = runtime_vars["data_batch"].atom_mol_batch
        if self.pooling_type in ["mean", "sum"]:
            mol_prop: torch.Tensor = scatter(reduce=self.pooling_type, src=atom_prop, index=atom_mol_batch, dim=0)
        else:
            raise NotImplementedError(self.pooling_type)
        
        runtime_vars["atom_embed_readout_pooling_mol_prop"] = mol_prop
        return runtime_vars

    def use_mol_embed_forward(self, runtime_vars: dict):
        combined_embed: torch.Tensor = runtime_vars["vi"]
        n_mols: int = runtime_vars["data_batch"].N.shape[0]
        mol_embed = combined_embed[-n_mols:, :]
        mol_prop = self.readout_layer(mol_embed)
        runtime_vars["mol_embed_readout_pooling_mol_prop"] = mol_prop
        return runtime_vars
