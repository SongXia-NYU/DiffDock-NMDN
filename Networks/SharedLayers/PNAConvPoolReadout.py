import torch
import torch.nn as nn
from torch_geometric.nn import PNAConv
from torch_geometric.data import Dataset

from Networks.SharedLayers.PoolingReadoutLayer import ReadoutLayerFactory

class PNAConvPoolReadout(nn.Module):
    """
    Implement imaginary node pooling with PNAConv
    """
    def __init__(self, n_feature: int, readout_type: str, activation: str, ds: Dataset, **readout_kwargs) -> None:
        super().__init__()

        # if the readout_type is set to "identity", we use pna_conv as both convolution and readout
        # otherwise
        out_channels = readout_kwargs["n_output"] if readout_type == "identity" else n_feature
        # the in-degree of each imaginary node in the training set --> the number of real atoms in the molecule
        deg = torch.bincount(ds.data.N[ds.train_index])
        self.pna_conv = PNAConv(n_feature, out_channels, ["mean", "min", "max", "std"], ["identity", "amplification", "attenuation"], deg)

        readout_factory = ReadoutLayerFactory()
        self.readout_layer = readout_factory.get_layer(readout_type, activation=activation, n_feature=n_feature, **readout_kwargs)

    def forward(self, runtime_vars: dict):
        """
        Add imaginary nodes at the end of atom embeddings and perform message passing onto the imaginary node.
        """
        atom_embed: torch.Tensor = runtime_vars["vi"]
        atom_mol_batch: torch.LongTensor = runtime_vars["data_batch"].atom_mol_batch
        n_mols: int = runtime_vars["data_batch"].N.shape[0]
        n_atoms: int = atom_embed.shape[0]

        imaginary_nodes = torch.zeros((n_mols, atom_embed.shape[-1]), dtype=atom_embed.dtype, device=atom_embed.device)
        atom_embed = torch.concat([atom_embed, imaginary_nodes], dim=0)

        imaginary_nodes_id: torch.LongTensor = atom_mol_batch + n_atoms
        real_node_id: torch.LongTensor = torch.arange(atom_mol_batch.shape[0]).to(atom_mol_batch.device)
        real2img_edge_index = torch.stack([real_node_id, imaginary_nodes_id], dim=0)

        pna_out: torch.Tensor = self.pna_conv(atom_embed, real2img_edge_index)
        readout_out: torch.Tensor = self.readout_layer(pna_out)
        mol_prop: torch.Tensor = readout_out[-n_mols:, :]
        runtime_vars["pooling_reaout_mol_prop"] = mol_prop
        return runtime_vars
    