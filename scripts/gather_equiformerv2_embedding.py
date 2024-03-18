# retrieve the atom embedding of equiformer v2
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from Networks.SharedLayers.EquiformerV2 import EquiformerV2


def embedding_test():
    model_cfg = {"equiformer_v2_ckpt": "/scratch/sx801/scripts/ocp-1205/checkpoints/downloaded/eq2_31M_ec4_allmd.pt",
                 "equiformer_v2_narrow_embed": False, 
                 "equiformer_v2_for_energy": False}
    equiformer_v2 = EquiformerV2(model_cfg).cuda()
    equiformer_v2.eval()

    runtime_vars = {}

    print("Distance == 11")
    batch = Data(R=torch.as_tensor([[0., 0., 0.], [11., 0., 0.]]).view(2, 3),
                 Z=torch.as_tensor([1, 1]),
                 N=torch.as_tensor([1, 1]),
                 atom_mol_batch=torch.as_tensor([0, 0]))
    runtime_vars["data_batch"] = batch.cuda()
    print(equiformer_v2.forward(runtime_vars)["vi"])

    print("Distance == 11.9")
    batch = Data(R=torch.as_tensor([[0., 0., 0.], [11.9, 0., 0.]]).view(2, 3),
                 Z=torch.as_tensor([1, 1]),
                 N=torch.as_tensor([1, 1]),
                 atom_mol_batch=torch.as_tensor([0, 0]))
    runtime_vars["data_batch"] = batch.cuda()
    print(equiformer_v2.forward(runtime_vars)["vi"])


def embedding():
    model_cfg = {"equiformer_v2_ckpt": "/scratch/sx801/scripts/ocp-1205/checkpoints/downloaded/eq2_31M_ec4_allmd.pt",
                 "equiformer_v2_narrow_embed": False, 
                 "equiformer_v2_for_energy": False}
    equiformer_v2 = EquiformerV2(model_cfg).cuda()
    equiformer_v2.eval()

    embeddings = []

    for atom_num in tqdm(range(0, 85), total=85):
        # atom_num 0 is only used as padding
        if atom_num == 0: atom_num = 1
        runtime_vars = {}
        batch = Data(R=torch.as_tensor([[0., 0., 0.], [11.9, 0., 0.]]).view(2, 3),
                 Z=torch.as_tensor([atom_num, atom_num]),
                 N=torch.as_tensor([1, 1]),
                 atom_mol_batch=torch.as_tensor([0, 0]))
        runtime_vars["data_batch"] = batch.cuda()
        atom_embed = equiformer_v2.forward(runtime_vars)["vi"].detach().cpu()
        embeddings.append(atom_embed[0, :].view(1, -1))
    embeddings = torch.concat(embeddings, dim=0)
    torch.save(embeddings, "/scratch/sx801/data/im_datasets/processed/equiformer_v2_atom_embed.pth")

if __name__ == "__main__":
    embedding()
