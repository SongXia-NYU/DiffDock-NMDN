import torch
from utils.utils_functions import atom_mean_std
prop_tensor = torch.load("/scratch/sx801/scripts/Mol3DGenerator/scripts/AF-SwissProt-Frags/AF-SwissProt-CAP-FRAG1To2-Martini-GBSA-c10.prop.pth")
n_tensor = torch.load("/scratch/sx801/scripts/Mol3DGenerator/scripts/AF-SwissProt-Frags/AF-SwissProt-CAP-FRAG1To2-Martini-GBSA-c10.n.pth")

mean_atom, std_atom = atom_mean_std(prop_tensor, n_tensor, torch.arange(len(n_tensor)))
print(mean_atom)
print(std_atom)