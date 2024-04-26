from utils.utils_functions import torchdrug_imports
torchdrug_imports()
import torch
import os.path as osp
from utils.LossFn import calculate_probablity
from utils.eval.tester import Tester
from torch.utils.data import Subset
from torch.distributions import Normal
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt

"""
Predict and save some tensors to visulize MDN layers.
"""

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def pred_mdn():
    modeldir = "./exp_pl_534_run_2024-01-22_211045__480688"
    wanted_pdb = "6fap"

    args = Tester(modeldir).cfg
    args["mdn_w_lig_atom_props"] = 0.
    args["mdn_w_prot_sasa"] = 0.
    tester = Tester(modeldir, explicit_ds_config=args)
    for idx in tester.ds.idx2pdb_mapper:
        this_pdb = tester.ds.idx2pdb_mapper[idx]
        if this_pdb == wanted_pdb:
            print(idx)
            print(tester.ds.idx2lig_mapper[idx])
            print("---")
    
    # data = tester.ds.get(38513)
    dl = DataLoader(Subset(tester.ds, [idx]))
    for data in dl:
        model_out = tester.model(data)
        print(model_out)
        break

    for key in model_out:
        d = model_out[key]
        if isinstance(d, torch.Tensor):
            d = d.detach().cpu()
        model_out[key] = d

    model_out["data_batch"] = data
    
    torch.save(model_out, osp.join(modeldir, f"{wanted_pdb}.mdn.pth"))

def draw_graphs():
    d = torch.load("exp_pl_534_run_2024-01-22_211045__480688/6fap.mdn.pth", map_location="cpu")
    STEP = 0.05
    print(d.keys())

    prob = calculate_probablity(d["pi"], d["sigma"], d["mu"], d["dist"])
    argsort = torch.argsort(prob)[-30:]
    pair = torch.concat([prob.view(-1, 1), d["dist"].view(-1, 1)], dim=-1).numpy()
    print(pair[argsort, :])

    # wanted_pair_mask = (d["dist"] > 3.67) & (d["dist"] < 3.68)
    wanted_pair_mask = (d["dist"] > 8.71959) & (d["dist"] < 8.71960)
    wanted_pair_mask = wanted_pair_mask.view(-1)
    pi = d["pi"][wanted_pair_mask].view(-1, 1)
    sigma = d["sigma"][wanted_pair_mask].view(-1, 1)
    mu = d["mu"][wanted_pair_mask].view(-1, 1)
    tgt_dist = d["dist"][wanted_pair_mask].view(-1, 1)
    data = d["data_batch"]

    # selected_idx = data.PL_oneway_edge_index[:, wanted_pair_mask]
    # idx1 = selected_idx[0]
    # idx2 = selected_idx[1]
    # print(f"Atom 1: {data.R[idx1, :]} {data.Z[idx1]}")
    # print(f"Atom 2: {data.R[idx2, :]} {data.Z[idx2]}")

    pl_edge = data[("ligand", "interaction", "protein")].min_dist_edge_index
    pl_dist = data[("ligand", "interaction", "protein")].min_dist 
    pl_edge = pl_edge[:, pl_dist<9.]
    selected_idx = pl_edge[:, wanted_pair_mask]
    idx1 = selected_idx[0]
    idx2 = selected_idx[1]

    dist_range = torch.arange(0, 10, STEP).view(1, -1)
    normal = Normal(mu, sigma)
    logprob = normal.log_prob(dist_range.expand(10, -1))
    logprob += torch.log(pi)
    prob = logprob.exp().sum(0)

    logprob_tgt = normal.log_prob(tgt_dist.expand(10, -1))
    logprob_tgt += torch.log(pi)
    tgt_prob = logprob_tgt.exp().sum(0)

    plt.plot(dist_range.numpy().reshape(-1), prob.numpy().reshape(-1))
    plt.xlabel("Distance (A)")
    plt.ylabel("Probability Density")
    plt.title("Model Predicted Distance Probability")
    tgt_dist = tgt_dist.item()
    tgt_prob = tgt_prob.item()
    plt.vlines(x=tgt_dist, ymin=0, ymax=tgt_prob, colors="purple", linestyles="--")
    plt.annotate("P(dist)={:.2f}".format(tgt_prob), xy=(tgt_dist, tgt_prob), 
                 arrowprops={"facecolor": "black", "width": 0.2, "headwidth": 3.5}, 
                 xytext=(tgt_dist-6, tgt_prob-0.5))
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("exp_pl_534_run_2024-01-22_211045__480688/6fap.mdnprob.png")

    cumu_prob = (prob * STEP).sum()
    print(cumu_prob)
    print("finished")

if __name__ == "__main__":
    draw_graphs()
