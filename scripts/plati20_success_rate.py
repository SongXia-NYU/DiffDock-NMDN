import glob

import pandas as pd
import torch
import os.path as osp


def plati20_conf2mol_batch():
    d = torch.load("plati20_mmff_pyg.pt")
    d1 = torch.load("plati20_qm_pyg.pt")
    diff = (d[0].gasEnergy - d1[0].gasEnergy).abs().sum()
    print(f"Diff gasEnergy: {diff}")
    diff = (d[0].CalcSol - d1[0].CalcSol).abs().sum()
    print(f"Diff CalcSol: {diff}")
    smiles = d[0].smiles
    mols = list(set(smiles))
    print(f"Total different molecules: {len(mols)}")
    batch = []
    for smi in smiles:
        batch.append(mols.index(smi))
    batch = torch.as_tensor(batch)
    torch.save(batch, "../../dataProviders/data/processed/plati20_conf2mol_batch.pt")


def plati20_success_rate():
    if not osp.exists("../../dataProviders/data/processed/plati20_conf2mol_batch.pt"):
        plati20_conf2mol_batch()

    batch = torch.load("../../dataProviders/data/processed/plati20_conf2mol_batch.pt")
    root = "/home/carrot_of_rivia/Documents/PycharmProjects/raw_data/frag20-sol-finals"
    exp = "exp_frag20sol_012_active_external_plati20_ALL_2022-05-01_112820"

    test_root = glob.glob(osp.join(root, exp))[0]
    loss_pt = glob.glob(osp.join(test_root, "loss_*_test.pt"))[0]
    loss = torch.load(loss_pt)

    success = {
        "gasEnergy": [],
        "watEnergy": [],
        "octEnergy": []
    }
    unique_mols = 0
    for i in range(batch.min(), batch.max()+1):
        unique_mols += 1
        mask = (batch == i)
        this_pred = loss["PROP_PRED"][mask, :3]
        this_tgt = loss["PROP_TGT"][mask, :3]
        for j, name in enumerate(["gasEnergy", "watEnergy", "octEnergy"]):
            this_this_pred = this_pred[:, j]
            this_this_tgt = this_tgt[:, j]
            if torch.argmin(this_this_pred) == torch.argmin(this_this_tgt):
                success[name].append(1.)
            else:
                success[name].append(0.)

    success_rate = {
        key: sum(success[key]) / len(success[key]) for key in success
    }
    success_rate["unique_mols"] = unique_mols

    success_df = pd.DataFrame(success_rate, index=[0])
    success_df.to_csv(osp.join(test_root, "success_rate.csv"), index=False)

    print("finished")


if __name__ == '__main__':
    plati20_success_rate()
