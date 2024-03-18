from collections import OrderedDict, defaultdict
import json
import os

import torch
import os.path as osp

def prepare_shffule():
    model = "exp_pl_269_run_2023-03-29_161854__845744/best_model.pt"
    outdir = "./pretrained/exp_pl_269_shuffle_272"
    model_params = torch.load(model, map_location="cpu")
    # request_layers = [3, 4, 5, 0, 1, 2]  # for PL271
    request_layers = [3, 0, 4, 1, 5, 2]  # for PL272
    prefix = "module.main_module_list."

    old2new_mapper = defaultdict(lambda: [])
    for new_idx, old_idx in enumerate(request_layers):
        old2new_mapper[old_idx].append(new_idx)

    out = OrderedDict()
    for param in model_params.keys():
        if not param.startswith(prefix):
            out[param] = model_params[param]
            continue

        i = int(param.split(prefix)[-1].split(".")[0])
        rest_name = ".".join(param.split(prefix)[-1].split(".")[1:])
        for new_idx in old2new_mapper[i]:
            new_name = f"{prefix}{new_idx}.{rest_name}"
            out[new_name] = model_params[param]

    os.makedirs(outdir)
    torch.save(out, osp.join(outdir, "best_model.pt"))
    print(out.keys())


def prepare_pl_274():
    model = "./pretrained/exp_frag20sol_012_run_2022-04-20_143627/best_model.pt"
    outdir = "./pretrained/exp_frag20sol_012_for_pl_274"
    model_params = torch.load(model, map_location="cpu")

    for param in list(model_params.keys()):
        prefix = "module.main_module_list."
        if not param.startswith(prefix):
            continue
        if param.endswith(".interaction.message_pass_layer.lin_for_diff.weight"):
            continue
        if param.endswith(".interaction.message_pass_layer.lin_for_diff.bias"):
            continue

        i = int(param.split(prefix)[-1].split(".")[0])
        rest_name = ".".join(param.split(prefix)[-1].split(".")[1:])

        new_name = f"{prefix}{i+3}.{rest_name}"
        model_params[new_name] = model_params[param].clone()
    model_params["module.scale"] = model_params["module.scale"][:, [1, 2]]
    model_params["module.shift"] = model_params["module.shift"][:, [1, 2]]
    os.makedirs(outdir, exist_ok=True)
    torch.save(model_params, osp.join(outdir, "best_model.pt"))
    print(model_params.keys())


def prepare4sep_pl_atom_and_martini():
    """
    Prepare the model for the transfer learning on PL learning. 
    The 0-2 layers and 6-8 layers are for atom embedding, and the 3-5 layers are for Martini-level protein embedding.
    """
    zero_g = True
    ligand_pretrained = "./results/exp_frag20sol_012_run_2022-04-20_143627"
    lig_params = torch.load(osp.join(ligand_pretrained, "best_model.pt"), map_location="cpu")
    protein_pretrained = "./results/exp_pl_200_run_2023-02-03_131644__188282/"
    prot_params = torch.load(osp.join(protein_pretrained, "best_model.pt"), map_location="cpu")

    embed_name = "module.embedding_layer.embedding.weight"
    embedding = prot_params[embed_name]
    embedding[:95, :] = lig_params[embed_name]
    lig_params[embed_name] = embedding

    for param in list(lig_params.keys()):
        prefix = "module.main_module_list."
        if param.startswith(prefix):
            i = int(param.split(prefix)[-1].split(".")[0])
            rest_name = ".".join(param.split(prefix)[-1].split(".")[1:])
            # martini pretrained on layer 3-5
            new_name = f"{prefix}{i+3}.{rest_name}"
            lig_params[new_name] = prot_params[param].clone()
            # small molecule pretrained on layer 0-2 and 6-8
            new_name = f"{prefix}{i+6}.{rest_name}"
            lig_params[new_name] = lig_params[param].clone()

    for key in list(lig_params.keys()):
        if key.startswith(("module.cutoff", "module.centers", "module.widths", "module.scale", "module.shift")):
            del lig_params[key]
        if zero_g and key.endswith("interaction.message_pass_layer.G.weight"):
            del lig_params[key]
    for key in lig_params.keys():
        print(key)
    output = "./results/TL4sep_pl_lig_sol12_prot_pl200"
    if zero_g:
        output += "_zero-g"
    os.makedirs(output, exist_ok=True)
    torch.save(lig_params, osp.join(output, "best_model.pt"))

    info = {"ligand_pretrained": ligand_pretrained, "prot_pretrained": protein_pretrained,
            "comment": "3 'stacked' sPhysNet models (9 layers in total), initialized by both pretrained"}
    with open(osp.join(output, "info.json"), "w") as f:
        json.dump(info, f, indent=2)


def prepare4sep_pl():
    ligand_pretrained = "../results/exp_frag20sol_012_run_2022-04-20_143627"
    protein_pretrained = "../results/exp_pl_051_run_2022-10-09_091147__533310"
    lig_params = torch.load(osp.join(ligand_pretrained, "best_model.pt"))
    prot_params = torch.load(osp.join(protein_pretrained, "best_model.pt"))

    for param in prot_params:
        prefix = "module.main_module_list."
        if param.startswith(prefix):
            i = int(param.split(prefix)[-1].split(".")[0])
            rest_name = ".".join(param.split(prefix)[-1].split(".")[1:])
            new_name = f"{prefix}{i+3}.{rest_name}"
            lig_params[new_name] = prot_params[param]

    embed_name = "module.embedding_layer.embedding.weight"
    embedding = prot_params[embed_name]
    embedding[:95, :] = lig_params[embed_name]
    lig_params[embed_name] = embedding

    for key in list(lig_params.keys()):
        if key.startswith(("module.cutoff", "module.centers", "module.widths", "module.scale", "module.shift")):
            del lig_params[key]
    for key in lig_params.keys():
        print(key)
    output = "../results/TL4sep_pl"
    os.makedirs(output, exist_ok=True)
    torch.save(lig_params, osp.join(output, "best_model.pt"))

    info = {"ligand_pretrained": ligand_pretrained, "protein_pretrained": protein_pretrained}
    with open(osp.join(output, "info.json"), "w") as f:
        json.dump(info, f, indent=2)


def prepare4sep_pl_atom():
    mix = False
    zero_g = True
    ligand_pretrained = "../results/exp_frag20sol_012_run_2022-04-20_143627"
    lig_params = torch.load(osp.join(ligand_pretrained, "best_model.pt"), map_location="cpu")

    protein_pretrained = "../results/exp_pl_080_run_2022-10-26_142146__050010"
    prot_params = torch.load(osp.join(protein_pretrained, "best_model.pt"), map_location="cpu")

    embed_name = "module.embedding_layer.embedding.weight"
    embedding = prot_params[embed_name]
    embedding[:95, :] = lig_params[embed_name]
    lig_params[embed_name] = embedding

    for param in list(lig_params.keys()):
        prefix = "module.main_module_list."
        if param.startswith(prefix):
            i = int(param.split(prefix)[-1].split(".")[0])
            if mix:
                i = i*3
                i1 = 1
                i2 = 2
            else:
                i1 = 3
                i2 = 6
            rest_name = ".".join(param.split(prefix)[-1].split(".")[1:])
            new_name = f"{prefix}{i+i1}.{rest_name}"
            lig_params[new_name] = lig_params[param].clone()
            new_name = f"{prefix}{i+i2}.{rest_name}"
            lig_params[new_name] = lig_params[param].clone()

    for key in list(lig_params.keys()):
        if key.startswith(("module.cutoff", "module.centers", "module.widths", "module.scale", "module.shift")):
            del lig_params[key]
        if zero_g and key.endswith("interaction.message_pass_layer.G.weight"):
            del lig_params[key]
    for key in lig_params.keys():
        print(key)
    output = "../results/TL4sep_pl_atom"
    if mix:
        output += "_mix"
    if zero_g:
        output += "_zero-g"
    os.makedirs(output, exist_ok=True)
    torch.save(lig_params, osp.join(output, "best_model.pt"))

    info = {"ligand_pretrained": ligand_pretrained, "protein_pretrained": protein_pretrained, "comment": "3 'stacked' sPhysNet models (9 layers in total)"}
    with open(osp.join(output, "info.json"), "w") as f:
        json.dump(info, f, indent=2)


if __name__ == '__main__':
    prepare_pl_274()
