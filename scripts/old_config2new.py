from glob import glob
import yaml
from utils.configs import schema

def old2new(old_cfg_file: str):
    old_dict = {}
    with open(old_cfg_file) as f:
        for line in f.readlines():
            line = line[2:].strip("\n")
            if "=" not in line:
                old_dict[line] = True
                continue
            key = line.split("=")[0]
            val = "=".join(line.split("=")[1:])
            try:
                val = int(val)
                old_dict[key] = val
                continue
            except ValueError:
                pass
            try:
                val = float(val)
                old_dict[key] = val
                continue
            except ValueError:
                pass
            if val == "False": val = False
            if val == "True": val = True
            old_dict[key] = val

    out_dict = {}
    if "prot_embedding_root" in old_dict:
        out_dict["data"] = {"pre_computed": {"prot_embedding_root": [old_dict["prot_embedding_root"]]}}
        del old_dict["prot_embedding_root"]
    if "target_names" in old_dict:
        out_dict["training"] = {"loss_fn": {"target_names": [old_dict["target_names"]]}}
        del old_dict["target_names"]
    if "record_name" in old_dict: del old_dict["record_name"]

    with open(old_cfg_file) as f:
        for line in f.readlines():
            line = line[2:].strip("\n")
            if "=" not in line:
                continue
            key = line.split("=")[0]
            val = "=".join(line.split("=")[1:])
            if key == "record_name":
                if key not in out_dict:
                    out_dict[key] = []
                out_dict["record_name"].append(val)

    if "kano_ds" in old_dict: del old_dict["kano_ds"]

    for key in old_dict:
        if key in schema.keys():
            out_dict[key] = old_dict[key]
        for subkey in ["model", "training", "data"]:
            if key in schema[subkey].keys():
                if subkey not in out_dict:
                    out_dict[subkey] = {}
                out_dict[subkey][key] = old_dict[key]

        for subsubkey in ["physnet", "mdn", "dimenet", "normalization", "kano", "comenet"]:
            if key in schema.model[subsubkey]:
                if subsubkey not in out_dict["model"]:
                    out_dict["model"][subsubkey] = {}
                out_dict["model"][subsubkey][key] = old_dict[key]

        for subsubkey in ["loss_fn"]:
            if key in schema.training[subsubkey]:
                if subsubkey not in out_dict["training"]:
                    out_dict["training"][subsubkey] = {}
                out_dict["training"][subsubkey][key] = old_dict[key]

        for subsubkey in ["pre_computed"]:
            if key in schema.data[subsubkey]:
                if subsubkey not in out_dict["data"]:
                    out_dict["data"][subsubkey] = {}
                out_dict["data"][subsubkey][key] = old_dict[key]
    outdir = old_cfg_file.replace(".txt", ".yaml")
    with open(outdir, "w") as f:
        yaml.safe_dump(out_dict, f)

for cfg_name in glob("configs/*.txt"):
    old2new(cfg_name)