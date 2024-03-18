import torch.nn as nn

class GatedEmbedLayer(nn.Module):
    """
    A gate layer to preserve the original ESM protein embedding.
    """
    def __init__(self, weight=0.999) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, input_dict):
        data_batch = input_dict["data_batch"]
        og_embed = data_batch.prot_embed
        new_embed = input_dict["prot_embed"]

        gated_embed = og_embed * self.weight + new_embed * (1-self.weight)
        input_dict["prot_embed"] = gated_embed
        return input_dict