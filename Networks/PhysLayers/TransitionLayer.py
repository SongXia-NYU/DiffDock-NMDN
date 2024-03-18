# import torch.nn as nn
#
# from Networks.SharedLayers.ActivationFns import current_activation_fn
#
#
# class TransitionLayer(nn.Module):
#     def __init__(self, F_before, F_after):
#         super().__init__()
#         self.liner = nn.Linear(F_before, F_after)
#
#         self.activation = current_activation_fn
#
#     def forward(self, data):
#         data = self.activation(data)
#         return self.liner(data)
