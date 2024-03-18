# import time
#
# import torch
# import torch_geometric
#
# import utils.grimme_d3
# from torch_geometric.nn import SAGPooling
# from Networks.MainModule import MainModule
# from Networks.TransitionLayer import TransitionLayer
# from utils.utils_functions import cal_coulomb_E, cal_p, cal_edge, k_e, device, floating_type, bessel_expansion_raw
#
#
# class PhysNetEFGs(torch.nn.Module):
#     """
#     The whole PhysNet model are included in the following code
#     F: number of features of a node
#     K: number of features of an edge
#     n_output: number of outputs, usually set to 2(If you set both cal_charge and cal_coulomb to False,
#         you can set this to 1). The actually outputs are irrelevant to n_output,
#         it depends on cal_force and cal_charge params
#     n_modules: number of main modules in PhysNet
#     lambda_nh: the coefficient of non-hierarchical penalty, see paper for more details
#     cal_force: whether calculate force or not
#     cal_charge: whether calculate charge or not
#     cal_coulomb: whether calculate coulomb interaction(long range interaction) or not
#     cal_dispersion: whether calculate dispersion term or not
#     """
#
#     def __init__(self, F_atom, K_atom, n_output, n_atomic_modules, n_efg_modules, lambda_nh, num_EFGs_types,
#                  cal_force, cal_charge, cal_atom_coulomb, cal_atom_dispersion, debug_mode, E_efg_shift, E_efg_scale,
#                  s6, s8, a1, a2, E_atomic_shift, E_atomic_scale, n_res_atomic, n_res_interaction, n_res_output,
#                  atom_feature_dist, efg_feature_dist, F_efg, K_efg, cal_efg_coulomb, SAGPool_EFG, SAGPool_ratio,
#                  normalize_EFGs, atom_prop_weight, trainable_atom_prop_weight, enable_cutoff):
#         super().__init__()
#         # number of neighbors to calculate edge information
#         # note that in kd_tree.query(x, k), k=num_neighbors+1 since the first neighbor will be the node itself
#         self.F_atom = F_atom
#         self.K_atom = K_atom
#         self.F_efg = F_efg
#         self.K_efg = K_efg
#
#         self.SAGPool_EFG = SAGPool_EFG
#         self.SAGPool_ratio = SAGPool_ratio
#         if SAGPool_EFG:
#             self.SAGPool_layer = SAGPooling(F_atom, SAGPool_ratio)
#
#         self.atom_feature_dist = torch.Tensor([atom_feature_dist]).type(floating_type).to(device)
#         self.efg_feature_dist = torch.Tensor([efg_feature_dist]).type(floating_type).to(device)
#
#         self.lambda_nh = torch.Tensor([lambda_nh]).type(floating_type).to(device)
#
#         self.cal_force = cal_force
#         self.cal_charge = cal_charge
#
#         self.cal_atom_coulomb = cal_atom_coulomb
#         self.cal_efg_coulomb = cal_efg_coulomb
#         self.cal_atom_dispersion = cal_atom_dispersion
#
#         self.normalize_EFGs = normalize_EFGs
#
#         self.atom_lr_interaction = cal_atom_coulomb or cal_atom_dispersion
#         self.efg_lr_interaction = cal_efg_coulomb
#
#         self.s6 = s6
#         self.s8 = s8
#         self.a1 = a1
#         self.a2 = a2
#         self.debug_mode = debug_mode
#
#         self.enable_cutoff = enable_cutoff
#
#         atom_prop_weight = torch.Tensor([atom_prop_weight]).view(-1).type(floating_type)
#         self.register_parameter('atom_prop_weight', torch.nn.Parameter(atom_prop_weight, trainable_atom_prop_weight))
#
#         '''
#         These options below are actually hyper-params which should be implemented to config-exp315.txt file in the future
#         '''
#         init_atomic_fraction = 1.0
#         init_efg_fraction = 1.0
#         self.cal_nh_EFGs_to_atom = False
#
#         embedding_matrix = torch.Tensor(95, F_atom).type(floating_type).uniform_(-1.732, +1.732)
#         self.register_parameter('embedding_matrix', torch.nn.Parameter(embedding_matrix, True))
#
#         self.n_atomic_modules = n_atomic_modules
#         for i in range(n_atomic_modules):
#             self.add_module("atomic_module" + str(i), MainModule(F=F_atom, K=K_atom, n_output=n_output,
#                                                                  n_res_atomic=n_res_atomic,
#                                                                  n_res_interaction=n_res_interaction,
#                                                                  n_res_output=n_res_output))
#
#         self.transition_layer = TransitionLayer(F_atom, F_efg)
#
#         self.n_efg_modules = n_efg_modules
#         for i in range(n_efg_modules):
#             self.add_module("efg_module" + str(i), MainModule(F=F_efg, K=K_efg, n_output=n_output,
#                                                               n_res_atomic=n_res_atomic,
#                                                               n_res_interaction=n_res_interaction,
#                                                               n_res_output=n_res_output))
#
#         atomic_scale = torch.Tensor(95, 2).type(floating_type)
#         # the scale of energy
#         atomic_scale[:, 0] = E_atomic_scale * init_atomic_fraction
#         atomic_scale[:, 1] = 1. * init_atomic_fraction  # the scale of charge
#         self.register_parameter('atomic_scale', torch.nn.Parameter(atomic_scale, True))
#         atomic_shift = torch.Tensor(95, 2).type(floating_type)
#         atomic_shift[:, 0] = E_atomic_shift * init_atomic_fraction
#         atomic_shift[:, 1] = 0. * init_atomic_fraction
#         self.register_parameter('atomic_shift', torch.nn.Parameter(atomic_shift, True))
#
#         if self.normalize_EFGs:
#             if self.SAGPool_layer:
#                 '''
#                 If using SAGPooling layer, EFG message is essentially (selected) atom message
#                 '''
#                 num_EFGs_types = 95
#                 E_efg_scale = E_atomic_scale
#                 E_efg_shift = E_efg_shift / self.SAGPool_ratio
#             efgs_scale = torch.Tensor(num_EFGs_types, 2).type(floating_type)
#             efgs_scale[:, 0] = E_efg_scale * init_efg_fraction
#             efgs_scale[:, 1] = 1. * init_efg_fraction
#             self.register_parameter('efgs_scale', torch.nn.Parameter(efgs_scale, True))
#             efgs_shift = torch.Tensor(num_EFGs_types, 2).type(floating_type)
#             efgs_shift[:, 0] = E_efg_shift * init_efg_fraction
#             efgs_shift[:, 1] = 0. * init_efg_fraction
#             self.register_parameter('efgs_shift', torch.nn.Parameter(efgs_shift, True))
#
#     def forward(self, _data, meta_data):
#
#         # 'unzip' the data instead of directly using the dict to improve computational efficiency
#         Z = _data.data.Z
#         R = _data.data.R
#         N = _data.data.N
#         EFG_R = _data.data.EFG_R
#         atom_to_efg_batch = _data.data.atom_to_EFG_batch
#         EFG_N = _data.data.EFG_N
#         EFGs_Z = _data.data.EFG_Z
#         atom_edge_index = _data.data.atom_edge_index
#         EFG_edge_index = _data.data.EFG_edge_index
#
#         atom_mol_batch = meta_data['atom_mol_batch']
#         efg_mol_batch = meta_data['efg_mol_batch']
#         atoms_prev_N = meta_data['atoms_prev_N']
#         EFG_prev_N = meta_data['EFG_prev_N']
#
#         # mol_lvl_detail layer
#         x = self.embedding_matrix[Z, :]
#
#         if self.cal_force:
#             # set R requires grad in order to calculate force from dE/dR
#             R.requires_grad_(True)
#
#         # debug: cal passed time to improve code efficiency
#         # t0 = time.time()
#
#         # calculate edge indexes and attributes for Atom-wise and EFG-wise
#         lr_atom_edge_dist, lr_atom_edge_index, atom_edge_dist, atom_edge_index = \
#             cal_edge(R, N, atoms_prev_N, atom_edge_index,
#                      cal_coulomb=self.atom_lr_interaction, use_cutoff=self.enable_cutoff)
#
#         lr_efg_edge_dist, lr_efg_edge_index, efg_edge_dist, efg_edge_index = \
#             cal_edge(EFG_R, EFG_N, EFG_prev_N, EFG_edge_index, cal_coulomb=self.efg_lr_interaction)
#
#         # Change edge_attr here:
#         # The first line of code is RBF function in PhysNet
#         # The second and third lines are Bessel expansion function inspired by the solve of Schrodinger Equation
#         # The idea of Bessel expansion comes from paper 'DIRECTIONAL MESSAGE PASSING FOR MOLECULAR GRAPHS'
#         # edge_attr = rbf_expansion(atom_edge_dist, self.centers, self.widths, self.cutoff.to(device))
#         edge_attr = bessel_expansion_raw(atom_edge_dist, self.K_atom, self.atom_feature_dist)
#         # edge_attr = bessel_expansion_continuous(atom_edge_dist, self.K, self.cutoff)
#         efg_edge_attr = bessel_expansion_raw(efg_edge_dist, self.K_efg, self.efg_feature_dist)
#
#         # debug: cal passed time to improve code
#         # print('T------cal edge: ', time.time() - t0)
#
#         # Go through m atomic modules and calculate non-hierarchical penalty
#         h_atom = x
#         loss_nh = torch.Tensor([0]).to(device).type(floating_type).view(-1)
#         Ei = torch.zeros_like(Z).type(floating_type).to(device).view(-1, 1)
#         Qi = torch.zeros_like(Z).type(floating_type).to(device).view(-1, 1)
#         for i in range(self.n_atomic_modules):
#             atomic_module_name = "atomic_module" + str(i)
#             out_atom, h_atom, msg_atom = self._modules[atomic_module_name](h_atom, atom_edge_index, edge_attr)
#             Ei = Ei + out_atom[:, 0].view(-1, 1)
#             Qi = Qi + out_atom[:, 1].view(-1, 1)
#             out_atom2 = out_atom ** 2
#             if i >= 1:
#                 loss_nh = loss_nh + torch.mean(out_atom2 / (out_atom2 + last_out_atom2 + 1e-7)) * self.lambda_nh
#             last_out_atom2 = out_atom2
#
#         if self.SAGPool_EFG:
#             '''
#             If true, calculate EFG through SAGPooling layer. The resulting EFG will be the remaining atoms in this mol.
#             The R will be the location of the atom chosen.
#             '''
#             h_efg, efg_edge_index, efg_edge_attr, efg_mol_batch, atom_efg_perm, _ = self.SAGPool_layer(h_atom,
#                                                                                                        atom_edge_index,
#                                                                                                        edge_attr,
#                                                                                                        atom_mol_batch)
#             EFGs_Z = Z[atom_efg_perm]
#         else:
#             '''
#             Otherwise, calculate EFG through pre-defined way: EFGs by Jocelyn Lu. Note in this way, the R:s
#             will be the mass center of all atoms in each EFG.
#             The EFGs representation will be the summation of every atom
#             '''
#             merged_atom = torch_geometric.utils.scatter_('add', h_atom, atom_to_efg_batch)
#             h_efg = self.transition_layer(merged_atom)
#
#         # Go through m efg modules and calculate non-hierarchical penalty
#         E_efg = torch.zeros_like(efg_mol_batch).type(floating_type).to(device).view(-1, 1)
#         Q_efg = torch.zeros_like(efg_mol_batch).type(floating_type).to(device).view(-1, 1)
#
#         for i in range(self.n_efg_modules):
#             efg_module_name = "efg_module" + str(i)
#             out_efg, h_efg, _ = self._modules[efg_module_name](h_efg, efg_edge_index, efg_edge_attr)
#             E_efg = E_efg + out_efg[:, 0].view(-1, 1)
#             Q_efg = Q_efg + out_efg[:, 1].view(-1, 1)
#             out_efg2 = out_efg ** 2
#             if i >= 1:
#                 loss_nh = loss_nh + torch.mean(out_efg2 / (out_efg2 + last_out_efg2 + 1e-7)) * self.lambda_nh
#             last_out_efg2 = out_efg2
#
#         # debug: cal passed time to improve code
#         # print('T------modules: ', time.time() - t0)
#
#         # Atom-wise and EFG level shifting and scaling
#         e_scale_atomic = self.atomic_scale[Z, 0].view(-1, 1)
#         q_scale_atomic = self.atomic_scale[Z, 1].view(-1, 1)
#         e_shift_atomic = self.atomic_shift[Z, 0].view(-1, 1)
#         q_shift_atomic = self.atomic_shift[Z, 1].view(-1, 1)
#         Ei = (e_scale_atomic * Ei.view(-1, 1) + e_shift_atomic).view(-1)
#         Qi = (q_scale_atomic * Qi.view(-1, 1) + q_shift_atomic).view(-1)
#
#         if self.normalize_EFGs:
#             e_scale_efgs = self.efgs_scale[EFGs_Z, 0].view(-1, 1)
#             q_scale_efgs = self.efgs_scale[EFGs_Z, 1].view(-1, 1)
#             e_shift_efgs = self.efgs_shift[EFGs_Z, 0].view(-1, 1)
#             q_shift_efgs = self.efgs_shift[EFGs_Z, 1].view(-1, 1)
#             E_efg = (e_scale_efgs * E_efg.view(-1, 1) + e_shift_efgs).view(-1)
#             Q_efg = (q_scale_efgs * Q_efg.view(-1, 1) + q_shift_efgs).view(-1)
#
#         # Calculate energy of whole molecule my summing up contributions of each atom or EFGs
#         Q_mol_atomic = torch_geometric.utils.scatter_('add', Qi, atom_mol_batch).view(-1)
#         E_mol_atomic = torch_geometric.utils.scatter_('add', Ei, atom_mol_batch).view(-1)
#         Q_mol_efgs = torch_geometric.utils.scatter_('add', Q_efg, efg_mol_batch).view(-1)
#         E_mol_efgs = torch_geometric.utils.scatter_('add', E_efg, efg_mol_batch).view(-1)
#
#         if self.cal_nh_EFGs_to_atom:
#             '''
#             Deprecated, will be removed in the future
#             '''
#             # non-hierarchical penalty
#             Q_mol_atomic2 = Q_mol_atomic ** 2
#             E_mol_atomic2 = E_mol_atomic ** 2
#             Q_mol_efgs2 = Q_mol_efgs ** 2
#             E_mol_efgs2 = E_mol_efgs ** 2
#             loss_nh = loss_nh + self.lambda_nh * 0.5 * torch.mean(
#                 Q_mol_efgs2 / (Q_mol_efgs2 + Q_mol_atomic2 + 1e-7) + E_mol_efgs2 / (E_mol_efgs2 + E_mol_atomic2 + 1e-7)
#             )
#
#         # debug: cal passed time to improve code
#         # print('T------E, Q pred: ', time.time() - t0)
#
#         if self.cal_atom_coulomb:
#             # Coulomb correction term
#             E_correct_atom = k_e * cal_coulomb_E(Qi, atom_mol_batch, lr_atom_edge_dist,
#                                                  lr_atom_edge_index, self.atom_feature_dist)
#             E_mol_atomic = E_correct_atom + E_mol_atomic
#
#         if self.cal_efg_coulomb:
#             # Coulomb correction term
#             if self.SAGPool_EFG:
#                 print('ERROR: SAGPooling=True, you cannot calculate EFG level coulomb yet. Exiting...')
#                 exit(0)
#             E_correct_efg = k_e * cal_coulomb_E(Q_efg, efg_mol_batch, lr_efg_edge_dist,
#                                                  lr_efg_edge_index, self.efg_feature_dist)
#             E_mol_efgs = E_correct_efg + E_mol_efgs
#
#         # debug: cal passed time to improve code
#         # print('T------coulomb correction: ', time.time() - t0)
#
#         if self.cal_atom_dispersion:
#             # calculate grimme D3 dispersion term
#             E_d3_atom = utils.grimme_d3.d3_autoev * \
#                    utils.grimme_d3.cal_d3_dispersion(Z, atom_mol_batch,
#                                                      lr_atom_edge_dist, lr_atom_edge_index,
#                                                      self.s6, self.s8, self.a1, self.a2)
#             E_mol_atomic = E_d3_atom + E_mol_atomic
#
#         # debug: cal passed time to improve code
#         # print('T------dispersion correction: ', time.time() - t0)
#
#         if self.debug_mode:
#             if not self.cal_atom_coulomb:
#                 E_correct_atom = torch.zeros(1)
#             if not self.cal_atom_dispersion:
#                 E_d3_atom = torch.zeros(1)
#             if not self.cal_efg_coulomb:
#                 E_correct_efg = torch.zeros(1)
#             print('Atomic Energy: {:.3f}, Coulomb: {:.3f}, Dispersion: {:.3f}'.format(E_mol_atomic.mean(),
#                                                                                       E_correct_atom.mean(),
#                                                                                       E_d3_atom.mean()))
#             print('EFG Level Energy: {:.3f}, Coulomb: {:.3f}, Dispersion: None'.format(E_mol_efgs.mean(),
#                                                                                        E_correct_efg.mean()))
#
#         E_pred = E_mol_atomic * self.atom_prop_weight + E_mol_efgs * (1 - self.atom_prop_weight)
#         Q_pred = Q_mol_atomic * self.atom_prop_weight + Q_mol_efgs * (1 - self.atom_prop_weight)
#
#         F_pred = None
#         if self.cal_force:
#             # F prediction term by calculation dE/dR
#             F_pred = -torch.autograd.grad(E_pred, R, torch.ones_like(E_pred),
#                                           create_graph=True, retain_graph=True)[0]
#             F_pred = F_pred.to(device)
#             R.requires_grad_(False)
#
#         # calculate dipole from predicted partial q and coordinates of atoms
#         p_pred = cal_p(Qi, R, atom_mol_batch)
#
#         # debug: cal passed time to improve code
#         # print('T------cal force: ', time.time() - t0)
#
#         return E_pred, F_pred, Q_pred, p_pred, loss_nh
