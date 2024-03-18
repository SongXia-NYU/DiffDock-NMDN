import logging
import math
from typing import Union, List

import torch

from utils.utils_functions import floating_type, print_val_results


def val_step(model, _data_loader, data_size, loss_fn, mae_fn, mse_fn, dataset_name='dataset', detailed_info=False,
             print_to_log=True, action: Union[List[str], str] = "E"):
    print("this valid step is deprecated_code.")
    model.eval()
    loss, emae, emse, fmae, fmse, qmae, qmse, pmae, pmse = 0, 0, 0, 0, 0, 0, 0, 0, 0

    if detailed_info:
        E_pred_aggr = torch.zeros(data_size).cpu().type(floating_type)
        Q_pred_aggr = torch.zeros_like(E_pred_aggr)
        D_pred_aggr = torch.zeros(data_size, 3).cpu().type(floating_type)
        idx_before = 0
    else:
        E_pred_aggr, Q_pred_aggr, D_pred_aggr, idx_before = None, None, None, None

    for val_data in _data_loader:
        _batch_size = len(val_data.E)

        E_pred, F_pred, Q_pred, D_pred, loss_nh = model(val_data)

        # IMPORTANT the .item() function is necessary here, otherwise the graph will be unintentionally stored and
        # never be released
        # And you will run out of memory after several val()
        loss1 = loss_fn(E_pred, F_pred, Q_pred, D_pred, val_data).item()
        loss2 = loss_nh.item()
        loss += _batch_size * (loss1 + loss2)

        emae += _batch_size * mae_fn(E_pred, getattr(val_data, action).to(device)).item()
        emse += _batch_size * mse_fn(E_pred, getattr(val_data, action).to(device)).item()

        if Q_pred is not None:
            qmae += _batch_size * mae_fn(Q_pred, val_data.Q.to(device)).item()
            qmse += _batch_size * mse_fn(Q_pred, val_data.Q.to(device)).item()

            pmae += _batch_size * mae_fn(D_pred, val_data.D.to(device)).item()
            pmse += _batch_size * mse_fn(D_pred, val_data.D.to(device)).item()

        if detailed_info:
            for aggr, pred in zip([E_pred_aggr, Q_pred_aggr, D_pred_aggr], [E_pred, Q_pred, D_pred]):
                if pred is not None:
                    aggr[idx_before: _batch_size + idx_before] = pred.detach().cpu()
                # aggr[idx_before: _batch_size + idx_before] = val_data.E.detach().cpu().view(-1)
            idx_before = idx_before + _batch_size

    loss = loss / data_size
    emae = emae / data_size
    ermse = math.sqrt(emse / data_size)
    qmae = qmae / data_size
    qrmse = math.sqrt(qmse / data_size)
    pmae = pmae / data_size
    prmse = math.sqrt(pmse / data_size)

    if print_to_log:
        log_info = print_val_results(dataset_name, loss, emae, ermse, qmae, qrmse, pmae, prmse)
        logging.info(log_info)

    result = {'loss': loss, 'emae': emae, 'ermse': ermse, 'qmae': qmae, 'qrmse': qrmse, 'pmae': pmae, 'prmse': prmse}

    if detailed_info:
        result['E_pred'] = E_pred_aggr
        result['Q_pred'] = Q_pred_aggr
        result['D_pred'] = D_pred_aggr

    return

#
# def get_test_set(dataset_name, args):
#     _, arg_in_name = data_provider_solver(dataset_name, {})
#     if dataset_name == 'conf20':
#         dataset = PhysDimeIMDataset(processed_prefix='conf20', root='../dataProviders/data',
#                                     pre_transform=my_pre_transform,
#                                     infile_dic={'PhysNet': 'conf20_QM_PhysNet.npz', 'SDF': 'conf20_QM.pt'})
#     elif dataset_name.split('[')[0] == 'frag9to20_all' and args.action == "E" and (not arg_in_name["add_sol"]):
#         # convert all to jianing split for consistent split
#         dataset_name = dataset_name.replace('all', 'jianing', 1)
#         dataset_cls, kw_arg = data_provider_solver(dataset_name, {'root': '../dataProviders/data',
#                                                                   'pre_transform': my_pre_transform,
#                                                                   'record_long_range': True,
#                                                                   'type_3_body': 'B',
#                                                                   'cal_3body_term': True})
#         dataset_train = dataset_cls(**kw_arg)
#         kw_arg['training_option'] = 'test'
#         return dataset_train, dataset_cls(**kw_arg)
#     else:
#         dataset_cls, _kwargs = data_provider_solver(dataset_name, default_kwargs)
#         _kwargs = _add_arg_from_config(_kwargs, args)
#         dataset = dataset_cls(**_kwargs)
#     print("used dataset: {}".format(dataset.processed_file_names))
#     return dataset


    # if test_dataset == 'platinum':
    #     sep_heavy_atom = False
    #     if sep_heavy_atom:
    #         for i in range(10, 21):
    #             data_provider = PlatinumTestIMDataSet('../dataProviders/data', pre_transform=my_pre_transform,
    #                                                   sep_heavy_atom=True, num_heavy_atom=i,
    #                                                   cal_3body_term=True, bond_atom_sep=True, record_long_range=True)
    #
    #             test_index = torch.arange(len(data_provider))
    #             test_data_loader = torch.utils.data.DataLoader(
    #                 data_provider[torch.as_tensor(test_index)], batch_size=32, collate_fn=collate_fn,
    #                 pin_memory=torch.cuda.is_available(), shuffle=False)
    #
    #             test_step(args, net, test_data_loader, len(data_provider), loss_fn=loss_fn, mae_fn=mae_fn,
    #                       mse_fn=mse_fn,
    #                       dataset_name='platinum_{}'.format(i), run_dir=test_dir, n_forward=n_forward)
    #     else:
    #         test_data = PlatinumTestIMDataSet('../dataProviders/data', pre_transform=my_pre_transform,
    #                                           sep_heavy_atom=False,
    #                                           num_heavy_atom=None, cal_3body_term=False, bond_atom_sep=False,
    #                                           record_long_range=True, qm=False)
    #
    #         test_index = test_data.test_index
    #         test_data_loader = torch.utils.data.DataLoader(test_data[torch.as_tensor(test_index)], batch_size=32,
    #                                                        collate_fn=collate_fn, pin_memory=torch.cuda.is_available(),
    #                                                        shuffle=False)
    #         # ------------------------- Absolute Error -------------------------- #
    #         test_info, test_info_std = test_step(args, net, test_data_loader, len(test_index), loss_fn=loss_fn,
    #                                              mae_fn=mae_fn, mse_fn=mse_fn,
    #                                              dataset_name='platinum', run_dir=test_dir, n_forward=n_forward)
    #         E_pred = test_info['E_pred']
    #         test_info_analyze(23.061 * E_pred, 23.061 * test_data.data.E[test_index], test_dir, logger)
    #         csv1 = pd.read_csv("../dataProviders/data/raw/Plati20_index_10_13.csv")
    #         csv2 = pd.read_csv("../dataProviders/data/raw/Plati20_index_14_20.csv")
    #         mol_batch = torch.cat([torch.as_tensor(csv1["molecule_id"].values).view(-1),
    #                                torch.as_tensor(csv2["molecule_id"].values).view(-1)])
    #
    #         conf_id = torch.cat([torch.as_tensor(csv1["idx_name"].values).view(-1),
    #                              torch.as_tensor(csv2["idx_name"].values).view(-1)])
    #
    #         overlap_mask = torch.zeros_like(mol_batch).bool().fill_(True)
    #         overlap1 = pd.read_csv("../dataProviders/data/raw/overlap_molecules_10_13.csv", header=None).values
    #         overlap_mask[overlap1] = False
    #         overlap2 = pd.read_csv("../dataProviders/data/raw/overlap_molecules_14_20.csv", header=None).values
    #         overlap_mask[overlap2 + len(csv1["molecule_id"].values)] = False
    #
    #         # -------------------------- Relative Error ------------------------- #
    #         mol_batch = mol_batch[overlap_mask]
    #         conf_id = conf_id[overlap_mask]
    #         E_pred = torch.load(osp.join(test_dir, "loss.pt"))['E_pred'].view(-1)[overlap_mask]
    #         E_tgt = test_data.data.E.view(-1)[overlap_mask]
    #         n_mol = mol_batch[-1].item()
    #         lowest_e_tgt = torch.zeros_like(mol_batch).double()
    #         lowest_e_pred = torch.zeros_like(mol_batch).double()
    #         lowest_e_id_tgt = torch.zeros(n_mol).long().view(-1)
    #         lowest_e_id_pred = torch.zeros(n_mol).long().view(-1)
    #         for i in range(0, n_mol):
    #             mask = (mol_batch == i + 1)
    #             if mask.sum() == 0:
    #                 continue
    #             lowest_e_pred[mask] = E_pred[mask].min()
    #             lowest_e_tgt[mask] = E_tgt[mask].min()
    #             lowest_e_id_tgt[i] = conf_id[mask][E_tgt[mask].argmin()]
    #             lowest_e_id_pred[i] = conf_id[mask][E_pred[mask].argmin()]
    #         r_e_tgt = E_tgt - lowest_e_tgt
    #         r_e_pred = E_pred - lowest_e_pred
    #         mol_mae = scatter(reduce="mean", src=(r_e_tgt - r_e_pred).abs(), idx_name=mol_batch - 1, dim=0)
    #         mol_rmse = torch.sqrt(scatter(reduce="mean", src=(r_e_tgt - r_e_pred) ** 2, idx_name=mol_batch - 1, dim=0))
    #         logger.info("Relative EMAE: {}, ERMSE: {}, sucess rate: {}%".format(
    #             mol_mae.mean(), mol_rmse.mean(), (lowest_e_id_pred == lowest_e_id_tgt).sum() * 100. / n_mol))
    # elif test_dataset in ['csd20_qm', 'csd20_mmff']:
    #     _data = PhysDimeIMDataset(root='../dataProviders/data', processed_prefix=test_dataset.upper(),
    #                               pre_transform=my_pre_transform,
    #                               record_long_range=True)
    #     test_index = _data.test_index
    #     test_data_loader = torch.utils.data.DataLoader(
    #         _data[torch.as_tensor(test_index)], batch_size=args.valid_batch_size, collate_fn=collate_fn,
    #         pin_memory=torch.cuda.is_available(), shuffle=False)
    #     test_info, test_info_std = test_step(args, net, test_data_loader, len(test_index), loss_fn=loss_fn,
    #                                          mae_fn=mae_fn, mse_fn=mse_fn, dataset_name='{}_test'.format(test_dataset),
    #                                          run_dir=test_dir, n_forward=n_forward)
    #     E_pred = test_info['E_pred']
    #     test_info_analyze(23.061 * E_pred, 23.061 * _data.data.E[test_index], test_dir, logger)
    # elif test_dataset.split('_')[0].split('[')[0] in ['qm9', 'frag9', 'frag9to20', 'qm9+extBond', 'conf20', "frag20",
    #                                                   "dummy"]:
    #     data_provider = get_test_set(test_dataset, args)
    #     if isinstance(data_provider, tuple):
    #         data_provider_test = data_provider[1]
    #         data_provider = data_provider[0]
    #     else:
    #         data_provider_test = data_provider
    #     val_index = data_provider.val_index
    #     test_index = data_provider_test.test_index
    #     logger.info("dataset: {}".format(data_provider.processed_file_names))
    #     logger.info("valid size: {}".format(len(val_index)))
    #     logger.info("test size: {}".format(len(test_index)))
    #     if args.remove_atom_ids > 0:
    #         _, val_index, _ = remove_atom_from_dataset(args.remove_atom_ids, data_provider, ("valid",),
    #                                                    (None, val_index, None))
    #         logger.info('removing B from test dataset...')
    #         _, _, test_index = remove_atom_from_dataset(args.remove_atom_ids, data_provider_test, ('test',),
    #                                                     (None, None, test_index))
    #
    #     if val_index is not None:
    #         val_data_loader = torch.utils.data.DataLoader(
    #             data_provider[torch.as_tensor(val_index)], batch_size=args.valid_batch_size, collate_fn=collate_fn,
    #             pin_memory=torch.cuda.is_available(), shuffle=False)
    #         test_step(args, net, val_data_loader, len(val_index), loss_fn=loss_fn, mae_fn=mae_fn, mse_fn=mse_fn,
    #                   dataset_name='{}_valid'.format(test_dataset), run_dir=test_dir, n_forward=n_forward,
    #                   action=args.action)
    #     test_data_loader = torch.utils.data.DataLoader(
    #         data_provider_test[torch.as_tensor(test_index)], batch_size=args.valid_batch_size, collate_fn=collate_fn,
    #         pin_memory=torch.cuda.is_available(), shuffle=False)
    #     test_info, test_info_std = test_step(args, net, test_data_loader, len(test_index), loss_fn=loss_fn,
    #                                          mae_fn=mae_fn, mse_fn=mse_fn, dataset_name='{}_test'.format(test_dataset),
    #                                          run_dir=test_dir, n_forward=n_forward, action=args.action)
    #     # if not os.path.exists(os.path.join(test_directory, 'loss.pt')):
    #     #     loss = cal_loss(test_info, data_provider_test.data.E[test_index], data_provider_test.data.D[test_index],
    #     #                     data_provider_test.data.Q[test_index], mae_fn=mae_fn, mse_fn=mse_fn)
    #     #     torch.save(loss, os.path.join(test_directory, 'loss.pt'))
    #     if "E_pred" in test_info:
    #         E_pred = test_info['E_pred']
    #         if test_info_std is not None:
    #             E_pred_std = test_info_std['E_pred']
    #         else:
    #             E_pred_std = None
    #         test_info_analyze(23.061 * E_pred, 23.061 * data_provider_test.data.E[test_index],
    #                           test_dir, logger, pred_std=E_pred_std, x_forward=x_forward)
    # elif test_dataset.split(':')[0] == 'frag20n9':
    #     data_provider = Frag9to20MixIMDataset(root='../dataProviders/data', split_settings=uniform_split,
    #                                           pre_transform=my_pre_transform, frag20n9=True)
    #     n_frag9_val = int(test_dataset.split(':')[1].split('+')[2])
    #     n_frag20_val = int(test_dataset.split(':')[1].split('+')[3])
    #     val_data_loader = torch.utils.data.DataLoader(
    #         data_provider[108000: 108000 + n_frag9_val + n_frag20_val], batch_size=32, collate_fn=collate_fn,
    #         pin_memory=torch.cuda.is_available(), shuffle=False)
    #     test_data_loader = torch.utils.data.DataLoader(
    #         data_provider[-1000:-500], batch_size=32, collate_fn=collate_fn,
    #         pin_memory=torch.cuda.is_available(), shuffle=False)
    #     val_step(net, val_data_loader, n_frag9_val + n_frag20_val, loss_fn=loss_fn, mae_fn=mae_fn, mse_fn=mse_fn,
    #              dataset_name="{} validation set".format(test_dataset))
    #     test_info = val_step(net, test_data_loader, 500, loss_fn=loss_fn, mae_fn=mae_fn, mse_fn=mse_fn,
    #                          dataset_name="{} test set".format(test_dataset), detailed_info=True)
    #     test_info_analyze(23.061 * test_info['E_pred'], 23.061 * test_info['E_target'], test_dir, logger)
    # else:
    #     print('unrecognized test set: {}'.format(test_dataset))
