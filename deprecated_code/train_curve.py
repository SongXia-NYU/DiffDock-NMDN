# import sys
# import argparse
# import logging
# import math
# import os
# import shutil
# import time
# from datetime import datetime
#
# import numpy as np
# import torch
# import torch.cuda
# import torch.utils.data
# from warmup_scheduler import GradualWarmupScheduler
#
# from Frag20n9MixIMDataset import Frag20n9MixIMDataset
# from Frag9IMDataset import Frag9IMDataset
# from Networks.PhysDimeNet import PhysDimeNet
# from DataPrepareUtils import pre_transform
# from qm9InMemoryDataset import Qm9InMemoryDataset
# from utils.EMA_AMSgrad import EmaAmsGrad
# from utils.LossFn import LossFn
# from utils.time_meta import print_function_runtime
# from utils.utils_functions import device, add_parser_arguments, floating_type, kwargs_solver, get_lr, collate_fn, \
#     load_data_from_index, get_n_params, cal_mean_std
#
#
# def train_step(model, _optimizer, data_batch, loss_fn, max_norm, warm_up_scheduler):
#     # torch.cuda.synchronize()
#     # t0 = time.time()
#
#     model.train()
#     _optimizer.zero_grad()
#
#     E_pred, F_pred, Q_pred, p_pred, loss_nh = model(data_batch)
#
#     # t0 = record_data('forward', t0, True)
#
#     loss = loss_fn(E_pred, F_pred, Q_pred, p_pred, data_batch) + loss_nh
#
#     loss.backward()
#
#     # t0 = record_data('backward', t0, True)
#
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#     _optimizer.step()
#     warm_up_scheduler.step()
#
#     # t0 = record_data('step', t0, True)
#
#     result_loss = loss.data[0]
#
#     return result_loss
#
#
# def val_step(model, _data_loader, data_size, loss_fn, mae_fn, mse_fn, dataset_name='dataset'):
#     log_info = f'Validating {dataset_name}: '
#
#     model.eval()
#     loss, emae, emse, fmae, fmse, qmae, qmse, pmae, pmse = 0, 0, 0, 0, 0, 0, 0, 0, 0
#     for val_data in _data_loader:
#         _batch_size = len(val_data.E)
#
#         E_pred, F_pred, Q_pred, D_pred, loss_nh = model(val_data)
#
#         # IMPORTANT the .item() function is necessary here, otherwise the graph will be unintentionally stored and
#         # never be released
#         # And you will run out of memory after several val()
#         loss1 = loss_fn(E_pred, F_pred, Q_pred, D_pred, val_data).item()
#         loss2 = loss_nh.item()
#         loss += _batch_size * (loss1 + loss2)
#
#         emae += _batch_size * mae_fn(E_pred, val_data.E.to(device)).item()
#         emse += _batch_size * mse_fn(E_pred, val_data.E.to(device)).item()
#
#         qmae += _batch_size * mae_fn(Q_pred, val_data.Q.to(device)).item()
#         qmse += _batch_size * mse_fn(Q_pred, val_data.Q.to(device)).item()
#
#         pmae += _batch_size * mae_fn(D_pred, val_data.D.to(device)).item()
#         pmse += _batch_size * mse_fn(D_pred, val_data.D.to(device)).item()
#
#     loss = loss / data_size
#     emae = emae / data_size
#     ermse = math.sqrt(emse / data_size)
#     qmae = qmae / data_size
#     qrmse = math.sqrt(qmse / data_size)
#     pmae = pmae / data_size
#     prmse = math.sqrt(pmse / data_size)
#
#     log_info += (' loss: {:.6f} '.format(loss))
#     log_info += ('emae: {:.6f} '.format(emae))
#     log_info += ('ermse: {:.6f} '.format(ermse))
#     log_info += ('qmae: {:.6f} '.format(qmae))
#     log_info += ('qrmse: {:.6f} '.format(qrmse))
#     log_info += ('pmae: {:.6f} '.format(pmae))
#     log_info += ('prmse: {:.6f} '.format(prmse))
#
#     logging.info(log_info)
#
#     return {'loss': loss, 'emae': emae, 'ermse': ermse, 'qmae': qmae, 'qrmse': qrmse, 'pmae': pmae, 'prmse': prmse}
#
#
# def train(config_args, data_provider_class, data_provider_kwargs):
#     net_kwargs = kwargs_solver(config_args)
#
#     debug_mode = (config_args.debug_mode.lower() == 'true')
#     use_trained_model = (config_args.use_trained_model.lower() == 'true')
#
#     # bond_atom_sep = ('B' in bonding_type_list) or ('N' in bonding_type_list)
#     # # NOTICE: Potential bug, this setting will calculate bonding by default
#     # cal_3body_term = ('D' in modules_list)
#     # net_kwargs['cal_3body_term'] = cal_3body_term
#     # net_kwargs['bond_atom_sep'] = bond_atom_sep
#     # data_provider_kwargs['cal_3body_term'] = cal_3body_term
#     # data_provider_kwargs['bond_atom_sep'] = bond_atom_sep
#     data_provider_kwargs['edge_version'] = args.edge_version
#     data_provider_kwargs['cutoff'] = args.cutoff
#     data_provider_kwargs['boundary_factor'] = args.boundary_factor
#     data_provider = data_provider_class(**data_provider_kwargs)
#
#     # set up run directory
#     if not debug_mode:
#         current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
#         run_directory = config_args.folder_prefix + '_run_' + current_time
#         os.mkdir(run_directory)
#     else:
#         run_directory = 'debug'
#         if not os.path.exists(run_directory):
#             os.mkdir(run_directory)
#
#     shutil.copyfile(config_name, os.path.join(run_directory, config_name))
#
#     # Logger setup
#     logging.basicConfig(filename=os.path.join(run_directory, config_args.log_file_name),
#                         format='%(asctime)s %(message)s',
#                         filemode='w')
#     logger = logging.getLogger()
#     logger.setLevel(logging.DEBUG)
#
#     train_index, val_index, test_index = data_provider.train_index, data_provider.val_index, data_provider.test_index
#     train_index_frag9 = train_index[:90000]
#     train_index_frag20 = train_index[90000:99000]
#     val_index_frag9 = val_index[:700]
#     val_index_frag20 = val_index[700:]
#
#     # select part of frag9 or frag20 to train
#     logger.info('THIS IS A TRAINING CURVE EXPERIMENT')
#     logger.info(f'frag9 size: {frag9_train_size}; frag20 size: {frag20_train_size}')
#     train_index = torch.cat([train_index_frag9[:frag9_train_size], train_index_frag20[:frag20_train_size]])
#
#     '''
#     DEBUG ONLY!!! PLEASE COMMENT THE LINE BELOW
#     '''
#     # train_index = train_index[:1000]
#     # val_index = torch.LongTensor([2])
#     # val_index = val_index[:1]
#
#     train_size = len(train_index)
#     val_size = len(val_index)
#     test_size = len(test_index)
#     logger.info('data size: ' + str(train_size + val_size + test_size))
#     logger.info('train size: ' + str(train_size))
#     logger.info('validation size: ' + str(val_size))
#     logger.info('test size: ' + str(test_size))
#     num_train_batches = train_size // config_args.batch_size + 1
#
#     t0 = time.time()
#     '''
#     There are some bugs when using torch.DataLoader, we may fix it later on(or may not :().
#     Usually, it's sufficient to set use_torch_loader = False
#     '''
#     use_torch_loader = False
#     if use_torch_loader:
#         train_data_loader = torch.utils.data.DataLoader(
#             data_provider[train_index], batch_size=config_args.batch_size, collate_fn=collate_fn,
#             pin_memory=torch.cuda.is_available())
#
#         val_data_loader = torch.utils.data.DataLoader(
#             data_provider[val_index], batch_size=config_args.batch_size, collate_fn=collate_fn,
#             pin_memory=torch.cuda.is_available())
#     else:
#         train_index_loader = torch.utils.data.DataLoader(train_index, batch_size=config_args.batch_size)
#         train_data_loader = [load_data_from_index(data_provider, idx_name) for idx_name in train_index_loader]
#         valid_index_frag9_loader = torch.utils.data.DataLoader(val_index_frag9, batch_size=config_args.valid_batch_size)
#         val_frag9_loader = [load_data_from_index(data_provider, idx_name) for idx_name in valid_index_frag9_loader]
#         valid_index_frag20_loader = torch.utils.data.DataLoader(val_index_frag20, batch_size=config_args.valid_batch_size)
#         val_frag20_loader = [load_data_from_index(data_provider, idx_name) for idx_name in valid_index_frag20_loader]
#     logging.info('prepare data, time spent: {:.1f}s'.format(time.time() - t0))
#
#     mae_fn = torch.nn.L1Loss(reduction='mean')
#     mse_fn = torch.nn.MSELoss(reduction='mean')
#     w_e, w_f, w_q, w_p = 1, config_args.force_weight, config_args.charge_weight, config_args.dipole_weight
#     loss_fn = LossFn(w_e=w_e, w_f=w_f, w_q=w_q, w_p=w_p)
#
#     if torch.cuda.is_available():
#         logger.info('device name: ' + torch.cuda.get_device_name(device))
#         logger.info("Cuda mem allocated: {:.2f} MB".format(torch.cuda.memory_allocated(device) * 1e-6))
#
#     # Normalization of PhysNet atom-wise prediction
#     E_mean_atom, E_std_atom = cal_mean_std(data_provider.data.E, data_provider.data.N, train_index)
#     E_atomic_scale = E_std_atom
#     E_atomic_shift = E_mean_atom
#
#     net_kwargs['energy_shift'] = E_atomic_shift
#     net_kwargs['energy_scale'] = E_atomic_scale
#     _info = ''
#     for _key in net_kwargs.keys():
#         logger.info("{} = {}".format(_key, net_kwargs[_key]))
#
#     net = PhysDimeNet(**net_kwargs)
#     shadow_net = PhysDimeNet(**net_kwargs)
#     net = net.to(device)
#     net = net.type(floating_type)
#
#     print('model params: ', get_n_params(net, logger))
#
#     shadow_net = shadow_net.to(device)
#     shadow_net = shadow_net.type(floating_type)
#
#     if use_trained_model:
#         model_data = torch.load('best_model.pt', map_location=device)
#         net.load_state_dict(model_data)
#         shadow_net.load_state_dict(model_data)
#     logger.info("Model #Params: {}".format(get_n_params(net)))
#     shadow_net.load_state_dict(net.state_dict())
#     optimizer = EmaAmsGrad(net.parameters(), shadow_net, lr=config_args.learning_rate, ema=config_args.ema_decay)
#     logger.info('Init lr: {}'.format(get_lr(optimizer)))
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config_args.decay_steps, gamma=0.1)
#     warm_up_scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=3000, after_scheduler=scheduler)
#
#     logger.info('start training...')
#     best_loss = np.inf
#
#     shadow_net = optimizer.shadow_model
#     val_loss = val_step(shadow_net, val_frag9_loader, len(val_index_frag9), loss_fn=loss_fn, mae_fn=mae_fn,
#                         mse_fn=mse_fn,
#                         dataset_name='frag9')
#     val_loss = val_step(shadow_net, val_frag20_loader, len(val_index_frag20), loss_fn=loss_fn, mae_fn=mae_fn,
#                         mse_fn=mse_fn,
#                         dataset_name='frag20')
#
#     # The line below is for debug only
#     # tmp = print_function_runtime(logger)
#     # exit()
#
#     loss_data = []
#
#     for epoch in range(config_args.num_epochs):
#         train_loss = 0.
#         for batch_num, data in enumerate(train_data_loader):
#             this_size = len(data)
#
#             train_loss += train_step(net, _optimizer=optimizer, data_batch=data, loss_fn=loss_fn,
#                                      max_norm=config_args.max_norm,
#                                      warm_up_scheduler=warm_up_scheduler) * this_size / train_size
#
#             if debug_mode & ((batch_num + 1) % 600 == 0):
#                 logger.info("Batch num: {}/{}, train loss: {} ".format(batch_num, num_train_batches, train_loss))
#
#         logger.info('epoch {} ended, learning rate: {} '.format(epoch, get_lr(optimizer)))
#         shadow_net = optimizer.shadow_model
#         # torch.save(shadow_net.state_dict(), os.path.join(run_directory, 'model_ckpt_'+str(epoch)+'.pt'))
#         val_loss = val_step(shadow_net, val_frag9_loader, len(val_index_frag9), loss_fn=loss_fn, mae_fn=mae_fn,
#                             mse_fn=mse_fn,
#                             dataset_name='frag9')
#         val_loss = val_step(shadow_net, val_frag20_loader, len(val_index_frag20), loss_fn=loss_fn, mae_fn=mae_fn,
#                             mse_fn=mse_fn,
#                             dataset_name='frag20')
#         _loss_data_this_epoch = {'epoch': epoch,
#                                  't_loss': train_loss,
#                                  'v_loss': val_loss['loss'],
#                                  'v_emae': val_loss['emae'],
#                                  'v_ermse': val_loss['ermse'],
#                                  'v_qmae': val_loss['qmae'],
#                                  'v_qrmse': val_loss['qrmse'],
#                                  'v_pmae': val_loss['pmae'],
#                                  'v_prmse': val_loss['prmse'],
#                                  'time': time.time()}
#         loss_data.append(_loss_data_this_epoch)
#         torch.save(loss_data, os.path.join(run_directory, 'loss_data.pt'))
#         # torch.save(optimizer.state_dict(), os.path.join(run_directory, 'optimizer.pt'))
#         if val_loss['loss'] < best_loss:
#             best_loss = val_loss['loss']
#             torch.save(shadow_net.state_dict(), os.path.join(run_directory, 'best_model.pt'))
#
#     print_function_runtime(logger)
#
#
# if __name__ == "__main__":
#     # set up parser and arguments
#     parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
#     parser = add_parser_arguments(parser)
#
#     # parse config file
#     config_name = 'config-exp315.txt'
#     if len(sys.argv) == 1:
#         if os.path.isfile(config_name):
#             args = parser.parse_args(["@" + config_name])
#         else:
#             raise Exception('couldn\'t find \"config-exp315.txt\" !')
#     else:
#         args = parser.parse_args()
#         config_name = args.config_name
#         frag9_train_size = args.frag9_train_size
#         frag20_train_size = args.frag20_train_size
#         args = parser.parse_args(["@" + config_name])
#
#     # define data provider
#     # If you want to change input data, just write your own data provider
#     # See readme.md for more details
#     # qm9_provider = Qm9InMemoryDataset(root='data', edge_version=args.edge_version, cutoff=args.cutoff,
#     #                                   boundary_factor=args.boundary_factor, pre_transform=pre_transform)
#     # frag9_provider = Frag9IMDataset(root='data', edge_version=args.edge_version, cutoff=args.cutoff,
#     #                                 boundary_factor=args.boundary_factor, pre_transform=pre_transform)
#     _kwargs = {'root': '../dataProviders/data', 'pre_transform': pre_transform, 'record_long_range': True,
#                'train_split': (90000, 9000), 'val_split': (700, 700), 'test_split': (500, 500)}
#     train(args, Frag20n9MixIMDataset, _kwargs)
