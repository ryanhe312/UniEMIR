import argparse
import os
import warnings
import torch
import torch.multiprocessing as mp

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric

def main_worker(gpu, ngpus_per_node, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=opt['init_method'],
                                             world_size=opt['world_size'],
                                             rank=opt['global_rank'],
                                             group_name='mtorch'
                                             )
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = False
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and data'''
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]
    phase_loaders, val_loaders = [], []

    from copy import deepcopy
    datasets_opt = deepcopy(opt['datasets'][opt['phase']]['which_dataset'])
    print('datasets_opt:', datasets_opt)

    for dataset_opt in datasets_opt:
        opt['datasets'][opt['phase']]['which_dataset'] = dataset_opt
        phase_loader, val_loader = define_dataloader(phase_logger, opt)  
        print(len(phase_loader), opt['local_rank'])
        phase_loaders.append(phase_loader)
        val_loaders.append(val_loader)
    opt['datasets'][opt['phase']]['which_dataset'] = datasets_opt

    ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt=opt,
        networks=networks,
        phase_loader=phase_loaders,
        val_loader=val_loaders,
        losses=losses,
        metrics=metrics,
        logger=phase_logger,
        writer=phase_writer
    )

    # calculate flops 
    # from thop import profile, clever_format
    # input = torch.randn(1, 1, 1, 256, 256).cuda()
    # flops, params = profile(networks[0].model, inputs=input)
    # print(clever_format([flops, params]))
    # quit()

    phase_logger.info('Begin model {}.'.format(opt['phase']))

    if opt['phase'] == 'train':
        model.train()
    else:
        model.test()

    phase_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='JSON file for configuration')
    parser.add_argument('--path', type=str, default=None, help='patch of cropped patches')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], help='Run train or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('--gpu', type=str, default=None, help='the gpu devices used')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-z', '--z_times', default=None, type=int, help='The anisotropy time of the volume em')
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('--mean', type=int, default=1, help='Test-time augmentation')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--percent', type=float, default=1, help='Percentage of the training data')
    parser.add_argument('--resume', type=str, default=None, help='Resume state path and load epoch number')

    ''' parser configs '''
    args = parser.parse_args()

    opt = Praser.parse(args)

    if args.percent != 1:
        opt['datasets'][opt['phase']]['which_dataset'][0]["args"]["percent"] = args.percent

    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids'])  # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:' + args.port
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1
        main_worker(0, 1, opt)
