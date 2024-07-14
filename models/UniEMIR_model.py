import torch
import tqdm.auto as tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
import os
import shutil
import torch.nn as nn
import torchvision.transforms.functional as F
import core.util as Util
from warmup_scheduler import GradualWarmupScheduler

class UniEMIR(BaseModel):
    def __init__(self, networks, losses, optimizers, task=0, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(UniEMIR, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]

        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        self.load_networks()
        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        self.schedulers.append(GradualWarmupScheduler(self.optG, multiplier=1, total_epoch=20))
        self.resume_training()

        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
        else:
            self.netG.set_loss(self.loss_fn)

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.task = task
        self.scaler = torch.cuda.amp.GradScaler()

    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.mask = self.set_device(data.get('mask'))
        self.mask_image = data.get('mask_image')
        self.img_min = data.get('img_min', None)
        self.img_max = data.get('img_max', None)
        self.gt_min = data.get('gt_min', None)
        self.gt_max = data.get('gt_max', None)
        self.path = data['path']
        self.batch_size = len(data['path'])

    def get_current_visuals(self, phase='train', task = 0):
        if task < 3:
            dict = {
                'gt_image': (self.gt_image.detach()[:].float().cpu() + 1) / 2,
                'cond_image': (self.cond_image.detach()[:].float().cpu() + 1) / 2,
            }
            if phase != 'train':
                dict.update({
                    'output': (self.output.detach()[:].float().cpu() + 1) / 2
                })
        else:
            dict = {
                'gt_image': (self.gt_image[:,0:1].detach().float().cpu() + 1) / 2,
                'cond_image': (self.cond_image[:,0:1].detach().float().cpu() + 1) / 2,
            }
            if phase != 'train':
                dict.update({
                    'output': (self.output[:,0:1].detach().float().cpu() + 1) / 2
                })

        return dict

    def save_current_results(self, task = 0):
        ret_path = []
        ret_result = []
        if task < 3:
            for idx in range(self.batch_size):
                ret_path.append('GT_{}'.format(self.path[idx]))
                ret_result.append(self.gt_image[idx].detach().float().cpu())
                ret_path.append('Out_{}'.format(self.path[idx]))
                ret_result.append(self.output[idx].detach().float().cpu())
                ret_path.append('Input_{}'.format(self.path[idx]))
                ret_result.append(self.cond_image[idx].detach().float().cpu())
                if self.mean > 1 and self.phase == 'test':
                    for output_index in range(len(self.outputs)):
                        ret_path.append('Out_round{}_{}'.format(output_index, self.path[idx]))
                        ret_result.append(self.outputs[output_index][idx].detach().float().cpu())
                else:
                    ret_path.append('Out_uncertainty_{}'.format(self.path[idx]))
                    ret_result.append(self.output[idx].detach().float().cpu())
            self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        else:
            for idx in range(self.batch_size):
                for i in range(self.gt_image.shape[1]):
                    ret_path.append('GT_{}_{}'.format(i, self.path[idx]))
                    ret_result.append(self.gt_image[idx,i,:,: ].detach().float().cpu())
                    ret_path.append('Out_{}_{}'.format(i, self.path[idx]))
                    ret_result.append(self.output[idx, i, :, :].detach().float().cpu())
                # ret_path.append('Process_{}'.format(self.path[idx]))
                # ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())
                ret_path.append('Input_upper_{}'.format(self.path[idx]))
                ret_result.append(self.cond_image[idx, 0, :, :].detach().float().cpu())
                ret_path.append('Input_lower_{}'.format(self.path[idx]))
                ret_result.append(self.cond_image[idx, 1, :, :].detach().float().cpu())

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()
    
    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        pbar = tqdm.tqdm(self.phase_loader[self.epoch % len(self.phase_loader)])
        self.task = self.opt['datasets'][self.opt['phase']]['which_dataset'][self.epoch % len(self.phase_loader)]['task']

        print('train step at epoch {} rank {} task {}'.format(self.epoch, self.opt['local_rank'], self.task))
        if self.opt['distributed']:
            ''' sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas use a different random ordering for each epoch '''
            self.phase_loader[self.epoch % len(self.phase_loader)].sampler.set_epoch(self.epoch) 

        for i, train_data in enumerate(pbar):
            self.set_input(train_data)
            self.optG.zero_grad()

            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask, task=self.task)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optG)
            self.scaler.update()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)

            pbar.set_description('Epoch: [{:3d}/{:3d}] Loss: {:.4f}'.format(self.epoch, self.opt['train']['n_epoch'], loss.item()))

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()

    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        pbar = tqdm.tqdm(self.val_loader[self.epoch % len(self.val_loader)])
        self.task = self.opt['datasets'][self.opt['phase']]['which_dataset'][self.epoch % len(self.val_loader)]['task']

        with torch.no_grad():
            for val_data in pbar:
                self.set_input(val_data)
                if self.opt['distributed']:
                    self.output = self.netG.module.restoration(self.cond_image, y_0=self.gt_image, task=self.task)
                else:
                    self.output = self.netG.restoration(self.cond_image, y_0=self.gt_image, task=self.task)

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='val',task=self.task).items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results(self.task), norm=self.opt['norm'])

        return self.val_metrics.result()
    
    def model_test(self, geometric_transform=0):
        # 8 geometric transforms: none, vflip, hflip, rotate, vflip_rotate, hflip_rotate, vflip_hflip, hflip_vflip_rotate
        operators = [   lambda x: x, 
                        F.vflip, 
                        F.hflip, 
                        lambda x: torch.rot90(x, 1, [-1, -2]),
                        lambda x: F.vflip(torch.rot90(x, 1, [-1, -2])),
                        lambda x: F.hflip(torch.rot90(x, 1, [-1, -2])),
                        lambda x: F.vflip(F.hflip(x)),
                        lambda x: F.hflip(F.vflip(torch.rot90(x, 1, [-1, -2])))]
        
        self.cond_image = operators[geometric_transform % 8](self.cond_image)

        if self.opt['distributed']:
            output = self.netG.module.restoration(self.cond_image, task=self.task)
        else:
            output = self.netG.restoration(self.cond_image, task=self.task)
        
        operators_inv = [   lambda x: x, 
                            F.vflip, 
                            F.hflip, 
                            lambda x: torch.rot90(x, -1, [-1, -2]),
                            lambda x: torch.rot90(F.vflip(x), -1, [-1, -2]),
                            lambda x: torch.rot90(F.hflip(x), -1, [-1, -2]),
                            lambda x: F.hflip(F.vflip(x)),
                            lambda x: torch.rot90(F.vflip(F.hflip(x)), -1, [-1, -2])]
        return operators_inv[geometric_transform % 8](output)

    def test(self):
        self.netG.eval()
        self.test_metrics.reset()
        pbar = tqdm.tqdm(self.phase_loader[0])
        self.task = self.opt['datasets'][self.opt['phase']]['which_dataset'][0]['task']

        print('test at rank {} task {}'.format(self.opt['local_rank'], self.task))
        with torch.no_grad():
            for phase_data in pbar:
                self.set_input(phase_data)
                self.outputs = []
                for i in range(self.mean):
                    output = self.model_test(i)
                    self.outputs.append(output)
                if self.mean > 1:
                    self.output = torch.stack(self.outputs, dim=0).mean(dim=0)
                    self.model_uncertainty = torch.stack(self.outputs, dim=0).std(dim=0)
                else:
                    self.output = self.outputs[0]
                    self.model_uncertainty = torch.zeros_like(self.output)
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.test_metrics.update(key, value, n=len(self.output))
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='test',task=self.task).items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results(self.task), norm=self.opt['norm'])

        test_log = self.test_metrics.result()
        ''' save logged informations into log dict '''
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard '''
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        self.save_training_state()

    def load_network(self, network, network_label, strict=True):
        if self.opt['path']['resume_state'] is None:
            return
        self.logger.info('Beign loading pretrained model [{:s}] ...'.format(network_label))

        model_path = "{}_{}.pth".format(self.opt['path']['resume_state'], network_label)

        if not os.path.exists(model_path):
            self.logger.warning('Pretrained model in [{:s}] is not existed, Skip it'.format(model_path))
            return

        self.logger.info('Loading pretrained model from [{:s}] ...'.format(model_path))
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        network.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: Util.set_device(storage)),
                                strict=strict)

    def train(self):
        ## sanity check
        # if self.val_loader[self.epoch % len(self.val_loader)] is None:
        #     self.logger.warning('Validation stop where dataloader is None, Skip it.')
        # else:
        #     self.logger.info("\n\n\n------------------------------Validation Start------------------------------")
        #     val_log = self.val_step()
        #     for key, value in val_log.items():
        #         self.logger.info('{:5s}: {}\t'.format(str(key), value))
        #     self.logger.info("\n------------------------------Validation End------------------------------\n\n")

        while self.epoch <= self.opt['train']['n_epoch'] and self.iter <= self.opt['train']['n_iter']:
            train_log = self.train_step()

            ''' save logged informations into log dict '''
            print('epoch {}: training start'.format(self.epoch))
            train_log.update({'epoch': self.epoch, 'iters': self.iter})

            ''' print logged informations to the screen and tensorboard ''' 
            for key, value in train_log.items():
                self.logger.info('{:5s}: {}\t'.format(str(key), value))

            if self.epoch % self.opt['train']['save_checkpoint_epoch'] == 0:
                self.logger.info('Saving the self at the end of epoch {:.0f}'.format(self.epoch))
                self.save_everything()

            if self.epoch % self.opt['train']['val_epoch'] == 0:
                self.logger.info("\n\n\n------------------------------Validation Start------------------------------")
                if self.val_loader[self.epoch % len(self.val_loader)] is None:
                    self.logger.warning('Validation stop where dataloader is None, Skip it.')
                else:
                    val_log = self.val_step()
                    for key, value in val_log.items():
                        self.logger.info('{:5s}: {}\t'.format(str(key), value))
                self.logger.info("\n------------------------------Validation End------------------------------\n\n")

            self.epoch += 1
        
        self.logger.info('Number of Epochs or Iterations has reached the limit, End.')