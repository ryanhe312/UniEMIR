import torch

from models.UniEMIR.Unimodel import UniModel
from core.base_network import BaseNetwork

class Network(BaseNetwork):
    def __init__(self, unimodel, module_name='UniEMIR', **kwargs):
        super(Network, self).__init__(**kwargs)
        self.model = UniModel(**unimodel)

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    @torch.no_grad()
    def restoration(self, y_cond, y_0=None, mask=None, task=0):
        self.model.eval()
        self.model.task = task
        if mask is not None:
            return self.model(y_cond) * mask + y_0 * (1 - mask)
        else:
            return self.model(y_cond)
    
    @torch.cuda.amp.autocast()
    def forward(self, y_0, y_cond, mask=None, task=0):
        self.model.train()
        self.model.task = task
        if mask is not None:
            y_hat = self.model(y_cond) * mask + y_0 * (1 - mask)
        else:
            y_hat = self.model(y_cond)
        
        loss = self.loss_fn(y_hat, y_0)
        return loss
