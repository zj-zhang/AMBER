import numpy as np
import torch
from torch import nn
from operator import mul
from functools import reduce
from pdb import set_trace as bp


__all__ = ['get_ntk', 'Linear_Region_Collector']


def get_ntk(inputs, targets, network, criterion=torch.nn.BCELoss(reduction='none'), train_mode=True):
    if not criterion:
        criterion = torch.nn.BCELoss(reduction='none')
    device = torch.cuda.current_device()
    if train_mode:
        network.train()
    else:
        network.eval()
    grads_x = [] # size: #training samples. grads of all W from each sample
    # device management should happen outside of get_ntk
    #inputs = torch.Tensor(inputs).cuda(device=device, non_blocking=True)
    #targets = torch.Tensor(targets).cuda(device=device, non_blocking=True)

    ch = 16
    for idx in np.arange(0, len(inputs), ch):
        logit = network(inputs[idx:idx+ch])
        # choose specific class for loss
        loss = criterion(logit, targets[idx:idx+ch])
        for _idx in range(len(inputs[idx:idx+ch])):
            # logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
            # use criterion to get gradient
            loss[_idx:_idx+1].backward(torch.ones_like(loss[_idx:_idx+1]), retain_graph=_idx < len(inputs[idx:idx+ch])-1)
            grad = []
            for name, W in network.named_parameters():
                if 'classifier' in name or 'fc' in name or 'out' in name: continue
                if 'weight' in name and W.grad is not None:
                    grad.append(W.grad.view(-1).detach())
            grads_x.append(torch.cat(grad, -1).detach())
            network.zero_grad()
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    _grads_x = torch.stack([item.cuda() for item in grads_x], 0)
    torch.cuda.empty_cache()
    ntk = torch.einsum('nc,mc->nm', [_grads_x, _grads_x])
    del _grads_x
    try:
        # eigenvalues, _ = torch.symeig(ntk)  # ascending
        eigenvalues, _ = torch.linalg.eigh(ntk, UPLO='L')
    except:
        ntk[ntk == float("Inf")] = 0
        ntk[ntk == 0] = ntk.max() # TODO avoid inf in ntk
        eigenvalues, _ = torch.linalg.eigh(ntk + ntk.mean().item() * torch.eye(ntk.shape[0]).cuda() * 1e-4, UPLO='L')  # ascending
    _cond = torch.div(eigenvalues[-1], eigenvalues[0])
    if torch.isnan(_cond):
        return -1, float(loss.mean().item) # bad gradients
    else:
        return _cond.item(), float(loss.mean().item())


class LinearRegionCount(object):
    """Computes and stores the average and current value"""
    def __init__(self, n_samples):
        self.ActPattern = {}
        self.n_LR = -1
        self.n_samples = n_samples
        self.ptr = 0
        self.activations = None

    @torch.no_grad()
    def update2D(self, activations):
        n_batch = activations.size()[0]
        n_neuron = activations.size()[1]
        self.n_neuron = n_neuron
        if self.activations is None:
            self.activations = torch.zeros(self.n_samples, n_neuron).cuda()
        self.activations[self.ptr:self.ptr+n_batch] = torch.sign(activations)  # after ReLU
        self.ptr += n_batch

    @torch.no_grad()
    def calc_LR(self):
        res = torch.matmul(self.activations.half(), (1-self.activations).T.half()) # each element in res: A * (1 - B)
        res += res.T # make symmetric, each element in res: A * (1 - B) + (1 - A) * B, a non-zero element indicate a pair of two different linear regions
        res = 1 - torch.sign(res) # a non-zero element now indicate two linear regions are identical
        res = res.sum(1) # for each sample's linear region: how many identical regions from other samples
        res = 1. / res.float() # contribution of each redudant (repeated) linear region
        self.n_LR = res.sum().item() # sum of unique regions (by aggregating contribution of all regions)
        del self.activations, res
        self.activations = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def update1D(self, activationList):
        code_string = ''
        for key, value in activationList.items():
            n_neuron = value.size()[0]
            for i in range(n_neuron):
                if value[i] > 0:
                    code_string += '1'
                else:
                    code_string += '0'
        if code_string not in self.ActPattern:
            self.ActPattern[code_string] = 1

    def getLinearReginCount(self):
        if self.n_LR == -1:
            self.calc_LR()
        return self.n_LR


class Linear_Region_Collector:
    def __init__(self, data, model, train_mode=True):
        self.models = []
        self.input_size = data[0][0].shape  # BLC
        self.sample_batch = len(data)
        # self.input_numel = reduce(mul, self.input_size, 1)
        self.interFeature = []
        self.data = data
        self.reinit(model)

    def reinit(self, model):
        self.model = model
        self.register_hook(model)
        self.LRCount = LinearRegionCount(self.input_size[0]*self.sample_batch)
        self.loader = iter(self.train_loader)
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def clear(self):
        self.LRCount = LinearRegionCount(self.input_size[0]*self.sample_batch)
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def register_hook(self, model):
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(hook=self.hook_in_forward)

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 3:
            self.interFeature.append(output.detach())  # for ReLU

    def forward_batch_sample(self):
        # for idx in range(self.sample_batch):
            # inputs, targets = self.loader.next()
        for inputs, targets in self.data:
            self.forward(self.model, self.LRCount, inputs)
        return self.LRCount.getLinearReginCount()

    def forward(self, model, LRCount, input_data):
        self.interFeature = []
        with torch.no_grad():
            model.forward(input_data.cuda())
            if len(self.interFeature) == 0: return
            feature_data = torch.cat([f.view(input_data.size(0), -1) for f in self.interFeature], 1)
            LRCount.update2D(feature_data)
