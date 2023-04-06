import numpy as np
import torch
from torch import nn
from operator import mul
from functools import reduce
from torch import autograd
from pdb import set_trace as bp


__all__ = ['get_ntk', 'Linear_Region_Collector', 'curve_complexity']


def get_ntk(data, network, criterion=torch.nn.BCELoss(reduction='none'), train_mode=True):
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

    for i_b, batch in enumerate(data):
        inputs, targets = batch
        logit = network(inputs)
        # choose specific class for loss
        loss = criterion(logit, targets)
        for _idx in range(len(inputs)):
            # logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
            # use criterion to get gradient
            loss[_idx:_idx+1].backward(torch.ones_like(loss[_idx:_idx+1]), retain_graph= not (_idx == len(inputs)-1 and i_b == len(data)-1))
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
        activations_list = activations
        if not isinstance(activations, list):
            assert isinstance(activations, nn.Tensor)
            activations_list = [activations]
        n_relu = len(activations_list)
        n_batch = activations_list[0].size()[0]
        n_neuron = [act.size()[1] for act in activations_list]
        if self.activations is None:
            self.activations = [torch.zeros(self.n_samples, _n).cuda() for _n in n_neuron]
        for idx, act in enumerate(activations_list):
            self.activations[idx][self.ptr:self.ptr+n_batch] = torch.sign(act)  # after ReLU
        self.n_neuron = n_neuron
        self.ptr += n_batch

    @torch.no_grad()
    def calc_LR(self):
        n_LRs = []
        for activations in self.activations:
            _n_LR = self._calc_LR_layer(activations)
            n_LRs.append(_n_LR)
        del self.activations
        torch.cuda.empty_cache()
        # print(n_LRs)
        self.n_LR = reduce(mul, n_LRs, 1)


    @torch.no_grad()
    def _calc_LR_layer(self, activations):
        res = torch.matmul(activations.half(), (1-activations).T.half()) # each element in res: A * (1 - B)
        res += res.clone().T # make symmetric, each element in res: A * (1 - B) + (1 - A) * B, a non-zero element indicate a pair of two different linear regions
        res = 1 - torch.sign(res) # a non-zero element now indicate two linear regions are identical
        res = res.sum(1) # for each sample's linear region: how many identical regions from other samples
        res = 1. / res.float() # contribution of each redudant (repeated) linear region
        n_LR = res.sum().item() # sum of unique regions (by aggregating contribution of all regions)
        del res
        torch.cuda.empty_cache()
        return n_LR

    def getLinearReginCount(self):
        if self.n_LR == -1:
            self.calc_LR()
        return self.n_LR


class Linear_Region_Collector:
    def __init__(self, data, model, train_mode=True, mode='multiply'):
        self.models = []
        self.input_size = data[0][0].shape  # BLC
        self.sample_batch = len(data)
        # self.input_numel = reduce(mul, self.input_size, 1)
        self.interFeature = []
        self.data = data
        self.reinit(model)
        self.mode = mode

    def reinit(self, model):
        self.model = model
        self.register_hook(model)
        self.LRCount = LinearRegionCount(self.input_size[0]*self.sample_batch)
        # self.loader = iter(self.train_loader)
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
            feature_data = [f.view(input_data.size(0), -1) for f in self.interFeature]
            if self.mode == 'concat':
                feature_data = torch.cat(feature_data, 1)
            LRCount.update2D(feature_data)


def curve_complexity(data, network, criterion=torch.nn.BCELoss(reduction='none'), train_mode=True, need_graph=True, reduction='mean', differentiable=False):
    assert isinstance(data, list) # multiple batch of samples
    for _data in data:
        if isinstance(_data, list) or isinstance(_data, tuple):
            for _item in _data:
                _item.requires_grad_(True)
        else:
            _data.requires_grad_(True)
    LE = 0
    network = network.cuda()
    if train_mode:
        network.train()
    else:
        network.eval()
    network.zero_grad()
    _idx = 0
    for batch_data in data:
        X, Y = batch_data
        output = network(X)
        output = criterion(output, Y)
        output = output.reshape(output.size(0), -1)
        n, c = output.size()
        jacobs = []
        for coord in range(c):
            _gradients = autograd.grad(outputs=output[:, coord].sum(), inputs=[ X ], only_inputs=True, retain_graph=need_graph, create_graph=need_graph)
            if differentiable:
                jacobs.append(_gradients[0]) # select gradient of "theta"
            else:
                jacobs.append(_gradients[0].detach()) # select gradient of "theta"
        jacobs = torch.stack(jacobs, 0).reshape(n, -1)
        jacobs = jacobs.permute(1, 0)
        gE = torch.einsum('nd,nd->n', jacobs, jacobs).sqrt()
        LE += gE.sum()
        torch.cuda.empty_cache()
    if reduction == 'mean':
        return LE.item() / len(data) / len(batch_data)
    else:
        return LE.item()


# https://github.com/SamsungLabs/zero-cost-nas/blob/main/foresight/pruners/p_utils.py#L56
def get_layer_metric_array(net, metric):#, mode): 
    metric_array = []

    for layer in net.modules():
        # if mode=='channel' and hasattr(layer,'dont_ch_prune'):
        #     continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))
    
    return metric_array


# https://github.com/SamsungLabs/zero-cost-nas/blob/main/foresight/pruners/measures/synflow.py
def compute_synflow(data, net, criterion=torch.nn.BCELoss(reduction='mean'), train_mode=True):

    # device = inputs.device

    #convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    #convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net)

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    # https://github.com/SamsungLabs/zero-cost-nas/blob/main/foresight/pruners/predictive.py#L100
    def sum_arr(arr):
        sum = 0.
        for i in range(len(arr)):
            sum += torch.sum(arr[i])
        return sum.item()
    
    net.zero_grad()
    # net.double()
    for i, batch in enumerate(data):
        # # Compute gradients with input of 1s 
        # input_dim = list(inputs[0,:].shape)
        # inputs = torch.ones([1] + input_dim).double().to(device)
        # output = net.forward(inputs)
        # torch.sum(output).backward() 

        data,label = batch[0],batch[1]
        data,label = data.cuda(), label.cuda()

        logits = net(data)
        loss = criterion(logits, label)
        loss.backward()

    grads_abs = get_layer_metric_array(net, synflow)#, mode)

    # apply signs of all params
    nonlinearize(net, signs)

    # return grads_abs
    return sum_arr(grads_abs)


# https://github.com/SLDGroup/ZiCo/blob/main/ZeroShotProxy/compute_zico.py
def getgrad(model:torch.nn.Module, grad_dict:dict, step_iter=0):
    if step_iter==0:
        for name,mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                # print(mod.weight.grad.data.size())
                # print(mod.weight.data.size())
                grad_dict[name]=[mod.weight.grad.data.cpu().reshape(-1).numpy()]
    else:
        for name,mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                grad_dict[name].append(mod.weight.grad.data.cpu().reshape( -1).numpy())
    return grad_dict


def caculate_zico(grad_dict):
    allgrad_array = None
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname]= np.array(grad_dict[modname])
    # nsr_mean_sum = 0
    nsr_mean_sum_abs = 0
    # nsr_mean_avg = 0
    nsr_mean_avg_abs = 0
    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx])
        if tmpsum==0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
            nsr_mean_avg_abs += np.log(np.mean(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx]))
    return nsr_mean_sum_abs


def compute_zico(data, network, criterion=torch.nn.BCELoss(reduction='mean'), train_mode=True):
    grad_dict= {}
    network.train()

    network.cuda()
    for i, batch in enumerate(data):
        data,label = batch[0],batch[1]
        data,label = data.cuda(), label.cuda()

        logits = network(data)
        loss = criterion(logits, label)
        loss.backward()
        grad_dict = getgrad(network, grad_dict, i)
    res = caculate_zico(grad_dict)
    return res
